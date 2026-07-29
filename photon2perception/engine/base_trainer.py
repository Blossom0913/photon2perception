"""
`BaseTrainer`: task-agnostic training engine for Photon2Perception models.

This is the class-based home for what used to be free functions in
`tools/train.py` (`build_dataloaders`, `build_loss`, `build_optimizer`,
`build_scheduler`, `train_one_epoch`, `validate_one_epoch`, and the epoch
loop in `main()`). `tools/train.py` is now a thin CLI wrapper that
instantiates `BaseTrainer` (or a `tasks/{task}/trainer.py` subclass) and
calls `.fit()`.

Supports the same features as before:
- Single-GPU, multi-GPU (DDP), and CPU/MPS (Apple Silicon) training.
- Config-based model/dataset/optimizer specification with `_base_`
  inheritance and CLI dotted-key overrides (photon2perception.common.config).
- Checkpoint save/resume, including auto-resume from the last checkpoint
  in `output_dir` (useful for preemptible/spot AutoDL instances).
- Gradient clipping, warmup+cosine/poly LR schedules.
- Mixed-precision training via `torch.autocast` + `GradScaler`.
- Unified logging via `photon2perception.utils.logger.ExperimentLogger`.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from ..common.config import ConfigDict, save_config
from ..common.dataset.base_raw_dataset import BaseRAWDataset
from ..common.dataset.coco_raw_dataset import (
    CityscapesRawSegmentationDataset,
    CocoRawDetectionDataset,
    detection_collate_fn,
    segmentation_collate_fn,
)
from ..common.dataset.exported_feature_dataset import try_build_preexported_dataset
from ..common.loss.detection_loss import DetectionLoss
from ..common.loss.segmentation_loss import SegmentationLoss
from ..models.model_wrapper import build_perception_model
from ..utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..utils.distributed import DistributedInfo, cleanup_distributed, reduce_dict, setup_distributed
from ..utils.logger import ExperimentLogger, MetricTracker


def move_targets_to_device(targets, task, device):
    if task == 'detection':
        return [{k: v.to(device) for k, v in t.items()} for t in targets]
    return targets.to(device)


def _default_strides(scale_factors, patch_stride):
    return [int(round(patch_stride / sf)) for sf in scale_factors]


class BaseTrainer:
    """Task-agnostic trainer: `detection`/`segmentation` forward-pass
    branching lives inline (both tasks share the exact same dataloader,
    optimizer, scheduler, checkpointing, and logging machinery), so
    `tasks/detection/trainer.py::DetectionTrainer` and
    `tasks/segmentation/trainer.py::SegmentationTrainer` are typically thin
    subclasses that exist mainly for symmetry with the reference
    `planning_training_pipeline/tasks/{task}/trainer.py` layout and as an
    extension point for future task-specific hooks.

    Args:
        config: Resolved experiment config (see `photon2perception.common.config`).
        output_dir: Directory to write checkpoints/logs/resolved_config.yaml to
            (already includes the experiment-name subdirectory).
        seed: Base random seed (already offset by `dist_info.rank` by the caller).
        resume: Explicit checkpoint path to resume from, if any.
        auto_resume: If True and `resume` is None, resume from the latest
            checkpoint found in `output_dir`, if any.
        use_tensorboard / use_wandb / wandb_project: Logging backends,
            normally derived from `config['logging']` with CLI overrides
            applied by the caller (see `tools/train.py::parse_args`).
    """

    def __init__(
        self,
        config: ConfigDict,
        output_dir: str,
        seed: int = 42,
        resume: Optional[str] = None,
        auto_resume: bool = False,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = 'photon2perception',
    ):
        self.config = config
        self.task = config['task']
        self.output_dir = Path(output_dir)

        self.dist_info: DistributedInfo = setup_distributed()
        self.device = self.dist_info.device
        self._set_seed(seed + self.dist_info.rank)

        if self.dist_info.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, str(self.output_dir / 'resolved_config.yaml'))

        self.logger = ExperimentLogger(
            log_dir=str(self.output_dir),
            exp_name=self.output_dir.name,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            config=dict(config),
            rank=self.dist_info.rank,
        )
        self.logger.log(f"Device: {self.device} | world_size={self.dist_info.world_size} "
                         f"| rank={self.dist_info.rank}")

        self.model = build_perception_model(config).to(self.device)
        if self.dist_info.is_main_process:
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.log(f"Model params: {num_params / 1e6:.2f}M | task={self.task}")

        # Keep an un-DDP-wrapped handle for checkpointing/eval.
        self.raw_model = self.model
        if self.dist_info.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.dist_info.local_rank] if self.device.type == 'cuda' else None,
            )

        self.train_loader, self.val_loader = self.build_dataloaders()
        self.logger.log(
            f"Train batches/epoch: {len(self.train_loader)} | "
            f"Val batches/epoch: {len(self.val_loader) if self.val_loader else 0}"
        )

        self.loss_fn = self.build_loss().to(self.device)
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(steps_per_epoch=len(self.train_loader))

        train_cfg = config['training']
        self.mixed_precision = train_cfg.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision) if self.mixed_precision else None
        self.grad_clip = train_cfg.get('grad_clip', 0.0)

        self.total_epochs = train_cfg['epochs']
        self.save_interval = train_cfg.get('save_interval', 5)
        self.val_interval = train_cfg.get('val_interval', 1)
        self.log_interval = train_cfg.get('log_interval', 10)

        logging_cfg = config.get('logging', {})
        self.log_images = logging_cfg.get('log_images', False)
        self.image_log_interval = logging_cfg.get('image_log_interval', 200)

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric: Optional[float] = None
        self._maybe_resume(resume, auto_resume)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        from ..utils.distributed import set_seed
        set_seed(seed)

    def _maybe_resume(self, resume: Optional[str], auto_resume: bool) -> None:
        resume_path = resume
        if resume_path is None and auto_resume:
            resume_path = find_latest_checkpoint(str(self.output_dir))
        if resume_path:
            ckpt = load_checkpoint(
                resume_path, model=self.raw_model, optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, map_location='cpu', strict=True,
            )
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.logger.log(f"Resumed from {resume_path} at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def build_dataloaders(self):
        """Build train and validation dataloaders based on `config['data']['type']`.

        Supported `data.type` values:
            'coco'       : real COCO detection annotations -> synthetic Bayer RAW
                           (CocoRawDetectionDataset)
            'cityscapes' : Cityscapes segmentation -> synthetic Bayer RAW
                           (CityscapesRawSegmentationDataset)
            'real'       : a real RAW dataset directory (BaseRAWDataset subclass)
            'synthetic'  : deprecated alias for 'coco'/'cityscapes' depending on task,
                           kept for backward compatibility with existing YAML configs.
        """
        config = self.config
        data_cfg = config['data']
        task = self.task
        dataset_type = data_cfg.get('type', 'coco' if task == 'detection' else 'cityscapes')
        if dataset_type == 'synthetic':
            dataset_type = 'coco' if task == 'detection' else 'cityscapes'

        img_size = tuple(data_cfg.get('img_scale', config['model']['img_size']))
        collate_fn = detection_collate_fn if task == 'detection' else segmentation_collate_fn

        # Prefer pre-exported feature shards (tools/export_features.py /
        # scripts/local_feature_exporter.sh) over the live on-the-fly dataset
        # when `feature_export.enabled: true` AND a manifest is actually
        # present on disk for a given split. Falls back silently to the live
        # dataset for a split with no exported manifest.
        feature_export_cfg = config.get('feature_export', {})
        pre_train = try_build_preexported_dataset(feature_export_cfg, split='train')
        pre_val = try_build_preexported_dataset(feature_export_cfg, split='val')
        if pre_train is not None or pre_val is not None:
            train_loader = self._make_dataloader(
                pre_train, data_cfg, shuffle=True, collate_fn=collate_fn,
            ) if pre_train is not None else None
            val_loader = self._make_dataloader(
                pre_val, data_cfg, shuffle=False, collate_fn=collate_fn,
            ) if pre_val is not None else None
            if train_loader is not None:
                return train_loader, val_loader

        if dataset_type == 'coco':
            train_dataset = CocoRawDetectionDataset(
                root_dir=data_cfg['train_img_dir'],
                ann_file=data_cfg['train_ann_file'],
                img_size=img_size,
                cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
                normalize=data_cfg.get('normalize', True),
            )
            val_dataset = None
            if data_cfg.get('val_img_dir') and data_cfg.get('val_ann_file'):
                val_dataset = CocoRawDetectionDataset(
                    root_dir=data_cfg['val_img_dir'],
                    ann_file=data_cfg['val_ann_file'],
                    img_size=img_size,
                    cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
                    normalize=data_cfg.get('normalize', True),
                )
        elif dataset_type == 'cityscapes':
            train_dataset = CityscapesRawSegmentationDataset(
                root_dir=data_cfg['root_dir'],
                split='train',
                img_size=img_size,
                cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
                normalize=data_cfg.get('normalize', True),
                num_classes=data_cfg['num_classes'],
            )
            val_dataset = CityscapesRawSegmentationDataset(
                root_dir=data_cfg['root_dir'],
                split='val',
                img_size=img_size,
                cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
                normalize=data_cfg.get('normalize', True),
                num_classes=data_cfg['num_classes'],
            )
        elif dataset_type == 'real':
            train_dataset = BaseRAWDataset(root_dir=data_cfg['root_dir'], split='train')
            val_dataset = BaseRAWDataset(root_dir=data_cfg['root_dir'], split='val')
        else:
            raise ValueError(f"Unknown data.type '{dataset_type}'")

        train_loader = self._make_dataloader(train_dataset, data_cfg, shuffle=True, collate_fn=collate_fn)
        val_loader = self._make_dataloader(val_dataset, data_cfg, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader

    def _make_dataloader(self, dataset, data_cfg, shuffle: bool, collate_fn):
        """Shared `DataLoader` construction for both the live on-the-fly
        datasets and `PreExportedFeatureDataset` -- identical batching/sampling
        behavior regardless of which dataset backend is in play.
        """
        if dataset is None:
            return None
        dist_info = self.dist_info
        if dist_info.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=shuffle
            )
            use_shuffle = False
        else:
            sampler = None
            use_shuffle = shuffle
        return DataLoader(
            dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=use_shuffle,
            sampler=sampler,
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    # ------------------------------------------------------------------
    # Loss / optimizer / scheduler
    # ------------------------------------------------------------------

    def build_loss(self) -> nn.Module:
        config = self.config
        task = self.task
        loss_cfg = config.get('loss', {})
        num_classes = config['data']['num_classes']

        if task == 'detection':
            strides = config['model'].get(
                'neck_strides',
                _default_strides(config['model'].get('neck_scale_factors', (4.0, 2.0, 1.0, 0.5)),
                                  config['model']['patch_size']),
            )
            return DetectionLoss(
                num_classes=num_classes,
                strides=tuple(strides),
                cls_weight=loss_cfg.get('cls_weight', 2.0),
                reg_weight=loss_cfg.get('reg_weight', 1.0),
                reg_loss_type=loss_cfg.get('reg_loss', 'l1'),
                focal_alpha=loss_cfg.get('focal_alpha', 0.25),
                focal_gamma=loss_cfg.get('focal_gamma', 2.0),
            )
        else:
            return SegmentationLoss(
                num_classes=num_classes,
                ce_weight=loss_cfg.get('ce_weight', 1.0),
                rmi_weight=loss_cfg.get('rmi_loss_weight', 0.1),
                aux_weight=loss_cfg.get('aux_loss_weight', 0.0),
            )

    def build_optimizer(self):
        train_cfg = self.config['training']
        return AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )

    def build_scheduler(self, steps_per_epoch: int):
        train_cfg = self.config['training']
        warmup_epochs = train_cfg.get('warmup_epochs', 5)
        total_epochs = train_cfg['epochs']
        warmup_iters = max(warmup_epochs * steps_per_epoch, 1)
        total_iters = max(total_epochs * steps_per_epoch, warmup_iters + 1)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_iters)

        schedule_type = train_cfg.get('lr_schedule', 'cosine')
        if schedule_type == 'cosine':
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_iters - warmup_iters, 1))
        elif schedule_type == 'poly':
            power = train_cfg.get('lr_power', 0.9)
            remaining = max(total_iters - warmup_iters, 1)

            def poly_lambda(step):
                return max(1.0 - step / remaining, 0.0) ** power

            main_scheduler = LambdaLR(self.optimizer, lr_lambda=poly_lambda)
        elif schedule_type == 'constant':
            main_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        else:
            raise ValueError(f"Unknown lr_schedule '{schedule_type}'")

        return SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters]
        )

    # ------------------------------------------------------------------
    # Train / validate one epoch
    # ------------------------------------------------------------------

    def log_input_sample(self, images: torch.Tensor, step: int, tag: str = 'train/input_sample') -> None:
        """Log a single Bayer RAW input (first sample of the batch) as a
        grayscale image to TensorBoard/W&B, for at-a-glance sanity checking
        (correct CFA tiling, normalization range, augmentation not
        degenerate, etc.) alongside the scalar loss curves.
        """
        # images are normalized to [-1, 1] by the dataset pipeline; undo that
        # for display.
        sample = images[0].detach().float().cpu()
        sample = (sample.clamp(-1, 1) + 1.0) / 2.0  # -> [0, 1]
        image_hwc = sample.permute(1, 2, 0).numpy()
        self.logger.log_image(tag, image_hwc, step)

    def train_one_epoch(self, epoch: int):
        model, dataloader = self.model, self.train_loader
        model.train()
        tracker = MetricTracker()
        use_amp = self.scaler is not None
        dist_info = self.logger and self.dist_info

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device, non_blocking=True)
            targets = move_targets_to_device(batch['targets'], self.task, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=use_amp,
                                 dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                if self.task == 'detection':
                    cls_scores, bbox_preds = model(images)
                    loss_dict = self.loss_fn(cls_scores, bbox_preds, targets)
                else:
                    seg_logits = model(images)
                    loss_dict = self.loss_fn(seg_logits, targets)
                loss = loss_dict['loss_total']

            if use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            scalar_loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            scalar_loss_dict = reduce_dict(scalar_loss_dict)
            tracker.update(scalar_loss_dict)

            if self.dist_info.is_main_process and batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_scalars({**scalar_loss_dict, 'lr': lr}, step=self.global_step, prefix='train/')
                self.logger.log(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                                 f"loss_total={scalar_loss_dict['loss_total']:.4f} | lr={lr:.6f}")

            if self.dist_info.is_main_process and self.log_images and self.global_step % self.image_log_interval == 0:
                self.log_input_sample(images, step=self.global_step)

            self.global_step += 1

        return tracker.averages()

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):
        dataloader = self.val_loader
        if dataloader is None:
            return {}
        model = self.model
        model.eval()
        tracker = MetricTracker()

        for batch in dataloader:
            images = batch['image'].to(self.device, non_blocking=True)
            targets = move_targets_to_device(batch['targets'], self.task, self.device)

            if self.task == 'detection':
                cls_scores, bbox_preds = model(images)
                loss_dict = self.loss_fn(cls_scores, bbox_preds, targets)
            else:
                seg_logits = model(images)
                loss_dict = self.loss_fn(seg_logits, targets)

            scalar_loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            scalar_loss_dict = reduce_dict(scalar_loss_dict)
            tracker.update(scalar_loss_dict)

        averages = tracker.averages()
        if self.dist_info.is_main_process:
            self.logger.log_scalars(averages, step=epoch, prefix='val/')
            self.logger.log(f"[val] Epoch {epoch} | " + ' '.join(f"{k}={v:.4f}" for k, v in averages.items()))
        return averages

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop: train/validate/checkpoint per epoch,
        for `config['training']['epochs']` epochs starting from
        `self.start_epoch` (0, unless resumed).
        """
        for epoch in range(self.start_epoch, self.total_epochs):
            if self.dist_info.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(epoch)
            if self.dist_info.is_main_process:
                self.logger.log(f"Epoch {epoch} done | " + ' '.join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

            if self.val_loader is not None and (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate_one_epoch(epoch)
                current_metric = val_metrics.get('loss_total')
                if self.dist_info.is_main_process and current_metric is not None:
                    if self.best_metric is None or current_metric < self.best_metric:
                        self.best_metric = current_metric
                        save_checkpoint(
                            str(self.output_dir / 'checkpoint_best.pth'), self.raw_model, self.optimizer,
                            self.scheduler, self.scaler,
                            epoch=epoch, global_step=self.global_step, best_metric=self.best_metric,
                            config=dict(self.config),
                        )

            if self.dist_info.is_main_process and (epoch + 1) % self.save_interval == 0:
                save_checkpoint(
                    str(self.output_dir / f'checkpoint_epoch_{epoch:04d}.pth'), self.raw_model, self.optimizer,
                    self.scheduler, self.scaler,
                    epoch=epoch, global_step=self.global_step, best_metric=self.best_metric,
                    config=dict(self.config),
                )

        if self.dist_info.is_main_process:
            save_checkpoint(
                str(self.output_dir / 'checkpoint_last.pth'), self.raw_model, self.optimizer,
                self.scheduler, self.scaler,
                epoch=self.total_epochs - 1, global_step=self.global_step, best_metric=self.best_metric,
                config=dict(self.config),
            )
            self.logger.log("Training complete.")
        self.logger.close()
        cleanup_distributed()
