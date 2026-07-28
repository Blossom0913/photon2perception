#!/usr/bin/env python3
"""
Training script for Photon2Perception models.

Fixes CLAUDE.md's "Known gaps" #1, #4, #5, #6:
1. Real detection/segmentation losses (photon2perception.losses), replacing
   the previous `loss = cls_token.sum() * 0.0` placeholder.
4. A real validation loop (build_dataloaders now returns a val_loader).
5. Mixed-precision training via `torch.autocast` + `GradScaler`, gated by
   `config['training']['mixed_precision']`.
6. Unified logging via `photon2perception.utils.logger.ExperimentLogger`
   (console + file + optional TensorBoard/W&B), replacing bare `print()`.

Supports:
- Single-GPU, multi-GPU (DDP), and CPU/MPS (Apple Silicon) training.
- Config-based model/dataset/optimizer specification with `_base_`
  inheritance and CLI dotted-key overrides (photon2perception.utils.config).
- Checkpoint save/resume, including auto-resume from the last checkpoint
  in `output_dir` (useful for preemptible/spot AutoDL instances).
- Gradient clipping, warmup+cosine/poly LR schedules.

Example:
    python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml
    python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml \\
        --override training.epochs=5 data.batch_size=2 --output_dir ./outputs/debug
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.datasets.base_raw_dataset import BaseRAWDataset
from photon2perception.datasets.coco_raw_dataset import (
    CityscapesRawSegmentationDataset,
    CocoRawDetectionDataset,
    detection_collate_fn,
    segmentation_collate_fn,
)
from photon2perception.datasets.exported_feature_dataset import try_build_preexported_dataset
from photon2perception.losses.detection_loss import DetectionLoss
from photon2perception.losses.segmentation_loss import SegmentationLoss
from photon2perception.models.model_wrapper import PerceptionModel, build_perception_model
from photon2perception.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from photon2perception.utils.config import apply_cli_overrides, load_config, save_config
from photon2perception.utils.distributed import (
    DistributedInfo,
    cleanup_distributed,
    reduce_dict,
    set_seed,
    setup_distributed,
)
from photon2perception.utils.logger import ExperimentLogger, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description='Train Photon2Perception')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from a specific checkpoint')
    parser.add_argument('--auto_resume', action='store_true',
                         help='Resume from the latest checkpoint in output_dir/exp_name, if any')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                         help='Experiment name (subdirectory of output_dir); defaults to the config filename stem')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default=None,
                         help='Enable Weights & Biases logging (overrides config.logging.use_wandb)')
    parser.add_argument('--no_tensorboard', action='store_true',
                         help='Disable TensorBoard logging (overrides config.logging.use_tensorboard)')
    parser.add_argument('--wandb_project', type=str, default=None,
                         help='Overrides config.logging.wandb_project')
    parser.add_argument(
        '--override', nargs='+', default=None,
        help="Dotted-key config overrides, e.g. training.epochs=5 data.batch_size=2",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------------

def build_dataloaders(config, dist_info: DistributedInfo):
    """Build train and validation dataloaders based on `config['data']['type']`.

    Supported `data.type` values:
        'coco'       : real COCO detection annotations -> synthetic Bayer RAW
                       (photon2perception.datasets.coco_raw_dataset.CocoRawDetectionDataset)
        'cityscapes' : Cityscapes segmentation -> synthetic Bayer RAW
                       (CityscapesRawSegmentationDataset)
        'real'       : a real RAW dataset directory (BaseRAWDataset subclass)
        'synthetic'  : deprecated alias for 'coco'/'cityscapes' depending on task,
                       kept for backward compatibility with existing YAML configs.
    """
    data_cfg = config['data']
    task = config['task']
    dataset_type = data_cfg.get('type', 'coco' if task == 'detection' else 'cityscapes')
    if dataset_type == 'synthetic':
        dataset_type = 'coco' if task == 'detection' else 'cityscapes'

    img_size = tuple(data_cfg.get('img_scale', config['model']['img_size']))
    collate_fn = detection_collate_fn if task == 'detection' else segmentation_collate_fn

    # Prefer pre-exported feature shards (tools/export_features.py /
    # scripts/local_feature_exporter.sh) over the live on-the-fly dataset
    # when `feature_export.enabled: true` AND a manifest is actually present
    # on disk for a given split -- this skips re-running resize + Bayer
    # unprocessing on every `__getitem__` call. Falls back silently to the
    # live dataset for a split with no exported manifest (e.g. only `train`
    # was exported), so partially-exported setups still work.
    feature_export_cfg = config.get('feature_export', {})
    pre_train = try_build_preexported_dataset(feature_export_cfg, split='train')
    pre_val = try_build_preexported_dataset(feature_export_cfg, split='val')
    if pre_train is not None or pre_val is not None:
        train_loader = _make_dataloader(
            pre_train, data_cfg, dist_info, shuffle=True, collate_fn=collate_fn,
        ) if pre_train is not None else None
        val_loader = _make_dataloader(
            pre_val, data_cfg, dist_info, shuffle=False, collate_fn=collate_fn,
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

    train_loader = _make_dataloader(train_dataset, data_cfg, dist_info, shuffle=True, collate_fn=collate_fn)
    val_loader = _make_dataloader(val_dataset, data_cfg, dist_info, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def _make_dataloader(dataset, data_cfg, dist_info: DistributedInfo, shuffle: bool, collate_fn):
    """Shared `DataLoader` construction for both the live on-the-fly
    datasets and `PreExportedFeatureDataset` -- identical batching/sampling
    behavior regardless of which dataset backend is in play.
    """
    if dataset is None:
        return None
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


# ----------------------------------------------------------------------------
# Loss construction
# ----------------------------------------------------------------------------

def build_loss(config):
    task = config['task']
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


def _default_strides(scale_factors, patch_stride):
    return [int(round(patch_stride / sf)) for sf in scale_factors]


# ----------------------------------------------------------------------------
# Optimizer / scheduler
# ----------------------------------------------------------------------------

def build_optimizer(model, config):
    train_cfg = config['training']
    return AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0001),
    )


def build_scheduler(optimizer, config, steps_per_epoch: int):
    train_cfg = config['training']
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    total_epochs = train_cfg['epochs']
    warmup_iters = max(warmup_epochs * steps_per_epoch, 1)
    total_iters = max(total_epochs * steps_per_epoch, warmup_iters + 1)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_iters)

    schedule_type = train_cfg.get('lr_schedule', 'cosine')
    if schedule_type == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_iters - warmup_iters, 1))
    elif schedule_type == 'poly':
        power = train_cfg.get('lr_power', 0.9)
        remaining = max(total_iters - warmup_iters, 1)

        def poly_lambda(step):
            return max(1.0 - step / remaining, 0.0) ** power

        main_scheduler = LambdaLR(optimizer, lr_lambda=poly_lambda)
    elif schedule_type == 'constant':
        main_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(f"Unknown lr_schedule '{schedule_type}'")

    return SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters]
    )


# ----------------------------------------------------------------------------
# Train / validate one epoch
# ----------------------------------------------------------------------------

def move_targets_to_device(targets, task, device):
    if task == 'detection':
        return [{k: v.to(device) for k, v in t.items()} for t in targets]
    return targets.to(device)


def log_input_sample(logger: ExperimentLogger, images: torch.Tensor, step: int, tag: str = 'train/input_sample') -> None:
    """Log a single Bayer RAW input (first sample of the batch) as a
    grayscale image to TensorBoard/W&B, for at-a-glance sanity checking of
    what the model actually sees (correct CFA tiling, normalization range,
    augmentation not degenerate, etc.) alongside the scalar loss curves.

    Gated by `config['logging']['log_images']` / `image_log_interval` (see
    `train_one_epoch`); a no-op if the logger has no active image backend.
    """
    # images are normalized to [-1, 1] by the dataset pipeline (see
    # BaseRAWDataset / CocoRawDetectionDataset); undo that for display.
    sample = images[0].detach().float().cpu()
    sample = (sample.clamp(-1, 1) + 1.0) / 2.0  # -> [0, 1]
    # ExperimentLogger.log_image expects HWC; a single-channel Bayer image
    # is (1, H, W) -> (H, W, 1), broadcastable as grayscale by most viewers.
    image_hwc = sample.permute(1, 2, 0).numpy()
    logger.log_image(tag, image_hwc, step)


def train_one_epoch(
    model, dataloader, optimizer, scheduler, loss_fn, task,
    device, epoch, dist_info, logger, scaler, grad_clip, log_interval, global_step,
    log_images: bool = False, image_log_interval: int = 200,
):
    model.train()
    tracker = MetricTracker()
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        targets = move_targets_to_device(batch['targets'], task, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp,
                             dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            if task == 'detection':
                cls_scores, bbox_preds = model(images)
                loss_dict = loss_fn(cls_scores, bbox_preds, targets)
            else:
                seg_logits = model(images)
                loss_dict = loss_fn(seg_logits, targets)
            loss = loss_dict['loss_total']

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        scalar_loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        scalar_loss_dict = reduce_dict(scalar_loss_dict)
        tracker.update(scalar_loss_dict)

        if dist_info.is_main_process and batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.log_scalars({**scalar_loss_dict, 'lr': lr}, step=global_step, prefix='train/')
            logger.log(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                       f"loss_total={scalar_loss_dict['loss_total']:.4f} | lr={lr:.6f}")

        if dist_info.is_main_process and log_images and global_step % image_log_interval == 0:
            log_input_sample(logger, images, step=global_step)

        global_step += 1

    return tracker.averages(), global_step


@torch.no_grad()
def validate_one_epoch(model, dataloader, loss_fn, task, device, epoch, dist_info, logger):
    if dataloader is None:
        return {}
    model.eval()
    tracker = MetricTracker()

    for batch in dataloader:
        images = batch['image'].to(device, non_blocking=True)
        targets = move_targets_to_device(batch['targets'], task, device)

        if task == 'detection':
            cls_scores, bbox_preds = model(images)
            loss_dict = loss_fn(cls_scores, bbox_preds, targets)
        else:
            seg_logits = model(images)
            loss_dict = loss_fn(seg_logits, targets)

        scalar_loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        scalar_loss_dict = reduce_dict(scalar_loss_dict)
        tracker.update(scalar_loss_dict)

    averages = tracker.averages()
    if dist_info.is_main_process:
        logger.log_scalars(averages, step=epoch, prefix='val/')
        logger.log(f"[val] Epoch {epoch} | " + ' '.join(f"{k}={v:.4f}" for k, v in averages.items()))
    return averages


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    dist_info = setup_distributed()
    set_seed(args.seed + dist_info.rank)
    device = dist_info.device

    exp_name = args.exp_name or Path(args.config).stem
    output_dir = Path(args.output_dir) / exp_name
    if dist_info.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, str(output_dir / 'resolved_config.yaml'))

    # `config['logging']` supplies defaults (so a config file alone fully
    # determines visualization behavior, e.g. for scripts/train_local.sh /
    # scripts/train_debug.sh); CLI flags (--use_wandb/--no_tensorboard/
    # --wandb_project) override those defaults when explicitly passed.
    logging_cfg = config.get('logging', {})
    use_tensorboard = logging_cfg.get('use_tensorboard', True) and not args.no_tensorboard
    use_wandb = args.use_wandb if args.use_wandb is not None else logging_cfg.get('use_wandb', False)
    wandb_project = args.wandb_project or logging_cfg.get('wandb_project', 'photon2perception')

    logger = ExperimentLogger(
        log_dir=str(output_dir),
        exp_name=exp_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        config=dict(config),
        rank=dist_info.rank,
    )
    logger.log(f"Config: {args.config}")
    logger.log(f"Device: {device} | world_size={dist_info.world_size} | rank={dist_info.rank}")

    # Model
    model = build_perception_model(config).to(device)
    if dist_info.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Model params: {num_params / 1e6:.2f}M | task={config['task']}")

    raw_model = model  # keep an un-DDP-wrapped handle for checkpointing/eval
    if dist_info.is_distributed:
        model = DDP(model, device_ids=[dist_info.local_rank] if device.type == 'cuda' else None)

    # Data
    train_loader, val_loader = build_dataloaders(config, dist_info)
    logger.log(f"Train batches/epoch: {len(train_loader)} | "
               f"Val batches/epoch: {len(val_loader) if val_loader else 0}")

    # Loss / optimizer / scheduler
    loss_fn = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    mixed_precision = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision) if mixed_precision else None
    grad_clip = config['training'].get('grad_clip', 0.0)

    # Resume
    start_epoch = 0
    global_step = 0
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(str(output_dir))
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=raw_model, optimizer=optimizer, scheduler=scheduler,
                                scaler=scaler, map_location='cpu', strict=True)
        start_epoch = ckpt.get('epoch', -1) + 1
        global_step = ckpt.get('global_step', 0)
        logger.log(f"Resumed from {resume_path} at epoch {start_epoch}")

    total_epochs = config['training']['epochs']
    save_interval = config['training'].get('save_interval', 5)
    val_interval = config['training'].get('val_interval', 1)
    log_interval = config['training'].get('log_interval', 10)
    task = config['task']

    best_metric = None
    for epoch in range(start_epoch, total_epochs):
        if dist_info.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, task, device,
            epoch, dist_info, logger, scaler, grad_clip, log_interval, global_step,
            log_images=logging_cfg.get('log_images', False),
            image_log_interval=logging_cfg.get('image_log_interval', 200),
        )
        if dist_info.is_main_process:
            logger.log(f"Epoch {epoch} done | " + ' '.join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            val_metrics = validate_one_epoch(model, val_loader, loss_fn, task, device, epoch, dist_info, logger)
            current_metric = val_metrics.get('loss_total')
            if dist_info.is_main_process and current_metric is not None:
                if best_metric is None or current_metric < best_metric:
                    best_metric = current_metric
                    save_checkpoint(
                        str(output_dir / 'checkpoint_best.pth'), raw_model, optimizer, scheduler, scaler,
                        epoch=epoch, global_step=global_step, best_metric=best_metric, config=dict(config),
                    )

        if dist_info.is_main_process and (epoch + 1) % save_interval == 0:
            save_checkpoint(
                str(output_dir / f'checkpoint_epoch_{epoch:04d}.pth'), raw_model, optimizer, scheduler, scaler,
                epoch=epoch, global_step=global_step, best_metric=best_metric, config=dict(config),
            )

    if dist_info.is_main_process:
        save_checkpoint(
            str(output_dir / 'checkpoint_last.pth'), raw_model, optimizer, scheduler, scaler,
            epoch=total_epochs - 1, global_step=global_step, best_metric=best_metric, config=dict(config),
        )
        logger.log("Training complete.")
    logger.close()
    cleanup_distributed()


if __name__ == '__main__':
    main()
