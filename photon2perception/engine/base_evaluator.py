"""
`BaseEvaluator`: task-agnostic evaluation engine for Photon2Perception models.

Class-based home for what used to be free functions in `tools/eval.py`
(model/checkpoint loading, dataloader construction, the efficiency
benchmark). Task-specific metric computation (COCO mAP vs. mIoU) is left to
`tasks/{detection,segmentation}/evaluator.py` subclasses via the abstract
`compute_task_metrics()` method -- mirrors the reference
`planning_training_pipeline/.../base_evaluator.py` +
`{task}/evaluator.py` split.
"""

from pathlib import Path
from typing import Any, Dict

import torch

from ..common.config import ConfigDict
from ..common.evaluation.efficiency import full_efficiency_report
from ..models.model_wrapper import build_perception_model
from ..utils.checkpoint import load_weights_only
from ..utils.distributed import DistributedInfo


def single_process_dist_info(device: torch.device) -> DistributedInfo:
    return DistributedInfo(rank=0, world_size=1, local_rank=0, is_distributed=False, device=device)


class BaseEvaluator:
    """Loads a checkpoint, builds the val dataloader, and runs task metrics
    plus an optional efficiency report.

    Subclasses (`tasks/detection/evaluator.py::DetectionTaskEvaluator`,
    `tasks/segmentation/evaluator.py::SegmentationTaskEvaluator`) must
    implement `compute_task_metrics()`.
    """

    def __init__(
        self,
        config: ConfigDict,
        checkpoint: str,
        device: torch.device = None,
    ):
        self.config = config
        self.task = config['task']
        self.checkpoint = checkpoint
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        )

        self.model = build_perception_model(config).to(self.device)
        load_weights_only(checkpoint, self.model, map_location='cpu', strict=True)
        self.model.eval()

        self.dist_info = single_process_dist_info(self.device)
        self.train_loader, self.val_loader = self.build_dataloaders()
        if self.val_loader is None:
            raise ValueError(
                "No validation set configured (data.val_img_dir/val_ann_file for detection, "
                "or the 'val' split for segmentation). Cannot compute task metrics."
            )

    def build_dataloaders(self):
        """Reuses `BaseTrainer.build_dataloaders` so eval always sees
        exactly the same dataset construction logic as training.
        """
        from .base_trainer import BaseTrainer
        # A bare BaseTrainer.__new__ + attribute set avoids re-running the
        # full training-only setup (model/optimizer/scheduler/logger) just
        # to reuse the dataloader-building method.
        stub = BaseTrainer.__new__(BaseTrainer)
        stub.config = self.config
        stub.task = self.task
        stub.dist_info = self.dist_info
        return stub.build_dataloaders()

    def compute_task_metrics(self) -> Dict[str, Any]:
        """Task-specific metric computation. Must be overridden by subclasses."""
        raise NotImplementedError

    def run_efficiency_benchmark(self) -> Dict[str, Any]:
        img_size = tuple(self.config['model']['img_size'])
        input_shape = (1, self.config['model'].get('in_chans', 1), *img_size)
        return full_efficiency_report(
            self.model, input_shape, input_format='bayer', device=str(self.device)
        )

    def evaluate(self, skip_efficiency: bool = False) -> Dict[str, Any]:
        task_metrics = self.compute_task_metrics()

        results = {
            'config': self.config,
            'checkpoint': self.checkpoint,
            'task_metrics': task_metrics,
        }
        if not skip_efficiency:
            results['efficiency'] = self.run_efficiency_benchmark()
        return results
