"""Segmentation evaluator: mIoU (+ efficiency report via `BaseEvaluator`)."""

from typing import Any, Dict

import torch

from photon2perception.common.evaluation.metrics import SegmentationEvaluator
from photon2perception.engine.base_evaluator import BaseEvaluator


class SegmentationTaskEvaluator(BaseEvaluator):
    """Evaluator for the segmentation task (`config['task'] == 'segmentation'`).

    `compute_task_metrics()` runs the val dataloader through the model,
    argmaxes the per-pixel logits, and accumulates mIoU via
    `SegmentationEvaluator`.
    """

    @torch.no_grad()
    def compute_task_metrics(self) -> Dict[str, Any]:
        num_classes = self.config['data']['num_classes']
        evaluator = SegmentationEvaluator(num_classes=num_classes, ignore_index=255)

        self.model.eval()
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            seg_logits = self.model(images)
            preds = seg_logits.argmax(dim=1)
            evaluator.update(preds, targets)

        return evaluator.compute()
