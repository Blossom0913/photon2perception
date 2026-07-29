"""Detection evaluator: COCO-style mAP (+ efficiency report via `BaseEvaluator`)."""

from typing import Any, Dict

import torch

from photon2perception.common.evaluation.metrics import DetectionEvaluator
from photon2perception.common.head.postprocess import postprocess_detections
from photon2perception.engine.base_evaluator import BaseEvaluator


class DetectionTaskEvaluator(BaseEvaluator):
    """Evaluator for the detection task (`config['task'] == 'detection'`).

    `compute_task_metrics()` runs the val dataloader through the model,
    post-processes raw (cls_scores, bbox_preds) into boxes via NMS
    (`photon2perception.common.head.postprocess.postprocess_detections`),
    and accumulates COCO mAP via `DetectionEvaluator`.
    """

    def __init__(self, config, checkpoint: str, score_thresh: float = 0.05,
                 nms_thresh: float = 0.5, device: torch.device = None):
        super().__init__(config, checkpoint, device=device)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    @torch.no_grad()
    def compute_task_metrics(self) -> Dict[str, Any]:
        num_classes = self.config['data']['num_classes']
        strides = self.model.get_strides()
        evaluator = DetectionEvaluator(num_classes=num_classes)

        self.model.eval()
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            cls_scores, bbox_preds = self.model(images)
            preds = postprocess_detections(
                cls_scores, bbox_preds, strides=strides, num_classes=num_classes,
                score_thresh=self.score_thresh, nms_thresh=self.nms_thresh,
                image_size=tuple(images.shape[-2:]),
            )
            image_sizes = [tuple(images.shape[-2:])] * images.shape[0]
            evaluator.update(batch['image_id'], preds, batch['targets'], image_sizes=image_sizes)

        return evaluator.compute()
