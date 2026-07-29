"""Segmentation trainer.

`BaseTrainer` already branches on `config['task']` for the forward pass
(cls_scores/bbox_preds vs. seg_logits) and loss construction, so
`SegmentationTrainer` is currently a thin subclass -- it exists for
symmetry with the reference
`planning_training_pipeline/tasks/{task}/trainer.py` layout and as the
extension point for future segmentation-only training hooks without
touching the shared `BaseTrainer`.
"""

from photon2perception.engine.base_trainer import BaseTrainer


class SegmentationTrainer(BaseTrainer):
    """Trainer for the segmentation task (`config['task'] == 'segmentation'`)."""
