"""Detection trainer.

`BaseTrainer` already branches on `config['task']` for the forward pass
(cls_scores/bbox_preds vs. seg_logits) and loss construction, so
`DetectionTrainer` is currently a thin subclass -- it exists for symmetry
with the reference `planning_training_pipeline/tasks/{task}/trainer.py`
layout and as the extension point for future detection-only training hooks
(e.g. detection-specific LR warmup, anchor refinement schedules) without
touching the shared `BaseTrainer`.
"""

from photon2perception.engine.base_trainer import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """Trainer for the detection task (`config['task'] == 'detection'`)."""
