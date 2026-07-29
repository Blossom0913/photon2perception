"""Task-agnostic training/evaluation engine.

Mirrors the reference `planning_training_pipeline/planning_base_trainer.py`
pattern: a single `BaseTrainer`/`BaseEvaluator` pair implements everything
that does not vary by task (dataloader construction, optimizer/scheduler,
the train/validate loop, checkpointing, efficiency benchmarking), while
`tasks/{detection,segmentation}/trainer.py` and `evaluator.py` provide thin
task-specific subclasses (and `tasks/{detection,segmentation}/model.py`
config-to-model glue). `tools/train.py` and `tools/eval.py` are now thin CLI
entry points that just parse args, pick the right task subclass from
`config['task']`, and call `.fit()` / `.evaluate()`.
"""

from .base_evaluator import BaseEvaluator
from .base_trainer import BaseTrainer

__all__ = ['BaseTrainer', 'BaseEvaluator']
