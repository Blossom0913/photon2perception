"""Segmentation model construction.

Thin, task-scoped wrapper around `build_perception_model` -- kept here
(rather than calling `build_perception_model` directly from
`tasks/segmentation/trainer.py`/`evaluator.py`) purely for symmetry with the
reference `planning_training_pipeline/tasks/{task}/model.py` layout, and as
a single place to hang future segmentation-only model construction logic
without touching the shared `photon2perception.models.model_wrapper` used
by all tasks.

Per CLAUDE.md: always go through `build_perception_model(config)` rather
than hand-assembling backbone+neck+head.
"""

from photon2perception.models.model_wrapper import PerceptionModel, build_perception_model


def build_segmentation_model(config) -> PerceptionModel:
    """Build a segmentation `PerceptionModel` from a resolved experiment config.

    `config['task']` must be `'segmentation'`.
    """
    if config['task'] != 'segmentation':
        raise ValueError(f"build_segmentation_model requires task='segmentation', got '{config['task']}'")
    return build_perception_model(config)
