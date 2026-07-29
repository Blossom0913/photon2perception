"""Detection model construction.

Thin, task-scoped wrapper around `build_perception_model` -- kept here
(rather than calling `build_perception_model` directly from
`tasks/detection/trainer.py`/`evaluator.py`) purely for symmetry with the
reference `planning_training_pipeline/tasks/{task}/model.py` layout, and as
a single place to hang future detection-only model construction logic
(e.g. task-specific weight init) without touching the shared
`photon2perception.models.model_wrapper` used by all tasks.

Per CLAUDE.md: always go through `build_perception_model(config)` rather
than hand-assembling backbone+neck+head.
"""

from photon2perception.models.model_wrapper import PerceptionModel, build_perception_model


def build_detection_model(config) -> PerceptionModel:
    """Build a detection `PerceptionModel` from a resolved experiment config.

    `config['task']` must be `'detection'`.
    """
    if config['task'] != 'detection':
        raise ValueError(f"build_detection_model requires task='detection', got '{config['task']}'")
    return build_perception_model(config)
