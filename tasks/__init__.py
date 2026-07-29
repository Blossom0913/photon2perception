"""Per-task training pipelines.

Mirrors the reference `planning_training_pipeline/tasks/{task_name}/` layout:
each subdirectory here (`detection/`, `segmentation/`) is a self-contained
task unit with its own `config/`, `model.py`, `trainer.py`, `evaluator.py`,
`visualizer.py`, and `scripts/`, built on top of the shared infrastructure in
`photon2perception.common` and `photon2perception.engine`.
"""
