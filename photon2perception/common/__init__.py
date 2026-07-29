"""Shared task-agnostic infrastructure: config loading, dataset loaders,
losses, task heads, and evaluation metrics/benchmarks.

Mirrors the `common/`, `dataset/`, `loss/`, `head/`, `evaluation/` layout of
the reference `planning_training_pipeline` structure -- everything here is
reused by both the `detection` and `segmentation` tasks under `tasks/`.
"""
