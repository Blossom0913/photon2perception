"""Segmentation visualization entry point.

Segmentation qualitative analysis reuses the shared, task-agnostic plotting
functions in `photon2perception.common.visualization` (token routing
heatmaps, attention maps, RAW-vs-RGB comparisons) -- re-exported here as the
per-task entry point, mirroring the reference
`planning_training_pipeline/tasks/{task}/visualizer.py` layout.
"""

from photon2perception.common.visualization import (
    visualize_attention_maps,
    visualize_raw_vs_rgb_comparison,
    visualize_routing_by_condition,
    visualize_token_routing,
)

__all__ = [
    'visualize_token_routing',
    'visualize_attention_maps',
    'visualize_raw_vs_rgb_comparison',
    'visualize_routing_by_condition',
]
