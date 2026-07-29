"""Shared, task-agnostic visualization utilities (token routing heatmaps,
attention maps, RAW-vs-RGB comparisons, degradation-condition overlays).

See `tasks/{detection,segmentation}/visualizer.py` for the thin per-task
entry points that re-export the subset relevant to each task.
"""

from .visualizer import (
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
