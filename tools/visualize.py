#!/usr/bin/env python3
"""
Visualization CLI for qualitative analysis (Section 4.6).

Thin re-export of `photon2perception.common.visualization` (token routing
heatmaps, attention maps, RAW-vs-RGB comparisons, degradation-condition
overlays) -- the actual implementations moved to
`photon2perception/common/visualization/visualizer.py` since they're shared,
task-agnostic plotting utilities used by both
`tasks/detection/visualizer.py` and `tasks/segmentation/visualizer.py`.
Kept here so `python tools/visualize.py` keeps working as a quick
self-test entry point alongside the other `tools/*.py` scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.common.visualization.visualizer import (  # noqa: F401
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


if __name__ == '__main__':
    import torch

    print("Testing visualization functions...")

    dummy_img = torch.rand(1, 224, 224)
    dummy_routing = torch.rand(196)  # 14x14 grid
    dummy_save = Path('./outputs/test_viz.png')
    dummy_save.parent.mkdir(parents=True, exist_ok=True)

    visualize_token_routing(
        dummy_img, dummy_routing, grid_h=14, grid_w=14,
        save_path=str(dummy_save),
    )

    print("Visualization tests complete.")
