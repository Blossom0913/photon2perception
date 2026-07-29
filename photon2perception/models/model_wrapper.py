"""
Unified model wrapper: backbone + neck + head as a single nn.Module.

Why this exists
----------------
`RawViT` returns `(cls_token, hidden_states)`, and `RawDetectionHead` /
`RawSegmentationHead` expect 2D feature maps. Gluing these together with an
`nn.ModuleDict` (as the original `tools/train.py` did) works for eager-mode
training but is a poor fit for:

1. **`torch.jit.trace` / `torch.onnx.export`**: both want a single callable
   `forward(*inputs) -> outputs` with a stable, flat signature. A ModuleDict
   accessed as `model['backbone'](x)` at the call site works in eager mode
   but forces every consumer (train loop, eval loop, export script) to
   re-implement the same "backbone -> neck -> head" glue code, which is
   exactly the kind of duplication that causes train/eval/export skew.
2. **Edge deployment**: NPU/inference-SDK toolchains (CANN, Cambricon
   Neuware, TensorRT) expect one exported graph with named inputs/outputs.
   `PerceptionModel` below is that single graph.

`PerceptionModel` composes:
    RawViT (backbone) -> Neck (token seq -> 2D pyramid) -> Head (task-specific)

and exposes exactly one tensor-in/tensor(s)-out forward signature per task,
so the *same* module instance is used unmodified by tools/train.py,
tools/eval.py, tools/export.py, and the deploy/ inference backends.

Sparse-routing caveat (important, read before deploying)
----------------------------------------------------------
`RawViT`'s sparse router (`SaliencyRouter` / `UncertaintyRouter` /
`PhysicalPriorRouter`) only runs during eval/inference if the backbone was
constructed with `route_at_inference=True` (default False, to preserve the
original training-only-routing behavior existing checkpoints may depend
on). If you leave this at the default and export a `use_sparse_routing=True`
model, calling `model.eval()` before export will silently skip the router
entirely, so the exported graph runs *dense* despite the model being
"configured" for sparse routing -- the efficiency win central to this
project's thesis would not actually materialize on the exported artifact.
`PerceptionModel.routing_active` reports whether routing will actually run
given the current `.training` state, and `tools/export.py` asserts on this
(and errors loudly) before exporting a routing-enabled model, so this
misconfiguration can't silently ship. To enable inference-time routing,
pass `route_at_inference=True` in the model config (see
`build_perception_model`).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbones.raw_vit import RawViT
from ..common.head.detection_head import RawDetectionHead
from ..common.head.segmentation_head import RawSegmentationHead
from .necks.fpn_bridge import SimpleFeaturePyramidNeck


class PerceptionModel(nn.Module):
    """End-to-end RAW perception model: RawViT backbone + neck + task head.

    Args:
        backbone: A `RawViT` instance.
        neck: A module mapping `hidden_states -> List[(B,C,H,W)]`
            (e.g. `SimpleFeaturePyramidNeck`). May be None for tasks that
            only need the CLS token (not currently used, reserved for
            future classification-only heads).
        head: Task-specific head (`RawDetectionHead` or `RawSegmentationHead`).
        task: 'detection' or 'segmentation' — controls how `neck`'s output
            is fed into `head` and what `forward` returns.
        seg_output_size: Target (H, W) for segmentation logits upsampling
            (only used when task == 'segmentation'; ignored otherwise).
    """

    def __init__(
        self,
        backbone: RawViT,
        neck: Optional[nn.Module],
        head: nn.Module,
        task: str = 'detection',
    ):
        super().__init__()
        if task not in ('detection', 'segmentation'):
            raise ValueError(f"task must be 'detection' or 'segmentation', got '{task}'")
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.task = task

    @property
    def routing_active(self) -> bool:
        """Whether this model's backbone has sparse routing configured AND
        that routing actually executes during a call to `forward` given the
        current `self.training` state (and `backbone.route_at_inference`).
        See module docstring for context.
        """
        router = self.backbone.router
        if router is None:
            return False
        return self.training or self.backbone.route_at_inference

    def forward(
        self,
        images: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            images: (B, 1, H, W) Bayer RAW input (post-normalization).
            raw_image: Optional (B, 1, H, W) *unnormalized* RAW image, used
                only by `PhysicalPriorRouter` for its physical-saliency
                prior. If None, `images` is reused (acceptable for most
                cases since routing only needs relative local variance).
        Returns:
            detection: (cls_scores, bbox_preds) — each a list of per-FPN-level
                tensors, matching `RawDetectionHead.forward`'s return format.
            segmentation: seg_logits, a single (B, num_classes, H_out, W_out) tensor.
        """
        if raw_image is None:
            raw_image = images

        cls_token, hidden_states = self.backbone(images, raw_image=raw_image)

        if self.task == 'detection':
            features = self.neck(hidden_states, grid_size=self.backbone.grid_size)
            cls_scores, bbox_preds = self.head(features)
            return cls_scores, bbox_preds

        # segmentation
        grid_h, grid_w = self.backbone.grid_size
        patch_tokens = hidden_states[-1][:, 1:, :]  # (B, N, D), drop CLS
        seg_logits = self.head(patch_tokens, grid_h, grid_w)
        return seg_logits

    def get_strides(self) -> List[int]:
        """Effective input-image strides of the neck's pyramid levels
        (detection only). Used for anchor generation.
        """
        if self.task != 'detection' or self.neck is None:
            raise ValueError("get_strides() is only valid for detection models with a neck")
        return self.neck.get_strides(patch_stride=self.backbone.patch_size)


def build_perception_model(config: Dict) -> PerceptionModel:
    """Construct a `PerceptionModel` from a full experiment config dict, as
    loaded from tasks/detection/config/*.yaml or tasks/segmentation/config/*.yaml.

    This is the single source of truth for "config dict -> nn.Module" used
    by tools/train.py, tools/eval.py, and tools/export.py, replacing the
    previous ad hoc `build_model` in tools/train.py. It intentionally takes
    the *entire* config (not just `config['model']`) because, per the
    existing YAML schema, `num_classes` lives under `config['data']` rather
    than `config['model']` — keeping this function config-schema-compatible
    with the existing YAML files avoids a churny, purely-cosmetic YAML
    migration.

    Args:
        config: Full experiment config (must contain 'task', 'model', 'data' keys).
    Returns:
        A `PerceptionModel` ready for `.to(device)`.
    """
    task = config['task']
    model_cfg = config['model']
    num_classes = config['data']['num_classes']

    img_size = tuple(model_cfg['img_size'])
    patch_size = model_cfg['patch_size']
    embed_dim = model_cfg['embed_dim']

    backbone = RawViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=model_cfg.get('in_chans', 1),
        embed_dim=embed_dim,
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
        dropout=model_cfg.get('dropout', 0.0),
        attn_dropout=model_cfg.get('attn_dropout', 0.0),
        cfa_pattern=model_cfg.get('cfa_pattern', 'rggb'),
        use_rope_2d=model_cfg.get('use_rope_2d', True),
        use_directional=model_cfg.get('use_directional', False),
        use_sparse_routing=model_cfg.get('use_sparse_routing', False),
        router_type=model_cfg.get('router_type', 'saliency'),
        keep_ratio=model_cfg.get('keep_ratio', 0.7),
        attn_backend=model_cfg.get('attn_backend', 'sdpa'),
        route_at_inference=model_cfg.get('route_at_inference', False),
    )
    grid_h, grid_w = backbone.grid_size
    neck_out_channels = model_cfg.get('neck_out_channels', 256)

    if task == 'detection':
        neck = SimpleFeaturePyramidNeck(
            embed_dim=embed_dim,
            out_channels=neck_out_channels,
            grid_size=(grid_h, grid_w),
            scale_factors=tuple(model_cfg.get('neck_scale_factors', (4.0, 2.0, 1.0, 0.5))),
        )
        head = RawDetectionHead(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            num_anchors=model_cfg.get('num_anchors', 9),
            feat_channels=model_cfg.get('feat_channels', neck_out_channels),
        )
    else:  # segmentation
        neck = None
        head = RawSegmentationHead(
            in_channels=embed_dim,
            num_classes=num_classes,
            hidden_dim=model_cfg.get('seg_hidden_dim', 256),
            img_size=tuple(model_cfg.get('seg_output_size', img_size)),
        )

    return PerceptionModel(backbone=backbone, neck=neck, head=head, task=task)
