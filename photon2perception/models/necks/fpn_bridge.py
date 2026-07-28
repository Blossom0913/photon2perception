"""
Token-sequence -> 2D feature-pyramid bridge for RawViT.

CLAUDE.md's "Known gaps" #3 states: "Detection head expects 2D features but
RawViT backbone outputs 1D CLS token + hidden states. Need a feature
reshaping/upsampling bridge (or use dense prediction heads like
Segmenter/DPT style)."

This module implements that bridge using the "simple feature pyramid"
recipe from ViTDet (Li et al., 2022): a plain (non-hierarchical) ViT
produces tokens at a single resolution/stride, so instead of a
top-down/bottom-up FPN over naturally multi-scale backbone stages, we take
the *single*-scale token grid from one (or a few) transformer layer(s) and
generate a pyramid purely via a small set of conv/deconv "scale converters":

    stride/4  <- ConvTranspose2d x2 (upsample)
    stride/8  <- ConvTranspose2d x1 (upsample)   [often the "native" stride]
    stride/16 <- Identity                        [native patch stride]
    stride/32 <- Conv2d stride=2 (downsample)

This keeps the neck lightweight, avoids any dynamic/data-dependent control
flow (pure conv ops with static shapes), and is friendly to ONNX/TensorRT/
NPU export since it only uses Conv2d / ConvTranspose2d / GroupNorm / GELU.

Design notes for edge deployment:
- GroupNorm is used instead of BatchNorm for the scale-conversion blocks,
  since BatchNorm requires running statistics that some NPU toolchains
  handle less efficiently at small batch size 1 inference; GroupNorm has
  no batch-dependent behavior and folds cleanly into a static graph.
- All spatial reshaping uses `.view()`/`.permute()` w/ statically-known
  `grid_h`/`grid_w` (passed in, not inferred from data), so the traced/
  exported graph has fixed shapes end-to-end for a fixed input resolution.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def tokens_to_2d(x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Reshape a (B, N, D) patch-token sequence into a (B, D, grid_h, grid_w) map.

    Args:
        x: (B, N, D) token sequence with N == grid_h * grid_w (no CLS token).
        grid_h, grid_w: Static spatial grid dimensions.
    Returns:
        (B, D, grid_h, grid_w) feature map.
    """
    b, n, d = x.shape
    if n != grid_h * grid_w:
        raise ValueError(f"Token count {n} != grid_h*grid_w ({grid_h}*{grid_w}={grid_h * grid_w})")
    return x.transpose(1, 2).reshape(b, d, grid_h, grid_w)


class ScaleConverter(nn.Module):
    """Converts a single-stride feature map to a different target stride
    via a small conv/deconv stack, following ViTDet's "simple feature
    pyramid" design.

    Args:
        in_channels: Channel dimension of the input (backbone embed_dim).
        out_channels: Channel dimension of the output pyramid level.
        scale_factor: Relative spatial scale vs the input.
            - 4 or 2: upsample by that factor (uses ConvTranspose2d).
            - 1: identity scale (channel projection only).
            - 0.5 or 0.25: downsample by that factor (uses strided Conv2d).
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        layers: List[nn.Module] = []

        if scale_factor == 4:
            layers += [
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                nn.GroupNorm(num_groups=32, num_channels=in_channels // 2),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
            ]
            mid_channels = in_channels // 4
        elif scale_factor == 2:
            layers += [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)]
            mid_channels = in_channels // 2
        elif scale_factor == 1:
            mid_channels = in_channels
        elif scale_factor == 0.5:
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)]
            mid_channels = in_channels
        elif scale_factor == 0.25:
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
                nn.GroupNorm(num_groups=32, num_channels=in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            ]
            mid_channels = in_channels
        else:
            raise ValueError(f"Unsupported scale_factor {scale_factor}; use one of 4, 2, 1, 0.5, 0.25")

        # Final 1x1 projection to the common pyramid channel width, followed
        # by a 3x3 conv to smooth aliasing artifacts from the up/downsampling
        # (standard FPN "output conv" practice).
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleFeaturePyramidNeck(nn.Module):
    """ViTDet-style simple feature pyramid built from one plain-ViT feature map.

    Bridges `RawViT`'s single-resolution token output into the multi-scale
    (B, C, H_i, W_i) feature list expected by `RawDetectionHead` /
    `RawSegmentationHead`.

    Args:
        embed_dim: Backbone token dimension (RawViT's `embed_dim`).
        out_channels: Uniform output channel width for every pyramid level
            (matches `RawDetectionHead(in_channels=out_channels)`).
        grid_size: (grid_h, grid_w) of the backbone's native token grid.
        scale_factors: Relative scales (vs native grid) of each output
            level, in order. Defaults to a standard 4-level pyramid
            matching a stride-16 backbone: strides {4, 8, 16, 32} relative
            to the input image.
        source_layer_index: Which entry of `hidden_states` (0-indexed, as
            returned by `RawViT.forward`) to use as the pyramid source.
            Defaults to -1 (the last transformer block, pre final-norm).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: int = 256,
        grid_size: Tuple[int, int] = (32, 32),
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        source_layer_index: int = -1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.scale_factors = tuple(scale_factors)
        self.source_layer_index = source_layer_index

        self.converters = nn.ModuleList([
            ScaleConverter(embed_dim, out_channels, sf) for sf in self.scale_factors
        ])

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: List of (B, N+1, D) per-layer outputs from
                `RawViT.forward` (CLS token included at index 0).
            grid_size: Override the configured (grid_h, grid_w) — useful
                when sparse routing has masked (zeroed, not removed) tokens,
                in which case N is unchanged and this argument is unnecessary;
                provided for flexibility if the backbone's grid ever varies
                between calls (e.g. multi-resolution training).
        Returns:
            List of (B, out_channels, H_i, W_i) feature maps, one per
            configured scale factor, ordered from highest to lowest
            resolution (matches standard FPN level ordering P2..P5).
        """
        grid_h, grid_w = grid_size if grid_size is not None else self.grid_size
        source = hidden_states[self.source_layer_index]
        patch_tokens = source[:, 1:, :]  # drop CLS token -> (B, N, D)
        feat_2d = tokens_to_2d(patch_tokens, grid_h, grid_w)  # (B, D, grid_h, grid_w)

        return [converter(feat_2d) for converter in self.converters]

    def get_strides(self, patch_stride: int = 16) -> List[int]:
        """Return the effective input-image stride of each output level,
        given the backbone's native patch stride (e.g. 16 for patch_size=16).
        Useful for anchor generation (`losses.detection_loss.generate_anchors`).
        """
        return [int(round(patch_stride / sf)) for sf in self.scale_factors]


class MultiLayerFusionNeck(nn.Module):
    """Alternative neck that fuses several transformer layers (instead of
    just the last one) into a single feature map before pyramid conversion.

    Motivation: intermediate ViT layers often carry different levels of
    semantic abstraction (DPT / Segmenter observation); fusing a handful of
    layers via a learned weighted sum can improve dense prediction quality
    over using only the final layer, at negligible extra cost.

    Args:
        embed_dim: Backbone token dimension.
        out_channels: Output channel width for every pyramid level.
        grid_size: Native (grid_h, grid_w) of the backbone token grid.
        layer_indices: Which `hidden_states` indices to fuse.
        scale_factors: Passed through to the internal `SimpleFeaturePyramidNeck`.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: int = 256,
        grid_size: Tuple[int, int] = (32, 32),
        layer_indices: Sequence[int] = (-4, -3, -2, -1),
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
    ):
        super().__init__()
        self.layer_indices = tuple(layer_indices)
        self.layer_weights = nn.Parameter(torch.ones(len(self.layer_indices)) / len(self.layer_indices))
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.pyramid = SimpleFeaturePyramidNeck(
            embed_dim=embed_dim,
            out_channels=out_channels,
            grid_size=grid_size,
            scale_factors=scale_factors,
            source_layer_index=0,  # placeholder; forward() bypasses this by fusing first
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        weights = torch.softmax(self.layer_weights, dim=0)
        fused = sum(
            weights[i] * hidden_states[idx] for i, idx in enumerate(self.layer_indices)
        )
        fused = self.fusion_norm(fused)
        return self.pyramid(hidden_states=[fused], grid_size=grid_size)
