"""
Directional Enhancement for Bayer Token Grids.

Optional module that models local directional patterns in the Bayer grid.
Examples: diagonal correlations between same-color pixels, vertical/horizontal
gradients within same-phase neighbors.

This is a lightweight, residual module applied after 2D RoPE.
It is explicitly designed to be optional and not load-bearing — the
framework should work well without it, and it provides a small additional
boost when enabled.

Design:
- Depthwise separable conv on the 2D token grid
- Predicts small residual offsets to token features
- Gated: learnable gate parameter controls contribution strength
"""

import torch
import torch.nn as nn
from typing import Optional


class DirectionalEnhance(nn.Module):
    """
    Directional enhancement for Bayer token features.

    Applies lightweight spatial convolutions on the 2D token grid
    to capture local Bayer-specific patterns (cross-color gradients,
    same-color diagonal correlations, etc.).

    Args:
        dim: Token feature dimension
        kernel_size: Spatial kernel size (default 3)
        num_directions: Number of directional filters (4: H, V, D1, D2)
        gate_init: Initial value for the learnable gate (0 = disabled initially)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        num_directions: int = 4,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_directions = num_directions

        # Directional convolution kernels implemented as grouped depthwise convs
        # Each "direction" is a separate conv group
        self.directional_conv = nn.Conv2d(
            dim, dim * num_directions,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,  # depthwise
            bias=False,
        )

        # Direction mixing: combine directional outputs
        self.direction_mix = nn.Conv2d(
            dim * num_directions, dim,
            kernel_size=1,
            bias=False,
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)

        # Learnable gate: starts at gate_init (0 = module disabled at start)
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # Activation
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token features
            grid_h: Token grid height
            grid_w: Token grid width
        Returns:
            x: (B, N, D) enhanced token features
        """
        B, N, D = x.shape
        if N != grid_h * grid_w:
            raise ValueError(
                f"Token count {N} doesn't match grid {grid_h}x{grid_w}"
            )

        # Reshape to 2D spatial grid
        x_2d = x.view(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
        # (B, D, grid_h, grid_w)

        # Apply directional convolutions
        residual = self.directional_conv(x_2d)
        # (B, D * num_directions, grid_h, grid_w)

        # Mix direction outputs
        residual = self.direction_mix(residual)
        # (B, D, grid_h, grid_w)

        residual = self.act(residual)

        # Reshape back to token sequence
        residual = residual.permute(0, 2, 3, 1).reshape(B, N, D)
        # (B, N, D)

        # Normalize residual
        residual = self.norm(residual)

        # Gated addition: output = x + tanh(gate) * residual
        # tanh(gate) bounds the contribution to [-1, 1]
        gate_val = torch.tanh(self.gate)
        x = x + gate_val * residual

        return x


class BayerDirectionalEnhance(DirectionalEnhance):
    """
    Bayer-specific directional enhancement.

    Extends the base DirectionalEnhance with awareness of the CFA pattern.
    Uses separate directional kernels for each Bayer phase's spatial neighborhood.

    The four Bayer phases have different spatial relationships:
    - R pixels are surrounded by G1 (horizontal) and G2 (vertical) neighbors
    - B pixels similarly have G neighbors
    - G1 and G2 have both R/B and each other as neighbors
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        gate_init: float = 0.0,
    ):
        super().__init__(
            dim=dim,
            kernel_size=kernel_size,
            num_directions=4,
            gate_init=gate_init,
        )

        # Per-phase directional offsets (learnable)
        # These model the asymmetric neighborhood of each Bayer phase
        self.phase_offsets = nn.Parameter(
            torch.zeros(4, dim)  # 4 phases: R, G1, G2, B
        )
        nn.init.trunc_normal_(self.phase_offsets, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        phase_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token features
            grid_h: Grid height
            grid_w: Grid width
            phase_mask: (B, N) or (N,) tensor with phase indices (0-3)
                        indicating which Bayer phase each token belongs to.
        Returns:
            x: enhanced features
        """
        # Apply base directional enhancement
        x = super().forward(x, grid_h, grid_w)

        # Add per-phase bias
        if phase_mask is not None:
            if phase_mask.dim() == 1:
                phase_mask = phase_mask.unsqueeze(0)
            B = x.shape[0]
            # Gather per-phase offsets
            offsets = self.phase_offsets[phase_mask]  # (B, N, D)
            # Gated addition
            gate_val = torch.tanh(self.gate)
            x = x + gate_val * offsets

        return x
