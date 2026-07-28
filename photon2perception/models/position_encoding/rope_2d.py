"""
2D Rotary Position Embedding (2D RoPE / DRoPE) for Bayer Token Grids.

Applies rotary position encoding separately to the height and width dimensions
of a 2D grid of tokens. This allows the attention mechanism to be aware of
the 2D spatial structure of the Bayer sensor.

Key innovation for RAW perception:
- Standard 1D RoPE treats tokens as a 1D sequence, losing 2D spatial info
- 2D RoPE encodes (x, y) position pairs, which is natural for image grids
- The frequencies can optionally be CFA-aware to account for Bayer phase patterns

Reference:
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- Standard 2D RoPE extensions for vision transformers
"""

import torch
import torch.nn as nn
from typing import Optional


def precompute_2d_freqs_cis(
    dim: int,
    grid_h: int,
    grid_w: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    cfa_aware: bool = False,
) -> torch.Tensor:
    """
    Precompute 2D rotary frequency cis (cos + i*sin) values.

    Splits the embedding dimension into 4 equal parts:
    - First quarter: x-axis frequencies (height position)
    - Second quarter: y-axis frequencies (width position)
    - Third quarter: diagonal frequencies (x + y)
    - Fourth quarter: anti-diagonal frequencies (x - y)

    This captures both axial and diagonal spatial relationships.

    Args:
        dim: Embedding dimension (must be divisible by 8: `apply_rotary_embedding`
            splits the D-dim vector into two D/2 halves for the rotation, and
            this function further splits each half's angle table into 4
            equal bands (x/y/diag/anti-diag), so D/2 must itself be
            divisible by 4).
        grid_h: Number of tokens along height
        grid_w: Number of tokens along width
        theta: Base frequency (10000.0 standard, larger = lower freqs)
        device: Target device
        cfa_aware: If True, use different base frequencies for different CFA phases

    Returns:
        freqs_cis: (grid_h, grid_w, dim//2) angle tensor. Concatenating 4
            bands of width `dim//8` each yields exactly `dim//2`, matching
            what `apply_rotary_embedding` expects (it rotates D-dim vectors
            using a `(..., D//2)` angle table, since a 2D rotation consumes
            a *pair* of scalars per angle).
    """
    if dim % 8 != 0:
        raise ValueError(f"Embedding dimension {dim} must be divisible by 8 for 2D RoPE")

    dim_quarter = dim // 8
    half_dim = dim // 2

    # Frequency bands: theta^(-2i/d) for i in [0, dim/8)
    freq_bands = 1.0 / (
        theta ** (torch.arange(0, dim_quarter, device=device).float() / dim_quarter)
    )
    # (dim_quarter,)

    # Position grids
    h_pos = torch.arange(grid_h, device=device, dtype=torch.float32)
    w_pos = torch.arange(grid_w, device=device, dtype=torch.float32)
    # (grid_h,), (grid_w,)

    # Mesh grid for 2D positions
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
    # (grid_h, grid_w), (grid_h, grid_w)

    # Compute angles for each frequency band and position
    # x-axis: encode height position
    angles_h = h_grid.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
    # (grid_h, grid_w, dim_quarter)

    # y-axis: encode width position
    angles_w = w_grid.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
    # (grid_h, grid_w, dim_quarter)

    # Diagonal (h + w): encode diagonal spatial relationships
    angles_diag = (h_grid + w_grid).unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
    # (grid_h, grid_w, dim_quarter)

    # Anti-diagonal (h - w): encode anti-diagonal relationships
    angles_anti = (h_grid - w_grid).unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
    # (grid_h, grid_w, dim_quarter)

    # Concatenate all angles
    angles = torch.cat([angles_h, angles_w, angles_diag, angles_anti], dim=-1)
    # (grid_h, grid_w, dim//2)

    return angles


def apply_rotary_embedding(
    x: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to token features.

    Args:
        x: (B, N, D) input token features
        angles: (grid_h, grid_w, D//2) precomputed angles
    Returns:
        x_out: (B, N, D) features with rotation applied
    """
    B, N, D = x.shape
    half_D = D // 2

    # Reshape tokens to 2D spatial grid
    grid_h, grid_w = angles.shape[0], angles.shape[1]
    if grid_h * grid_w != N:
        raise ValueError(
            f"Token count {N} doesn't match grid {grid_h}x{grid_w}={grid_h * grid_w}"
        )

    x_grid = x.view(B, grid_h, grid_w, D)
    # (B, grid_h, grid_w, D)

    # Split into two halves for rotation
    x1 = x_grid[..., :half_D]  # (B, grid_h, grid_w, D//2)
    x2 = x_grid[..., half_D:]  # (B, grid_h, grid_w, D//2)

    # Compute cos and sin
    cos = torch.cos(angles).unsqueeze(0)  # (1, grid_h, grid_w, D//2)
    sin = torch.sin(angles).unsqueeze(0)

    # Apply rotation: Rot(x) = (x1*cos - x2*sin, x1*sin + x2*cos)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    x_grid = torch.cat([x1_rot, x2_rot], dim=-1)
    # (B, grid_h, grid_w, D)

    x_out = x_grid.view(B, N, D)
    return x_out


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding layer.

    Stores precomputed frequency angles and applies rotation
    to input token features based on their 2D grid position.

    Args:
        dim: Feature/embedding dimension
        grid_h: Token grid height
        grid_w: Token grid width
        theta: Base frequency
        cfa_aware: Use different frequencies for each CFA phase
    """

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        theta: float = 10000.0,
        cfa_aware: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.theta = theta
        self.cfa_aware = cfa_aware

        # Precompute and register as buffer (persistent, not a parameter)
        angles = precompute_2d_freqs_cis(dim, grid_h, grid_w, theta, cfa_aware=cfa_aware)
        self.register_buffer('angles', angles, persistent=False)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply 2D rotary embedding.

        Args:
            x: (B, N, D) token features
            positions: Optional (B, N, 2) explicit (y, x) positions for each token
        Returns:
            x: (B, N, D) features with 2D positional rotation applied
        """
        if self.angles.device != x.device:
            self.angles = self.angles.to(x.device)

        return apply_rotary_embedding(x, self.angles)


class CFAwareRoPE2D(RoPE2D):
    """
    CFA-aware 2D RoPE with different frequency bases for each Bayer phase.

    The intuition: pixels of different colors in the Bayer pattern have
    different spatial sampling rates (green is sampled twice as often as
    red/blue). We can reflect this by using different base frequencies
    for tokens at different CFA phase positions.
    """

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        theta_base: float = 10000.0,
        phase_multipliers: tuple = (1.0, 0.8, 0.8, 1.2),  # R, G1, G2, B
    ):
        super().__init__(dim, grid_h, grid_w, theta_base, cfa_aware=True)
        self.phase_multipliers = torch.tensor(phase_multipliers).view(1, 1, 4, 1)

    def forward(
        self,
        x: torch.Tensor,
        phase_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) or (B, N, 4, D) token features
            phase_indices: (N,) or (B, N) tensor of phase indices (0=R, 1=G1, 2=G2, 3=B)
        Returns:
            x: rotated features
        """
        if phase_indices is not None:
            # Scale angles by per-phase multiplier
            phase_mult = self.phase_multipliers.to(x.device)
            if phase_indices.dim() == 1:
                phase_indices = phase_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            elif phase_indices.dim() == 2:
                phase_indices = phase_indices.unsqueeze(-1).unsqueeze(-1)
            # Gather multipliers for each token
            mult = phase_mult.gather(2, phase_indices.expand(-1, -1, -1, self.angles.shape[-1]))
            angles = self.angles.unsqueeze(0) * mult.squeeze(2)
            return apply_rotary_embedding(x, angles)

        return super().forward(x)
