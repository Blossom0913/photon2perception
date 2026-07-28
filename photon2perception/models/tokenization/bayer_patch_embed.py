"""
Bayer RAW CFA-aware Patch Embedding.

Converts a single-channel Bayer RAW image (H x W) into a sequence of tokens
while preserving the 2x2 RGGB Color Filter Array spatial structure.

Key design decisions:
- Patch size must be even to capture complete Bayer quads (2x2 blocks)
- Each patch covers (patch_size // 2) x (patch_size // 2) complete quads
- CFA phase information is preserved through phase embeddings
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BayerPatchEmbed(nn.Module):
    """
    CFA-aware patch embedding for Bayer RAW images.

    Unlike standard ViT patch embedding which treats all pixels uniformly,
    this module respects the 2x2 RGGB Bayer pattern. Each patch captures
    an integer number of complete Bayer quads.

    Args:
        img_size: Input image size (height, width). Both must be even.
        patch_size: Patch size. Must be even (e.g., 4, 8, 16).
        in_chans: Input channels. 1 for Bayer RAW.
        embed_dim: Output token dimension.
        norm_layer: Optional normalization layer after projection.
        cfa_pattern: Bayer CFA pattern. One of 'rggb', 'bggr', 'grbg', 'gbrg'.
        use_cfa_embed: Whether to add CFA phase embeddings.
    """

    CFA_PATTERNS = {
        'rggb': [[0, 1], [1, 2]],  # R=0, G1=1, G2=1, B=2 (G shared)
        'bggr': [[2, 1], [1, 0]],
        'grbg': [[1, 0], [2, 1]],
        'gbrg': [[1, 2], [0, 1]],
    }

    # Maps each CFA phase to a unique index (R, G1, G2, B)
    # R=0, G_at_R_row=1, G_at_B_row=2, B=3
    CFA_PHASE_MAP = {
        'rggb': {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},  # R, G1, G2, B
        'bggr': {(0, 0): 3, (0, 1): 2, (1, 0): 1, (1, 1): 0},
        'grbg': {(0, 0): 1, (0, 1): 0, (1, 0): 3, (1, 1): 2},
        'gbrg': {(0, 0): 2, (0, 1): 3, (1, 0): 0, (1, 1): 1},
    }

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        cfa_pattern: str = 'rggb',
        use_cfa_embed: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.cfa_pattern = cfa_pattern.lower()
        self.use_cfa_embed = use_cfa_embed

        if patch_size % 2 != 0:
            raise ValueError(
                f"patch_size must be even to capture complete Bayer quads, got {patch_size}"
            )
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError(
                f"Image dimensions {img_size} must be divisible by patch_size {patch_size}"
            )

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Projection: patch_size^2 pixels per patch -> embed_dim
        # We use a conv2d with stride=patch_size for efficient patching
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # CFA phase embeddings: 4 phases (R, G1, G2, B)
        if use_cfa_embed:
            self.cfa_phase_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, 4, embed_dim)
            )
            nn.init.trunc_normal_(self.cfa_phase_embed, std=0.02)
        else:
            self.cfa_phase_embed = None

        # Precompute CFA phase indices for each patch position
        self.register_buffer(
            'phase_indices',
            self._compute_phase_indices(),
            persistent=True
        )

    def _compute_phase_indices(self) -> torch.Tensor:
        """
        Compute which CFA phase each patch starts at.

        For a patch at grid position (i, j), the top-left pixel is at
        (i * patch_size, j * patch_size) in the original image.
        Its Bayer phase is determined by (i * patch_size % 2, j * patch_size % 2).

        Since patch_size is even, this equals (0, 0) for all patches,
        meaning all patches start at the same CFA phase (top-left of a Bayer quad).
        We add an offset of (0, 0) = phase 0 for the top-left corner of each patch.

        Returns:
            phase_indices: (grid_h, grid_w) tensor with values 0-3
        """
        phase_map = self.CFA_PHASE_MAP[self.cfa_pattern]
        h, w = self.grid_size
        indices = torch.zeros(h, w, dtype=torch.long)

        for i in range(h):
            for j in range(w):
                # Top-left pixel of this patch in image coordinates
                y = i * self.patch_size
                x = j * self.patch_size
                # Bayer phase at (y, x)
                phase_y = y % 2
                phase_x = x % 2
                indices[i, j] = phase_map[(phase_y, phase_x)]

        return indices

    def add_cfa_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add CFA phase embeddings to patch tokens.

        Since patch_size is even, all patches start at the same CFA phase (phase 0).
        However, within each patch, individual pixels have different phases.
        We add a learnable embedding per phase to inform the model.

        For simplicity, we add the phase-0 embedding to every patch (since all
        patches share the same starting phase when patch_size is even).
        When finer phase information is needed, use BayerFineTokenize instead.

        Args:
            x: (B, N, D) patch tokens
        Returns:
            x: (B, N, D) tokens with CFA phase embeddings added
        """
        if self.cfa_phase_embed is not None:
            # All patches start at phase 0 when patch_size is even
            # phase_idx 0 = top-left of Bayer quad
            B = x.shape[0]
            # For now, add the average phase embedding or phase-0
            # In the fine-grained version, we can apply per-phase embeddings
            # to individual tokens within each patch
            phase_embed = self.cfa_phase_embed[:, :, 0, :]  # phase 0
            x = x + phase_embed
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) Bayer RAW image tensor
        Returns:
            tokens: (B, N, D) patch token sequence
        """
        B, C, H, W = x.shape
        if C != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {C}")
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Expected image size {self.img_size}, got ({H}, {W})"
            )

        # Patchify and project: (B, 1, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims: (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # Add CFA phase embeddings
        x = self.add_cfa_embedding(x)

        # Normalize
        x = self.norm(x)

        return x


class BayerFineTokenize(nn.Module):
    """
    Fine-grained Bayer tokenizer that creates per-phase tokens.

    Instead of one token per patch, this creates 4 tokens per patch —
    one for each Bayer phase (R, G1, G2, B). This is useful when
    the model needs explicit per-phase processing.

    Each 2x2 Bayer quad produces 4 tokens, one per color channel value.
    These can optionally be merged back into a single token after
    phase-specific processing.

    Args:
        embed_dim: Dimension for each phase token.
        merge_phases: If True, merge 4 phase tokens back to 1 per quad.
    """

    def __init__(
        self,
        embed_dim: int = 192,  # 768/4 for equivalent total dimension
        merge_phases: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.merge_phases = merge_phases

        # Separate projection for each Bayer phase
        # Each phase pixel is projected to embed_dim
        self.phase_proj = nn.ModuleDict({
            'R': nn.Linear(1, embed_dim),
            'G1': nn.Linear(1, embed_dim),
            'G2': nn.Linear(1, embed_dim),
            'B': nn.Linear(1, embed_dim),
        })

        # Phase type embeddings
        self.phase_type_embed = nn.Parameter(
            torch.zeros(1, 1, 4, embed_dim)
        )
        nn.init.trunc_normal_(self.phase_type_embed, std=0.02)

        if merge_phases:
            self.merge = nn.Linear(4 * embed_dim, 4 * embed_dim)

    def extract_bayer_phases(self, x: torch.Tensor) -> dict:
        """
        Extract R, G1, G2, B sub-arrays from Bayer RAW image.

        For RGGB pattern:
            R = x[:, :, 0::2, 0::2]
            G1 = x[:, :, 0::2, 1::2]
            G2 = x[:, :, 1::2, 0::2]
            B = x[:, :, 1::2, 1::2]

        Args:
            x: (B, 1, H, W) Bayer RAW
        Returns:
            Dict with keys 'R', 'G1', 'G2', 'B', each (B, H', W')
        """
        return {
            'R': x[:, :, 0::2, 0::2],
            'G1': x[:, :, 0::2, 1::2],
            'G2': x[:, :, 1::2, 0::2],
            'B': x[:, :, 1::2, 1::2],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) Bayer RAW image
        Returns:
            If merge_phases: (B, H/2 * W/2, 4*embed_dim) tokens
            Else: (B, H/2 * W/2, 4, embed_dim) phase-separated tokens
        """
        B, C, H, W = x.shape

        # Extract four Bayer phases
        phases = self.extract_bayer_phases(x)
        # Each: (B, 1, H/2, W/2)

        # Project each phase
        tokens = {}
        for phase_name, phase_data in phases.items():
            B, _, h, w = phase_data.shape
            flat = phase_data.view(B, h * w, 1)  # (B, N_pixels, 1)
            tokens[phase_name] = self.phase_proj[phase_name](flat)  # (B, N, D)

        # Stack phases: (B, N, 4, D)
        stacked = torch.stack(
            [tokens['R'], tokens['G1'], tokens['G2'], tokens['B']],
            dim=2
        )

        # Add phase type embeddings
        stacked = stacked + self.phase_type_embed

        if self.merge_phases:
            B, N, _, D = stacked.shape
            merged = stacked.view(B, N, 4 * D)
            merged = self.merge(merged)
            return merged

        return stacked
