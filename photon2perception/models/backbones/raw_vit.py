"""
RAW-adapted Vision Transformer Backbone.

Wraps a standard ViT with RAW-specific components:
1. Bayer CFA-aware tokenization
2. 2D RoPE position encoding
3. Optional directional enhancement
4. Optional sparse routing for efficient inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from ..tokenization.bayer_patch_embed import BayerPatchEmbed, BayerFineTokenize
from ..position_encoding.rope_2d import RoPE2D, CFAwareRoPE2D
from ..position_encoding.directional import DirectionalEnhance, BayerDirectionalEnhance
from ..routing.router import SaliencyRouter, UncertaintyRouter, PhysicalPriorRouter


class RawViTBlock(nn.Module):
    """
    Transformer block for RAW token processing.

    Standard ViT block with the addition of:
    - 2D RoPE injection in self-attention
    - Optional sparse routing before the block
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Self-attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def _apply_rope_to_qk(self, q: torch.Tensor, k: torch.Tensor, rope: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        B, N, D = q.shape
        q_grid = q.view(B, N, self.num_heads, self.head_dim)
        k_grid = k.view(B, N, self.num_heads, self.head_dim)
        # Apply RoPE per head
        q_grid = rope(q_grid.view(B * self.num_heads, N, self.head_dim))
        k_grid = rope(k_grid.view(B * self.num_heads, N, self.head_dim))
        q = q_grid.view(B, N, D)
        k = k_grid.view(B, N, D)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Self-attention with optional RoPE
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)

        if self.use_rope and rope is not None:
            # Apply 2D RoPE - need to pass through rope module
            # For simplicity, we apply rotation in the token dimension
            q = q.permute(0, 2, 1, 3).reshape(B, N, D)
            k = k.permute(0, 2, 1, 3).reshape(B, N, D)
            q = rope(q)
            k = rope(k)
            q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_dropout(x_attn)
        x = x + x_attn

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class RawViT(nn.Module):
    """
    RAW-adapted Vision Transformer.

    Process:
    1. BayerPatchEmbed: CFA-aware patch tokenization
    2. CFA phase embeddings
    3. 2D RoPE applied in each transformer block's attention
    4. Optional directional enhancement after patch embedding
    5. Optional sparse routing before transformer blocks

    Args:
        img_size: Input image size (must be even)
        patch_size: Patch size (must be even)
        in_chans: Input channels (1 for Bayer RAW)
        embed_dim: Token embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        cfa_pattern: Bayer CFA pattern
        use_rope_2d: Enable 2D RoPE
        use_directional: Enable directional enhancement
        use_sparse_routing: Enable sparse routing
        router_type: 'saliency', 'uncertainty', or 'physical'
        keep_ratio: Token keep ratio for sparse routing
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cfa_pattern: str = 'rggb',
        use_rope_2d: bool = True,
        use_directional: bool = False,
        use_sparse_routing: bool = False,
        router_type: str = 'saliency',
        keep_ratio: float = 0.7,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_rope_2d = use_rope_2d
        self.use_directional = use_directional
        self.use_sparse_routing = use_sparse_routing

        # CFA-aware patch embedding
        self.patch_embed = BayerPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            cfa_pattern=cfa_pattern,
            use_cfa_embed=True,
        )
        grid_h, grid_w = self.patch_embed.grid_size
        self.grid_size = (grid_h, grid_w)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embedding (optional, on top of RoPE)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_h * grid_w + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Dropout
        self.pos_drop = nn.Dropout(dropout)

        # 2D RoPE
        if use_rope_2d:
            self.rope = RoPE2D(
                dim=embed_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            )
        else:
            self.rope = None

        # Directional enhancement
        if use_directional:
            self.directional = DirectionalEnhance(dim=embed_dim)
        else:
            self.directional = None

        # Sparse routing
        if use_sparse_routing:
            if router_type == 'saliency':
                self.router = SaliencyRouter(dim=embed_dim, keep_ratio=keep_ratio)
            elif router_type == 'uncertainty':
                self.router = UncertaintyRouter(dim=embed_dim, keep_ratio=keep_ratio)
            elif router_type == 'physical':
                self.router = PhysicalPriorRouter(
                    dim=embed_dim, patch_size=patch_size, keep_ratio=keep_ratio
                )
            else:
                raise ValueError(f"Unknown router type: {router_type}")
        else:
            self.router = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RawViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_rope=use_rope_2d,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self,
        x: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, H, W) Bayer RAW image
            raw_image: Same as x, used for physical prior routing
        Returns:
            cls_token: (B, D) final classification token features
            hidden_states: List of (B, N, D) from each block
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add learnable position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Optional directional enhancement
        if self.directional is not None:
            grid_h, grid_w = self.grid_size
            # Apply to patch tokens (exclude CLS)
            x_patches = x[:, 1:, :]
            x_patches = self.directional(x_patches, grid_h, grid_w)
            x = torch.cat([x[:, :1, :], x_patches], dim=1)

        # Optional sparse routing
        route_scores = None
        if self.router is not None and self.training:
            x_patches = x[:, 1:, :]
            if isinstance(self.router, PhysicalPriorRouter) and raw_image is not None:
                x_routed, route_scores = self.router(
                    x_patches, raw_image=raw_image,
                    grid_h=self.grid_size[0], grid_w=self.grid_size[1],
                    training=self.training
                )
            else:
                x_routed, route_scores = self.router(x_patches, training=self.training)
            x = torch.cat([x[:, :1, :], x_routed], dim=1)

        # Transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, rope=self.rope)
            hidden_states.append(x)

        # Final norm
        x = self.norm(x)

        # Return CLS token and hidden states
        return x[:, 0, :], hidden_states

    def get_num_tokens(self) -> int:
        """Return number of patch tokens."""
        return self.grid_size[0] * self.grid_size[1]
