"""
RAW-adapted Vision Transformer Backbone.

Wraps a standard ViT with RAW-specific components:
1. Bayer CFA-aware tokenization
2. 2D RoPE position encoding
3. Optional directional enhancement
4. Optional sparse routing for efficient inference
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..tokenization.bayer_patch_embed import BayerPatchEmbed, BayerFineTokenize
from ..position_encoding.rope_2d import RoPE2D, CFAwareRoPE2D
from ..position_encoding.directional import DirectionalEnhance, BayerDirectionalEnhance
from ..routing.router import SaliencyRouter, UncertaintyRouter, PhysicalPriorRouter

# `F.scaled_dot_product_attention` (SDPA) was added in torch 2.0. It
# transparently dispatches to FlashAttention-2 / memory-efficient attention
# / math kernels depending on hardware and input dtype, without requiring
# the `flash-attn` pip package. We treat SDPA as the primary "flash
# attention" backend for this project since it needs no extra native-CUDA
# extension build step (important for portability to edge/dev machines
# where compiling flash-attn from source is often infeasible), and it also
# has a CPU fallback (the math kernel) so eager-mode unit tests still pass
# on CPU-only machines (see CLAUDE.md's CPU-only test instructions).
_HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')

# Attention backend choices for `RawViTBlock(attn_backend=...)`:
#   'sdpa'  : torch.nn.functional.scaled_dot_product_attention (default;
#             auto-selects Flash/mem-efficient/math kernel; also exportable
#             via torch.onnx as of opset 14+/torch>=2.1 with static shapes).
#   'math'  : explicit eager softmax(QK^T)V, kept for debugging, for
#             environments without SDPA, and for exporters/NPU toolchains
#             (e.g. some ONNX->NPU converters) that don't yet recognize the
#             fused SDPA op and need the unrolled matmul/softmax graph.
#   'flash' : alias for 'sdpa' restricted to the flash-attention kernel via
#             `torch.nn.attention.sdpa_kernel` context manager (CUDA only;
#             silently falls back to 'sdpa' default dispatch elsewhere).
ATTENTION_BACKENDS = ('sdpa', 'math', 'flash')


def apply_shared_rope_multihead(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a `RoPE2D`-family module to (B, num_heads, N, head_dim) q/k tensors.

    `RoPE2D` is constructed with `dim=embed_dim` (see `RawViT.__init__`), so
    its precomputed `angles` table has last-dim size `embed_dim // 2` -- it
    expects to rotate the *full* concatenated multi-head vector, not each
    head's `head_dim` slice independently (`head_dim` is generally much
    smaller than `embed_dim`, so those two are not interchangeable; naively
    reshaping the head dim into the batch dim, as an earlier version of this
    function did, causes a shape mismatch inside `apply_rotary_embedding`
    since `head_dim // 2 != embed_dim // 2`).

    We therefore merge the head dimension back into the feature dimension
    (undoing the `(B, N, D) -> (B, num_heads, N, head_dim)` split done by
    the caller for attention), apply RoPE once over the full `D = num_heads
    * head_dim` vector -- matching `RawViT`'s single-shared-RoPE design --
    and re-split back into per-head tensors afterward.

    Args:
        q, k: (B, num_heads, N, head_dim)
        rope: A module with `forward(x: (B, N, D)) -> (B, N, D)` semantics,
            where `D == num_heads * head_dim`.
    Returns:
        q, k rotated, same shape as input.
    """
    B, H, N, Dh = q.shape
    D = H * Dh
    q_flat = q.transpose(1, 2).reshape(B, N, D)
    k_flat = k.transpose(1, 2).reshape(B, N, D)
    q_flat = rope(q_flat)
    k_flat = rope(k_flat)
    q_out = q_flat.view(B, N, H, Dh).transpose(1, 2)
    k_out = k_flat.view(B, N, H, Dh).transpose(1, 2)
    return q_out, k_out


class RawViTBlock(nn.Module):
    """
    Transformer block for RAW token processing.

    Standard ViT block with the addition of:
    - 2D RoPE injection in self-attention
    - Optional sparse routing before the block
    - Pluggable attention backend (SDPA/FlashAttention or explicit math),
      see `ATTENTION_BACKENDS`.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim expansion ratio.
        dropout: Dropout on projection output and MLP.
        attn_dropout: Dropout on attention weights (only applied by the
            'math' backend during training; SDPA applies it internally via
            its `dropout_p` argument).
        use_rope: Whether to apply 2D RoPE to q/k before attention.
        attn_backend: One of `ATTENTION_BACKENDS`. Defaults to 'sdpa'
            falling back to 'math' automatically if the installed torch
            version doesn't provide `F.scaled_dot_product_attention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_rope: bool = True,
        attn_backend: str = 'sdpa',
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.attn_dropout_p = attn_dropout

        if attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attn_backend must be one of {ATTENTION_BACKENDS}, got '{attn_backend}'")
        if attn_backend in ('sdpa', 'flash') and not _HAS_SDPA:
            warnings.warn(
                f"attn_backend='{attn_backend}' requested but this torch version has no "
                "F.scaled_dot_product_attention (requires torch>=2.0); falling back to 'math'.",
                RuntimeWarning,
            )
            attn_backend = 'math'
        self.attn_backend = attn_backend

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

    def set_attn_backend(self, attn_backend: str) -> None:
        """Switch attention backend after construction (e.g. force 'math'
        before ONNX export if the target NPU toolchain lacks an SDPA op).
        """
        if attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"attn_backend must be one of {ATTENTION_BACKENDS}, got '{attn_backend}'")
        if attn_backend in ('sdpa', 'flash') and not _HAS_SDPA:
            attn_backend = 'math'
        self.attn_backend = attn_backend

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Core attention computation. q/k/v: (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)."""
        if self.attn_backend == 'math':
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            return attn @ v

        dropout_p = self.attn_dropout_p if self.training else 0.0

        if self.attn_backend == 'flash' and q.is_cuda:
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            except Exception:
                # Flash kernel unavailable for this input config (e.g. dtype/
                # head_dim unsupported) -- fall back to default SDPA dispatch,
                # which will pick memory-efficient or math kernels instead.
                pass

        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

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
            # `rope.angles` only covers the `grid_h * grid_w` *patch* tokens,
            # not the extra CLS token prepended in `RawViT.forward` (N here
            # is grid_h*grid_w + 1). The CLS token has no 2D spatial
            # position, so it must be excluded from rotation -- rotating it
            # against a `grid_h*grid_w`-sized angle table would either
            # crash (shape mismatch) or silently rotate the wrong token.
            # We therefore split off q[:, :, :1] (CLS) and only rotate
            # q[:, :, 1:] (patches), then re-concatenate.
            q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]
            q_patches, k_patches = apply_shared_rope_multihead(q_patches, k_patches, rope)
            q = torch.cat([q_cls, q_patches], dim=2)
            k = torch.cat([k_cls, k_patches], dim=2)

        x_attn = self._attend(q, k, v)  # (B, num_heads, N, head_dim)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
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
        attn_backend: Attention kernel used by every `RawViTBlock`. One of
            `ATTENTION_BACKENDS` ('sdpa' default, 'flash', or 'math'). Can
            be changed after construction via `set_attn_backend()`.
        route_at_inference: If True, sparse routing also runs when the
            module is in `eval()` mode (using each router's hard top-K
            code path, `training=False`), so the efficiency benefit of
            routing is actually realized at inference time / after export.
            Defaults to False to preserve the original behavior (routing
            only active during `.train()`), which existing checkpoints/
            experiments may implicitly depend on. See the "Sparse-routing
            caveat" note in `photon2perception/models/model_wrapper.py`
            for why this matters before deploying a routing-enabled model.
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
        attn_backend: str = 'sdpa',
        route_at_inference: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_rope_2d = use_rope_2d
        self.use_directional = use_directional
        self.use_sparse_routing = use_sparse_routing
        self.route_at_inference = route_at_inference

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
        self.attn_backend = attn_backend
        self.blocks = nn.ModuleList([
            RawViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_rope=use_rope_2d,
                attn_backend=attn_backend,
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

    def set_attn_backend(self, attn_backend: str) -> None:
        """Switch the attention kernel of every block after construction.

        Typical use: force 'math' before tracing/exporting to a target
        (ONNX/NPU) that doesn't understand the fused SDPA op, while keeping
        'sdpa'/'flash' for actual GPU training/inference.
        """
        self.attn_backend = attn_backend
        for block in self.blocks:
            block.set_attn_backend(attn_backend)

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

        # Optional sparse routing.
        # Active during training unconditionally; active during eval only if
        # `route_at_inference=True` was set at construction time (see the
        # class docstring's "route_at_inference" arg for why this defaults
        # to False). Both branches pass `training=self.training` through to
        # the router itself, which controls Gumbel-Softmax (soft, training)
        # vs hard top-K (eval) token selection -- independent of whether
        # routing runs at all.
        route_scores = None
        routing_enabled = self.training or self.route_at_inference
        if self.router is not None and routing_enabled:
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
