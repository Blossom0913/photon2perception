"""
Sparse Routing for Efficient RAW Perception.

Implements saliency-aware and uncertainty-aware token pruning mechanisms.
The core idea: not all image regions are equally informative for perception.
By selectively routing computation to salient/uncertain regions, we reduce
latency, memory bandwidth, and compute while maintaining task performance.

Two routing strategies:
1. Saliency Router: Predicts keep/drop scores from token features
2. Uncertainty Router: Uses MC Dropout to estimate per-token variance

Physical prior: tokens in flat/uniform regions (low local variance in raw values)
are preferred for pruning, since they carry less perceptual information.

Reference inspiration:
- DynamicViT (Rao et al., 2021)
- A-ViT (Yin et al., 2022)
- Token Merging (Bolya et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SaliencyRouter(nn.Module):
    """
    Saliency-aware token router.

    Predicts a keep/drop score for each token using a lightweight MLP.
    Tokens with low saliency scores are pruned during inference.

    During training, uses Gumbel-Softmax for differentiable pruning.
    During inference, uses hard top-K selection.

    Args:
        dim: Token feature dimension
        hidden_dim: MLP hidden dimension
        temperature: Gumbel-Softmax temperature (lower = harder decisions)
        keep_ratio: Target fraction of tokens to keep (used at inference)
        min_keep_ratio: Minimum fraction to keep (safety bound)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        keep_ratio: float = 0.7,
        min_keep_ratio: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.keep_ratio = keep_ratio
        self.min_keep_ratio = min_keep_ratio
        self.temperature = temperature

        # Lightweight scoring MLP
        self.score_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Optional: incorporate CFA phase info
        self.phase_scale = nn.Parameter(torch.ones(4))  # one scale per Bayer phase

    def forward(
        self,
        x: torch.Tensor,
        phase_indices: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) token features
            phase_indices: (B, N) or (N,) CFA phase indices (0-3)
            training: If True, use Gumbel-Softmax; else hard top-K
        Returns:
            x_kept: (B, N, D) routed features (dropped tokens set to near-zero)
            scores: (B, N, 1) keep/drop scores
        """
        B, N, D = x.shape

        # Predict raw saliency scores
        scores = self.score_net(x)  # (B, N, 1)

        # Modulate by CFA phase if available
        if phase_indices is not None:
            if phase_indices.dim() == 1:
                phase_indices = phase_indices.unsqueeze(0).unsqueeze(-1)
            elif phase_indices.dim() == 2:
                phase_indices = phase_indices.unsqueeze(-1)
            phase_scale = self.phase_scale.view(1, 1, 4).to(x.device)
            phase_weight = phase_scale.gather(2, phase_indices.expand(-1, -1, 1))
            scores = scores * phase_weight.squeeze(-1).unsqueeze(-1)

        scores = torch.sigmoid(scores)

        if training:
            # Differentiable pruning with Gumbel-Softmax
            # Convert score to logit for Gumbel
            eps = 1e-7
            logits = torch.log(torch.clamp(scores, eps, 1 - eps))
            # Add Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + eps) + eps)
            logits = (logits + gumbel_noise) / self.temperature
            # Soft mask via sigmoid
            mask = torch.sigmoid(logits)
        else:
            # Hard top-K selection
            k = max(int(N * self.keep_ratio), int(N * self.min_keep_ratio))
            k = max(k, 1)
            _, top_indices = scores.squeeze(-1).topk(k, dim=1)  # (B, K)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_indices.unsqueeze(-1), 1.0)

        # Apply mask
        x_routed = x * mask

        return x_routed, scores


class UncertaintyRouter(nn.Module):
    """
    Uncertainty-aware token router using Monte Carlo Dropout.

    Tokens with high predictive variance (uncertainty) are kept,
    while consistently predictable tokens are pruned.

    The intuition: regions the model is uncertain about likely
    contain perceptually important information (edges, objects, texture),
    while confident low-activation regions are uniform backgrounds.

    Args:
        dim: Token feature dimension
        hidden_dim: MLP hidden dimension
        dropout_rate: Dropout rate for MC sampling
        num_samples: Number of MC samples for variance estimation
        keep_ratio: Target keep ratio
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.3,
        num_samples: int = 5,
        keep_ratio: float = 0.7,
    ):
        super().__init__()
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio

        self.score_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def estimate_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-token predictive variance via MC Dropout.

        Args:
            x: (B, N, D) token features
        Returns:
            variance: (B, N, 1) per-token variance estimate
        """
        self.score_net.train()  # Enable dropout for MC sampling

        samples = []
        for _ in range(self.num_samples):
            score = self.score_net(x)  # (B, N, 1)
            samples.append(score)

        samples = torch.stack(samples, dim=0)  # (S, B, N, 1)
        variance = samples.var(dim=0)  # (B, N, 1)
        mean_score = samples.mean(dim=0)  # (B, N, 1)

        return variance, mean_score

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) token features
            training: If True, use MC Dropout; else single pass
        Returns:
            x_routed: (B, N, D) routed features
            variance: (B, N, 1) per-token uncertainty
            mean_score: (B, N, 1) mean saliency score
        """
        B, N, D = x.shape

        if training:
            variance, mean_score = self.estimate_variance(x)
            # Keep tokens with high variance (uncertain) OR high mean score
            routing_score = variance + torch.sigmoid(mean_score)
        else:
            self.score_net.eval()
            mean_score = self.score_net(x)
            variance = torch.zeros_like(mean_score)
            routing_score = torch.sigmoid(mean_score)

        # Top-K selection
        k = max(int(N * self.keep_ratio), 1)
        _, top_indices = routing_score.squeeze(-1).topk(k, dim=1)
        mask = torch.zeros_like(routing_score)
        mask.scatter_(1, top_indices.unsqueeze(-1), 1.0)

        x_routed = x * mask

        return x_routed, variance, mean_score


class PhysicalPriorRouter(nn.Module):
    """
    Physics-driven router using RAW sensor characteristics.

    Computes a "physical saliency" score from the raw Bayer values:
    - Flat/uniform regions (low local variance) → low saliency → prune
    - High-variance, edge, textured regions → high saliency → keep

    The physical prior is combined with learned saliency for
    more robust routing decisions.

    This is a key differentiator from generic token pruning — it uses
    the sensor physics as a structural prior.

    Args:
        dim: Token feature dimension
        patch_size: Size of each patch in pixels (for computing local stats)
        physical_weight: Weight of physical prior vs learned score
        keep_ratio: Target fraction of tokens to keep
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 16,
        physical_weight: float = 0.3,
        keep_ratio: float = 0.7,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.physical_weight = physical_weight
        self.keep_ratio = keep_ratio

        # Learned saliency (same as SaliencyRouter)
        self.learned_router = SaliencyRouter(
            dim=dim,
            hidden_dim=64,
            keep_ratio=keep_ratio,
        )

        # Weight for combining physical + learned
        self.combine_weight = nn.Parameter(torch.tensor(physical_weight))

    def compute_physical_saliency(
        self,
        raw_image: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Compute physical saliency from raw Bayer image patches.

        Uses local variance within each patch as a proxy for
        information content.

        Args:
            raw_image: (B, 1, H, W) original Bayer RAW image
            grid_h: Token grid height
            grid_w: Token grid width
        Returns:
            phys_saliency: (B, N, 1) physical saliency scores [0, 1]
        """
        B, C, H, W = raw_image.shape
        P = self.patch_size

        # Extract patches and compute local variance
        # Use unfold to get patches
        patches = raw_image.unfold(2, P, P).unfold(3, P, P)
        # (B, 1, grid_h, grid_w, P, P)

        patches = patches.contiguous().view(B, grid_h, grid_w, P * P)
        # (B, grid_h, grid_w, P*P)

        # Local variance as saliency proxy
        local_var = patches.var(dim=-1, keepdim=False)  # (B, grid_h, grid_w)
        # Normalize to [0, 1] per image
        var_min = local_var.amin(dim=(1, 2), keepdim=True)
        var_max = local_var.amax(dim=(1, 2), keepdim=True)
        # Avoid division by zero
        var_range = var_max - var_min
        var_range = torch.where(var_range > 1e-8, var_range, torch.ones_like(var_range))
        local_var_norm = (local_var - var_min) / var_range

        # Flatten to token sequence
        phys_saliency = local_var_norm.view(B, grid_h * grid_w, 1)

        return phys_saliency

    def forward(
        self,
        x: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) token features
            raw_image: (B, 1, H, W) original Bayer image (for physical prior)
            grid_h, grid_w: Token grid dimensions
            training: Training or inference mode
        Returns:
            x_routed: (B, N, D) routed features
            combined_score: (B, N, 1) combined routing score
        """
        # Learned saliency
        x_learned, learned_score = self.learned_router(x, training=training)

        if raw_image is not None and grid_h is not None and grid_w is not None:
            # Physical saliency
            phys_score = self.compute_physical_saliency(raw_image, grid_h, grid_w)
            phys_score = phys_score.to(x.device)

            # Combine: w * physical + (1-w) * learned
            w = torch.sigmoid(self.combine_weight)
            combined_score = w * phys_score + (1 - w) * learned_score
        else:
            combined_score = learned_score

        # Top-K routing
        B, N, D = x.shape
        k = max(int(N * self.keep_ratio), 1)
        _, top_indices = combined_score.squeeze(-1).topk(k, dim=1)
        mask = torch.zeros_like(combined_score)
        mask.scatter_(1, top_indices.unsqueeze(-1), 1.0)

        x_routed = x * mask

        return x_routed, combined_score
