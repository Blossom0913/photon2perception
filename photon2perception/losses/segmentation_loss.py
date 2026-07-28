"""
Segmentation loss: Cross-Entropy + Region Mutual Information (RMI) loss.

Implements the loss referenced in configs/segmentation/photon2percept_seg_bayer.yaml
(`seg_loss: ce`, `rmi_loss_weight: 0.1`) and used by RawSeg-Net-style designs
in the reference literature (Lu 2023 RawSeg combines CE + RMI, see
reference_pdf/summary_notes). RMI models spatial pixel dependencies within
local patches rather than treating each pixel independently, which tends to
produce sharper segmentation boundaries.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMILoss(nn.Module):
    """Region Mutual Information loss (Zhao et al., NeurIPS 2019).

    Downsamples prediction/target to a coarser grid, extracts local PxP
    patches as vectors, and maximizes a lower bound on the mutual
    information between predicted and ground-truth patch distributions
    modeled as multivariate Gaussians.

    Args:
        num_classes: Number of segmentation classes.
        rmi_radius: Patch radius P (patch size = 2*radius + 1... here we use
            a square window of side `rmi_radius`, matching common
            implementations that default radius=3).
        downsampling_ratio: Stride for extracting patches (reduces compute).
        eps: Numerical stability term for covariance regularization.
    """

    def __init__(
        self,
        num_classes: int,
        rmi_radius: int = 3,
        downsampling_ratio: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.radius = rmi_radius
        self.downsampling_ratio = downsampling_ratio
        self.eps = eps
        self.half_d = rmi_radius * rmi_radius

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping RxR patches, flattened along the patch dim.

        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, C, R*R, num_patches)
        """
        b, c, h, w = x.shape
        r = self.radius
        # Pad so H, W are divisible by r (edge patches otherwise dropped).
        pad_h = (r - h % r) % r
        pad_w = (r - w % r) % r
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = h + pad_h, w + pad_w

        patches = x.unfold(2, r, r).unfold(3, r, r)  # (B, C, H/r, W/r, r, r)
        patches = patches.contiguous().view(b, c, -1, r * r)  # (B, C, num_patches, r*r)
        return patches.permute(0, 1, 3, 2)  # (B, C, r*r, num_patches)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W) raw segmentation logits.
            targets: (B, H, W) integer class labels, may contain `ignore_index`
                pixels which are treated as class 0 after one-hot but excluded
                from CE separately (RMI here focuses on structural agreement
                and is tolerant of a small amount of label noise at edges).
        Returns:
            Scalar RMI loss (lower is better; encourages high mutual information).
        """
        b, c, h, w = logits.shape

        probs = F.softmax(logits, dim=1)
        valid_targets = targets.clamp(min=0, max=self.num_classes - 1)
        target_one_hot = F.one_hot(valid_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Downsample for computational efficiency (standard RMI practice).
        if self.downsampling_ratio > 1:
            probs = F.avg_pool2d(probs, kernel_size=self.downsampling_ratio, stride=self.downsampling_ratio)
            target_one_hot = F.avg_pool2d(
                target_one_hot, kernel_size=self.downsampling_ratio, stride=self.downsampling_ratio
            )

        pred_patches = self._extract_patches(probs)      # (B, C, r*r, P)
        target_patches = self._extract_patches(target_one_hot)  # (B, C, r*r, P)

        # Merge batch and class dims: RMI treats each (batch, class) channel
        # independently as a P-dimensional multivariate Gaussian over the
        # r*r patch vector.
        pred_patches = pred_patches.reshape(b * c, self.half_d, -1)     # (B*C, D, P)
        target_patches = target_patches.reshape(b * c, self.half_d, -1)  # (B*C, D, P)

        eye = torch.eye(self.half_d, device=logits.device, dtype=logits.dtype).unsqueeze(0)

        pred_centered = pred_patches - pred_patches.mean(dim=-1, keepdim=True)
        target_centered = target_patches - target_patches.mean(dim=-1, keepdim=True)
        num_patches = pred_patches.shape[-1]

        cov_pred = pred_centered @ pred_centered.transpose(-1, -2) / max(num_patches - 1, 1) + self.eps * eye
        cov_target = target_centered @ target_centered.transpose(-1, -2) / max(num_patches - 1, 1) + self.eps * eye
        cov_cross = pred_centered @ target_centered.transpose(-1, -2) / max(num_patches - 1, 1)

        # Conditional covariance of pred given target:
        #   cov_pred|target = cov_pred - cov_cross @ inv(cov_target) @ cov_cross^T
        cov_target_inv = torch.linalg.inv(cov_target)
        cov_cond = cov_pred - cov_cross @ cov_target_inv @ cov_cross.transpose(-1, -2)
        cov_cond = cov_cond + self.eps * eye  # regularize for numerical stability

        # Upper bound on -log det via Cholesky-based logdet (stable & differentiable).
        chol = torch.linalg.cholesky(cov_cond)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1).clamp(min=1e-6)).sum(dim=-1)

        # RMI lower bound: (1/2) * log det(cov_pred|target); minimizing this
        # maximizes the mutual information bound between pred and target.
        rmi = 0.5 * log_det
        return rmi.mean()


class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy + RMI loss for semantic segmentation.

    loss = ce_weight * CE(pred, target) - rmi_weight * MI_lower_bound
    (RMI is subtracted because we want to *maximize* mutual information,
    matching the sign convention documented in the RawSeg paper summary:
    `L_seg = lambda_ce * L_ce - lambda_rmi * I_l(Y, Y_hat)`).

    Args:
        num_classes: Number of segmentation classes.
        ce_weight: Weight for the cross-entropy term.
        rmi_weight: Weight for the RMI term (0 disables it entirely, which
            also skips the (more expensive) patch-covariance computation).
        ignore_index: Label value to ignore in the CE loss (e.g. 255).
        aux_weight: Optional weight for an auxiliary loss on a secondary
            (lower-resolution / intermediate) prediction, matching the
            `aux_loss_weight` field in the segmentation YAML config.
    """

    def __init__(
        self,
        num_classes: int = 19,
        ce_weight: float = 1.0,
        rmi_weight: float = 0.1,
        ignore_index: int = 255,
        aux_weight: float = 0.4,
        rmi_radius: int = 3,
        rmi_downsampling_ratio: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.rmi_weight = rmi_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.rmi_loss = RMILoss(
            num_classes=num_classes,
            rmi_radius=rmi_radius,
            downsampling_ratio=rmi_downsampling_ratio,
        ) if rmi_weight > 0 else None

    def _single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        loss = self.ce_weight * self.ce_loss(logits, targets)
        if self.rmi_loss is not None:
            loss = loss + self.rmi_weight * self.rmi_loss(logits, targets)
        return loss

    def forward(
        self,
        seg_logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            seg_logits: (B, num_classes, H, W) main prediction logits.
            targets: (B, H, W) integer class labels.
            aux_logits: Optional (B, num_classes, H', W') auxiliary head
                prediction (e.g. from an intermediate feature map) for deep
                supervision.
        Returns:
            Dict with 'loss_main', 'loss_aux' (0 if unused), 'loss_total'.
        """
        loss_main = self._single_loss(seg_logits, targets)

        loss_aux = seg_logits.new_zeros(())
        if aux_logits is not None and self.aux_weight > 0:
            loss_aux = self._single_loss(aux_logits, targets)

        loss_total = loss_main + self.aux_weight * loss_aux
        return {
            'loss_main': loss_main,
            'loss_aux': loss_aux,
            'loss_total': loss_total,
        }
