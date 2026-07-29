"""
Detection loss: Focal loss (classification) + L1/GIoU loss (regression).

Implements the loss described in tasks/detection/config/photon2percept_det_bayer.yaml
(`cls_loss: focal`, `reg_loss: l1`) and referenced in CLAUDE.md's "Known gaps"
(tools/train.py previously used a dummy `loss = cls_token.sum() * 0.0`).

Design notes:
- Follows the standard RetinaNet-style anchor-based formulation (matches
  `RawDetectionHead`'s num_anchors * num_classes / num_anchors * 4 output
  layout), since that's what the existing detection head already produces.
- Anchor generation, target assignment (IoU matching) and loss computation
  are all pure-tensor operations with statically-known shapes at trace time
  (no data-dependent Python control flow beyond looping over a fixed number
  of FPN levels), which keeps this compatible with `torch.jit.trace` if the
  loss is ever included in a traced training step.
- GIoU loss is provided as an optional addition to L1 (see BayerDetect's
  `L1 + GIoU` design in the reference literature), configurable via
  `reg_loss_type`.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Classification: Focal Loss
# ----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Sigmoid focal loss for dense object detection (Lin et al., 2017).

    Args:
        alpha: Weighting factor for the rare (positive) class.
        gamma: Focusing parameter that down-weights easy examples.
        reduction: 'sum', 'mean', or 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw classification logits.
            targets: (N, C) one-hot (or multi-hot) targets in {0, 1}.
        Returns:
            Scalar loss (or (N, C) if reduction='none').
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss


# ----------------------------------------------------------------------------
# Regression: GIoU Loss
# ----------------------------------------------------------------------------

class GIoULoss(nn.Module):
    """Generalized IoU loss for bounding box regression (Rezatofighi et al., 2019).

    Boxes are expected in (x1, y1, x2, y2) absolute coordinate format.
    """

    def __init__(self, reduction: str = 'sum', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) predicted boxes (x1, y1, x2, y2).
            target_boxes: (N, 4) target boxes (x1, y1, x2, y2).
        Returns:
            Scalar GIoU loss (or (N,) if reduction='none').
        """
        giou = self._giou(pred_boxes, target_boxes)
        loss = 1.0 - giou
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss

    def _giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        union = area1 + area2 - inter + eps
        iou = inter / union

        enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

        giou = iou - (enclose_area - union) / enclose_area
        return giou


# ----------------------------------------------------------------------------
# Anchor generation
# ----------------------------------------------------------------------------

def generate_anchors(
    feat_size: Tuple[int, int],
    stride: int,
    anchor_scales: Tuple[float, ...] = (1.0, 1.26, 1.59),
    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    base_size: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate RetinaNet-style anchors for a single FPN level.

    Args:
        feat_size: (H, W) of the feature map at this level.
        stride: Downsampling stride of this feature map relative to input.
        anchor_scales: Per-location scale multipliers.
        anchor_ratios: Per-location aspect ratios (w/h).
        base_size: Base anchor size; defaults to `stride * 4` (standard
            RetinaNet convention).
        device, dtype: Placement of the output tensor.
    Returns:
        anchors: (H*W*num_anchors, 4) tensor of (x1, y1, x2, y2) anchors in
            input-image coordinates. num_anchors = len(scales) * len(ratios).
    """
    if base_size is None:
        base_size = stride * 4.0

    h, w = feat_size
    shift_x = (torch.arange(0, w, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(0, h, device=device, dtype=dtype) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')  # (H, W)
    centers = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)  # (H*W, 2)

    base_wh = []
    for scale in anchor_scales:
        for ratio in anchor_ratios:
            anchor_w = base_size * scale * (ratio ** 0.5)
            anchor_h = base_size * scale / (ratio ** 0.5)
            base_wh.append((anchor_w, anchor_h))
    base_wh = torch.tensor(base_wh, device=device, dtype=dtype)  # (num_anchors, 2)

    num_anchors = base_wh.shape[0]
    centers_exp = centers.unsqueeze(1).expand(-1, num_anchors, -1)  # (H*W, A, 2)
    wh_exp = base_wh.unsqueeze(0).expand(centers.shape[0], -1, -1)  # (H*W, A, 2)

    x1y1 = centers_exp - wh_exp / 2
    x2y2 = centers_exp + wh_exp / 2
    anchors = torch.cat([x1y1, x2y2], dim=-1).reshape(-1, 4)  # (H*W*A, 4)
    return anchors


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        iou: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Encode ground-truth boxes as regression targets relative to anchors
    (standard (dx, dy, dw, dh) parameterization, Faster R-CNN / RetinaNet style).
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_cx - anchor_cx) / anchor_w.clamp(min=1e-6)
    dy = (gt_cy - anchor_cy) / anchor_h.clamp(min=1e-6)
    dw = torch.log(gt_w / anchor_w.clamp(min=1e-6))
    dh = torch.log(gt_h / anchor_h.clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Inverse of `encode_boxes`: turn (dx, dy, dw, dh) predictions + anchors
    back into (x1, y1, x2, y2) boxes. Used both by the loss (to compute GIoU
    on decoded boxes) and by inference-time post-processing.
    """
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    dx, dy, dw, dh = deltas.unbind(-1)
    dw = dw.clamp(max=4.0)  # avoid exp() overflow, standard detection-repo guard
    dh = dh.clamp(max=4.0)

    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = torch.exp(dw) * anchor_w
    pred_h = torch.exp(dh) * anchor_h

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ----------------------------------------------------------------------------
# Full detection loss (anchor assignment + focal + L1/GIoU)
# ----------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """RetinaNet-style detection loss: anchor generation, target assignment,
    focal classification loss, and L1/GIoU regression loss.

    Args:
        num_classes: Number of foreground classes (excludes background).
        strides: FPN strides, one per feature level (must match the order of
            `cls_scores`/`bbox_preds` passed to `forward`).
        anchor_scales, anchor_ratios: Passed to `generate_anchors`.
        pos_iou_thr: IoU threshold above which an anchor is a positive match.
        neg_iou_thr: IoU threshold below which an anchor is a negative match
            (anchors between neg and pos thresholds are ignored).
        cls_weight, reg_weight: Loss term weights (matches the YAML config's
            `loss.cls_weight` / `loss.reg_weight`).
        reg_loss_type: 'l1', 'smooth_l1', or 'giou'.
        focal_alpha, focal_gamma: FocalLoss hyperparameters.
    """

    def __init__(
        self,
        num_classes: int = 80,
        strides: Tuple[int, ...] = (8, 16, 32, 64, 128),
        anchor_scales: Tuple[float, ...] = (1.0, 1.26, 1.59),
        anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.4,
        cls_weight: float = 2.0,
        reg_weight: float = 1.0,
        reg_loss_type: str = 'l1',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.reg_loss_type = reg_loss_type
        self.num_anchors_per_loc = len(anchor_scales) * len(anchor_ratios)

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')
        self.giou_loss = GIoULoss(reduction='sum')

    @torch.no_grad()
    def _assign_targets(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign each anchor a classification target and matched GT box.

        Args:
            anchors: (A, 4)
            gt_boxes: (G, 4) — G may be 0 (image with no annotations).
            gt_labels: (G,) integer class indices in [0, num_classes).
        Returns:
            cls_targets: (A, num_classes) one-hot targets (all-zero for negatives).
            matched_gt_boxes: (A, 4) GT box assigned to each anchor (only
                meaningful where pos_mask is True).
            pos_mask: (A,) bool, True where the anchor is a positive match.
            valid_mask: (A,) bool, True where the anchor contributes to the
                loss at all (positives + negatives; ignores in-between IoU anchors).
        """
        num_anchors = anchors.shape[0]
        device = anchors.device

        if gt_boxes.numel() == 0:
            cls_targets = torch.zeros(num_anchors, self.num_classes, device=device)
            matched_gt_boxes = torch.zeros(num_anchors, 4, device=device)
            pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
            valid_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
            return cls_targets, matched_gt_boxes, pos_mask, valid_mask

        iou = box_iou(anchors, gt_boxes)  # (A, G)
        max_iou, matched_idx = iou.max(dim=1)  # (A,)

        pos_mask = max_iou >= self.pos_iou_thr
        neg_mask = max_iou < self.neg_iou_thr
        valid_mask = pos_mask | neg_mask

        # Guarantee each GT has at least one positive anchor (best-match rule).
        best_anchor_per_gt = iou.argmax(dim=0)  # (G,)
        pos_mask[best_anchor_per_gt] = True
        valid_mask[best_anchor_per_gt] = True
        matched_idx[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0], device=device)

        matched_gt_boxes = gt_boxes[matched_idx]
        matched_labels = gt_labels[matched_idx]

        cls_targets = torch.zeros(num_anchors, self.num_classes, device=device)
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        cls_targets[pos_indices, matched_labels[pos_indices]] = 1.0

        return cls_targets, matched_gt_boxes, pos_mask, valid_mask

    def forward(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_scores: List of (B, A*num_classes, H, W) per FPN level.
            bbox_preds: List of (B, A*4, H, W) per FPN level.
            targets: List (len B) of dicts with:
                - 'boxes': (G_i, 4) tensor (x1, y1, x2, y2) in input coords.
                - 'labels': (G_i,) long tensor of class indices.
        Returns:
            Dict with 'loss_cls', 'loss_reg', 'loss_total', and 'num_pos'
            (useful for logging / debugging anchor assignment health).
        """
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # Build anchors once per level (shapes are static for a fixed input size).
        anchors_per_level = []
        for level, stride in enumerate(self.strides):
            h, w = cls_scores[level].shape[-2:]
            anchors_per_level.append(
                generate_anchors(
                    (h, w), stride,
                    anchor_scales=self.anchor_scales,
                    anchor_ratios=self.anchor_ratios,
                    device=device,
                )
            )
        all_anchors = torch.cat(anchors_per_level, dim=0)  # (A_total, 4)

        # Flatten predictions to (B, A_total, C) / (B, A_total, 4).
        flat_cls, flat_reg = [], []
        for level in range(len(cls_scores)):
            b, _, h, w = cls_scores[level].shape
            cls_lvl = cls_scores[level].view(b, self.num_anchors_per_loc, self.num_classes, h, w)
            cls_lvl = cls_lvl.permute(0, 3, 4, 1, 2).reshape(b, -1, self.num_classes)
            flat_cls.append(cls_lvl)

            reg_lvl = bbox_preds[level].view(b, self.num_anchors_per_loc, 4, h, w)
            reg_lvl = reg_lvl.permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            flat_reg.append(reg_lvl)

        flat_cls = torch.cat(flat_cls, dim=1)  # (B, A_total, num_classes)
        flat_reg = torch.cat(flat_reg, dim=1)  # (B, A_total, 4)

        total_cls_loss = flat_cls.new_zeros(())
        total_reg_loss = flat_reg.new_zeros(())
        total_num_pos = 0

        for b in range(batch_size):
            gt_boxes = targets[b]['boxes'].to(device)
            gt_labels = targets[b]['labels'].to(device)

            cls_targets, matched_gt_boxes, pos_mask, valid_mask = self._assign_targets(
                all_anchors, gt_boxes, gt_labels
            )
            num_pos = pos_mask.sum().clamp(min=1)
            total_num_pos += int(pos_mask.sum().item())

            cls_loss = self.focal_loss(flat_cls[b][valid_mask], cls_targets[valid_mask])
            total_cls_loss = total_cls_loss + cls_loss / num_pos

            if pos_mask.any():
                pos_anchors = all_anchors[pos_mask]
                pos_preds = flat_reg[b][pos_mask]
                pos_targets_boxes = matched_gt_boxes[pos_mask]

                if self.reg_loss_type == 'giou':
                    decoded = decode_boxes(pos_anchors, pos_preds)
                    reg_loss = self.giou_loss(decoded, pos_targets_boxes)
                else:
                    reg_targets = encode_boxes(pos_anchors, pos_targets_boxes)
                    if self.reg_loss_type == 'smooth_l1':
                        reg_loss = F.smooth_l1_loss(pos_preds, reg_targets, reduction='sum')
                    else:
                        reg_loss = F.l1_loss(pos_preds, reg_targets, reduction='sum')
                total_reg_loss = total_reg_loss + reg_loss / num_pos

        total_cls_loss = total_cls_loss / batch_size
        total_reg_loss = total_reg_loss / batch_size

        loss_total = self.cls_weight * total_cls_loss + self.reg_weight * total_reg_loss

        return {
            'loss_cls': total_cls_loss,
            'loss_reg': total_reg_loss,
            'loss_total': loss_total,
            'num_pos': torch.tensor(float(total_num_pos)),
        }
