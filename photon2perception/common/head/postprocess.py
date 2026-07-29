"""
Detection post-processing: anchor decoding, score thresholding, and NMS.

`RawDetectionHead` outputs raw per-level classification logits and box
regression deltas; this module turns those into final
(boxes, scores, labels) predictions, the format expected by
`photon2perception.evaluation.metrics.DetectionEvaluator` and by any
downstream visualization/deployment code.

Kept separate from `DetectionLoss` (which only needs anchors + encode/decode
for *training*-time target assignment) since post-processing has a
different set of concerns: per-class score thresholding, top-K
pre-NMS filtering, and NMS itself, none of which are needed during the
loss computation.
"""

from typing import Dict, List, Tuple

import torch
import torchvision

from ...losses.detection_loss import decode_boxes, generate_anchors


@torch.no_grad()
def postprocess_detections(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    strides: List[int],
    num_classes: int,
    anchor_scales: Tuple[float, ...] = (1.0, 1.26, 1.59),
    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    max_pre_nms: int = 1000,
    max_detections: int = 100,
    image_size: Tuple[int, int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Decode raw detection head outputs into final per-image predictions.

    Args:
        cls_scores: List of (B, A*num_classes, H, W) per FPN level (raw
            logits, as returned by `RawDetectionHead.forward`).
        bbox_preds: List of (B, A*4, H, W) per FPN level.
        strides: Effective input-image stride of each level (from
            `PerceptionModel.get_strides()`).
        num_classes: Number of foreground classes.
        anchor_scales, anchor_ratios: Must match the values used when
            anchors were assigned during training (`DetectionLoss`'s
            defaults are used here too, for consistency).
        score_thresh: Discard boxes with max class score below this.
        nms_thresh: IoU threshold for class-wise NMS.
        max_pre_nms: Keep at most this many highest-scoring boxes per image
            *before* NMS (standard RetinaNet-style efficiency guard).
        max_detections: Keep at most this many boxes per image *after* NMS.
        image_size: Optional (H, W) to clip decoded boxes into bounds.
    Returns:
        List (length B) of dicts: {'boxes': (K,4) xyxy, 'scores': (K,),
        'labels': (K,) int64 in [0, num_classes)}.
    """
    device = cls_scores[0].device
    batch_size = cls_scores[0].shape[0]
    num_anchors_per_loc = len(anchor_scales) * len(anchor_ratios)

    anchors_per_level = []
    for level, stride in enumerate(strides):
        h, w = cls_scores[level].shape[-2:]
        anchors_per_level.append(
            generate_anchors((h, w), stride, anchor_scales=anchor_scales,
                              anchor_ratios=anchor_ratios, device=device)
        )
    all_anchors = torch.cat(anchors_per_level, dim=0)  # (A_total, 4)

    flat_cls, flat_reg = [], []
    for level in range(len(cls_scores)):
        b, _, h, w = cls_scores[level].shape
        cls_lvl = cls_scores[level].view(b, num_anchors_per_loc, num_classes, h, w)
        cls_lvl = cls_lvl.permute(0, 3, 4, 1, 2).reshape(b, -1, num_classes)
        flat_cls.append(cls_lvl)

        reg_lvl = bbox_preds[level].view(b, num_anchors_per_loc, 4, h, w)
        reg_lvl = reg_lvl.permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
        flat_reg.append(reg_lvl)

    flat_cls = torch.cat(flat_cls, dim=1)  # (B, A_total, num_classes)
    flat_reg = torch.cat(flat_reg, dim=1)  # (B, A_total, 4)
    flat_scores = flat_cls.sigmoid()

    results = []
    for b in range(batch_size):
        scores_b = flat_scores[b]  # (A_total, num_classes)
        max_scores, max_labels = scores_b.max(dim=1)  # (A_total,)

        keep_score = max_scores > score_thresh
        if keep_score.sum() == 0:
            results.append({
                'boxes': torch.zeros(0, 4, device=device),
                'scores': torch.zeros(0, device=device),
                'labels': torch.zeros(0, dtype=torch.long, device=device),
            })
            continue

        anchors_kept = all_anchors[keep_score]
        deltas_kept = flat_reg[b][keep_score]
        scores_kept = max_scores[keep_score]
        labels_kept = max_labels[keep_score]

        if scores_kept.numel() > max_pre_nms:
            top_scores, top_idx = scores_kept.topk(max_pre_nms)
            anchors_kept = anchors_kept[top_idx]
            deltas_kept = deltas_kept[top_idx]
            scores_kept = top_scores
            labels_kept = labels_kept[top_idx]

        boxes = decode_boxes(anchors_kept, deltas_kept)
        if image_size is not None:
            h, w = image_size
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        # Class-wise NMS: offset boxes by class id so torchvision's batched
        # NMS treats different classes as non-overlapping (standard trick,
        # avoids a Python-level per-class loop).
        max_coord = boxes.max() if boxes.numel() > 0 else torch.tensor(0.0, device=device)
        offsets = labels_kept.to(boxes.dtype) * (max_coord + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep_nms = torchvision.ops.nms(boxes_for_nms, scores_kept, nms_thresh)
        keep_nms = keep_nms[:max_detections]

        results.append({
            'boxes': boxes[keep_nms],
            'scores': scores_kept[keep_nms],
            'labels': labels_kept[keep_nms],
        })

    return results
