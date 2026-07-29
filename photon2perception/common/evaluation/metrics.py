"""
Task evaluation metrics: COCO-style mAP for detection, mIoU for segmentation.

`BenchmarkRunner.run_task_benchmark` (common/evaluation/benchmarks/runner.py)
had a placeholder returning `{}` with a comment "integrate with mmdet/mmseg
evaluation". This module implements standalone, dependency-light versions:

- Detection: uses `pycocotools.cocoeval.COCOeval` if available (the de facto
  standard COCO mAP implementation, already a transitive dependency via
  `photon2perception.common.dataset.coco_raw_dataset.CocoRawDetectionDataset`).
  Falls back to a simple single-IoU-threshold AP computation if
  pycocotools isn't installed, so this module never hard-requires it.
- Segmentation: mIoU via a running confusion matrix (standard
  Cityscapes-style evaluation protocol), no external dependency needed.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Segmentation: mIoU via confusion matrix
# ----------------------------------------------------------------------------

class SegmentationEvaluator:
    """Accumulates a confusion matrix across batches and reports mIoU /
    per-class IoU / pixel accuracy, following the standard
    Cityscapes/PASCAL-VOC evaluation protocol.

    Usage:
        evaluator = SegmentationEvaluator(num_classes=19, ignore_index=255)
        for batch in val_loader:
            preds = model(batch['image']).argmax(dim=1)
            evaluator.update(preds, batch['targets'])
        metrics = evaluator.compute()  # {'mIoU': ..., 'pixel_acc': ..., 'iou_per_class': [...]}
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            preds: (B, H, W) predicted class indices.
            targets: (B, H, W) ground-truth class indices (may contain
                `ignore_index`, which is excluded from the confusion matrix).
        """
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)

        valid = targets_np != self.ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Also guard against out-of-range predictions (shouldn't happen with
        # a correctly-sized head, but avoids a hard crash during early/buggy
        # training runs where logits might not have converged to expected shapes).
        valid_range = (preds_np >= 0) & (preds_np < self.num_classes) & \
                      (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np = preds_np[valid_range]
        targets_np = targets_np[valid_range]

        indices = self.num_classes * targets_np + preds_np
        cm_update = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm_update.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix.astype(np.float64)
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        iou_per_class = np.divide(
            intersection, union, out=np.full_like(intersection, np.nan), where=union > 0
        )
        mean_iou = np.nanmean(iou_per_class)
        pixel_acc = intersection.sum() / max(cm.sum(), 1.0)

        return {
            'mIoU': float(mean_iou),
            'pixel_acc': float(pixel_acc),
            'iou_per_class': [float(x) if not np.isnan(x) else None for x in iou_per_class],
        }


# ----------------------------------------------------------------------------
# Detection: COCO-style mAP
# ----------------------------------------------------------------------------

class DetectionEvaluator:
    """Accumulates predictions/ground-truth and reports COCO-style mAP.

    Uses `pycocotools.cocoeval.COCOeval` when available for an exact,
    standard-compliant AP/AP50/AP75 computation. Falls back to a simplified
    single-IoU-threshold (0.5) average precision if pycocotools is not
    installed, clearly labeled as an approximation.

    Usage:
        evaluator = DetectionEvaluator(num_classes=80)
        for batch, (cls_scores, bbox_preds) in ...:
            decoded = decode_predictions(...)  # list of {'boxes','scores','labels'} per image
            evaluator.update(image_ids, decoded, batch['targets'])
        metrics = evaluator.compute()  # {'mAP': ..., 'AP50': ..., 'AP75': ...} (or approx_AP50)
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.predictions: List[Dict] = []  # COCO-format detection results
        self.ground_truth: List[Dict] = []  # COCO-format annotations
        self.images: List[Dict] = []
        self._ann_id = 0

    def reset(self) -> None:
        self.predictions.clear()
        self.ground_truth.clear()
        self.images.clear()
        self._ann_id = 0

    def update(
        self,
        image_ids: List[int],
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        image_sizes: Optional[List[tuple]] = None,
    ) -> None:
        """
        Args:
            image_ids: List of unique image identifiers (len B).
            predictions: List of dicts with 'boxes' (N,4) xyxy, 'scores' (N,),
                'labels' (N,) (0-indexed class ids), one per image.
            targets: List of dicts with 'boxes' (G,4) xyxy, 'labels' (G,)
                (0-indexed), one per image (ground truth).
            image_sizes: Optional list of (H, W) per image (used to populate
                COCO 'images' metadata; defaults to a placeholder size which
                doesn't affect AP computation).
        """
        for i, img_id in enumerate(image_ids):
            h, w = image_sizes[i] if image_sizes else (1, 1)
            self.images.append({'id': int(img_id), 'height': int(h), 'width': int(w)})

            gt_boxes = targets[i]['boxes'].detach().cpu().numpy()
            gt_labels = targets[i]['labels'].detach().cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.tolist()
                self._ann_id += 1
                self.ground_truth.append({
                    'id': self._ann_id,
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': max(x2 - x1, 0) * max(y2 - y1, 0),
                    'iscrowd': 0,
                })

            pred_boxes = predictions[i]['boxes'].detach().cpu().numpy()
            pred_scores = predictions[i]['scores'].detach().cpu().numpy()
            pred_labels = predictions[i]['labels'].detach().cpu().numpy()
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box.tolist()
                self.predictions.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score),
                })

    def compute(self) -> Dict[str, float]:
        try:
            return self._compute_pycocotools()
        except ImportError:
            return self._compute_approx()

    def _compute_pycocotools(self) -> Dict[str, float]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt_json = {
            'images': self.images,
            'annotations': self.ground_truth,
            'categories': [{'id': i, 'name': str(i)} for i in range(self.num_classes)],
        }
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()

        if len(self.predictions) == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

        coco_dt = coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            'mAP': float(coco_eval.stats[0]),
            'AP50': float(coco_eval.stats[1]),
            'AP75': float(coco_eval.stats[2]),
            'AP_small': float(coco_eval.stats[3]),
            'AP_medium': float(coco_eval.stats[4]),
            'AP_large': float(coco_eval.stats[5]),
        }

    def _compute_approx(self) -> Dict[str, float]:
        """Simplified single-IoU-threshold (0.5) AP, used only when
        pycocotools is unavailable. Not a substitute for the real COCO
        metric -- intended for quick sanity checks during development on
        machines without pycocotools installed.
        """
        from photon2perception.common.loss.detection_loss import box_iou

        preds_by_image: Dict[int, List[Dict]] = {}
        for p in self.predictions:
            preds_by_image.setdefault(p['image_id'], []).append(p)
        gt_by_image: Dict[int, List[Dict]] = {}
        for g in self.ground_truth:
            gt_by_image.setdefault(g['image_id'], []).append(g)

        aps = []
        for class_id in range(self.num_classes):
            tp, fp, num_gt = [], [], 0
            for img in self.images:
                img_id = img['id']
                gts = [g for g in gt_by_image.get(img_id, []) if g['category_id'] == class_id]
                preds = sorted(
                    [p for p in preds_by_image.get(img_id, []) if p['category_id'] == class_id],
                    key=lambda p: -p['score'],
                )
                num_gt += len(gts)
                matched = [False] * len(gts)
                if not gts:
                    fp.extend([1] * len(preds))
                    tp.extend([0] * len(preds))
                    continue
                gt_boxes = torch.tensor([
                    [g['bbox'][0], g['bbox'][1], g['bbox'][0] + g['bbox'][2], g['bbox'][1] + g['bbox'][3]]
                    for g in gts
                ])
                for p in preds:
                    x, y, w, h = p['bbox']
                    pred_box = torch.tensor([[x, y, x + w, y + h]])
                    ious = box_iou(pred_box, gt_boxes)[0]
                    best_iou, best_idx = ious.max(0)
                    if best_iou.item() >= 0.5 and not matched[best_idx.item()]:
                        matched[best_idx.item()] = True
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)
            if num_gt == 0:
                continue
            tp_arr = np.cumsum(tp)
            fp_arr = np.cumsum(fp)
            recalls = tp_arr / max(num_gt, 1)
            precisions = tp_arr / np.maximum(tp_arr + fp_arr, 1)
            ap = _voc_ap(recalls, precisions)
            aps.append(ap)

        return {'approx_AP50': float(np.mean(aps)) if aps else 0.0}


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated average precision (PASCAL VOC style)."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        p = np.max(precisions[mask]) if mask.any() else 0.0
        ap += p / 11.0
    return ap
