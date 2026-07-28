#!/usr/bin/env python3
"""
Evaluation script for Photon2Perception models.

Computes standard task metrics (COCO-style mAP for detection, mIoU for
segmentation, via photon2perception.evaluation.metrics) plus the efficiency
report (latency/FLOPs/memory/bandwidth, via
photon2perception.evaluation.efficiency), given a trained checkpoint.

Example:
    python tools/eval.py --config configs/detection/photon2percept_det_bayer.yaml \\
        --checkpoint outputs/photon2percept_det_bayer/checkpoint_best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.evaluation.efficiency import full_efficiency_report
from photon2perception.evaluation.metrics import DetectionEvaluator, SegmentationEvaluator
from photon2perception.models.heads.postprocess import postprocess_detections
from photon2perception.models.model_wrapper import build_perception_model
from photon2perception.utils.checkpoint import load_weights_only
from photon2perception.utils.config import apply_cli_overrides, load_config
from tools.train import build_dataloaders
from photon2perception.utils.distributed import DistributedInfo


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Photon2Perception models')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None, help='Path to save JSON results')
    parser.add_argument('--skip_efficiency', action='store_true', help='Skip latency/FLOPs benchmark')
    parser.add_argument('--score_thresh', type=float, default=0.05)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--override', nargs='+', default=None)
    return parser.parse_args()


def _single_process_dist_info(device: torch.device) -> DistributedInfo:
    return DistributedInfo(rank=0, world_size=1, local_rank=0, is_distributed=False, device=device)


@torch.no_grad()
def evaluate_detection(model, val_loader, config, device, score_thresh, nms_thresh):
    num_classes = config['data']['num_classes']
    strides = model.get_strides()
    evaluator = DetectionEvaluator(num_classes=num_classes)

    model.eval()
    for batch in val_loader:
        images = batch['image'].to(device)
        cls_scores, bbox_preds = model(images)
        preds = postprocess_detections(
            cls_scores, bbox_preds, strides=strides, num_classes=num_classes,
            score_thresh=score_thresh, nms_thresh=nms_thresh,
            image_size=tuple(images.shape[-2:]),
        )
        image_sizes = [tuple(images.shape[-2:])] * images.shape[0]
        evaluator.update(batch['image_id'], preds, batch['targets'], image_sizes=image_sizes)

    return evaluator.compute()


@torch.no_grad()
def evaluate_segmentation(model, val_loader, config, device):
    num_classes = config['data']['num_classes']
    evaluator = SegmentationEvaluator(num_classes=num_classes, ignore_index=255)

    model.eval()
    for batch in val_loader:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        seg_logits = model(images)
        preds = seg_logits.argmax(dim=1)
        evaluator.update(preds, targets)

    return evaluator.compute()


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    device = torch.device('cuda' if torch.cuda.is_available() else
                           ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    model = build_perception_model(config).to(device)
    load_weights_only(args.checkpoint, model, map_location='cpu', strict=True)
    model.eval()

    dist_info = _single_process_dist_info(device)
    _, val_loader = build_dataloaders(config, dist_info)
    if val_loader is None:
        raise ValueError(
            "No validation set configured (data.val_img_dir/val_ann_file for detection, "
            "or the 'val' split for segmentation). Cannot compute task metrics."
        )

    task = config['task']
    if task == 'detection':
        task_metrics = evaluate_detection(model, val_loader, config, device,
                                           args.score_thresh, args.nms_thresh)
    else:
        task_metrics = evaluate_segmentation(model, val_loader, config, device)

    print("Task metrics:")
    for k, v in task_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    results = {'config': args.config, 'checkpoint': args.checkpoint, 'task_metrics': task_metrics}

    if not args.skip_efficiency:
        img_size = tuple(config['model']['img_size'])
        input_shape = (1, config['model'].get('in_chans', 1), *img_size)
        print(f"Running efficiency benchmark at input shape {input_shape} on {device}...")
        results['efficiency'] = full_efficiency_report(
            model, input_shape, input_format='bayer', device=str(device)
        )

    output_path = args.output or str(Path(args.checkpoint).with_suffix('.eval.json'))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
