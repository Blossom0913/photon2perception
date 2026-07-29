#!/usr/bin/env python3
"""
Evaluation CLI for Photon2Perception models.

Thin entry point: parses CLI args, resolves the config, picks the right
per-task `*TaskEvaluator` subclass based on `config['task']`
(`tasks.detection.evaluator.DetectionTaskEvaluator` /
`tasks.segmentation.evaluator.SegmentationTaskEvaluator`), and calls
`.evaluate()`. Model/checkpoint loading, dataloader construction, and the
efficiency report all live in `photon2perception.engine.base_evaluator.BaseEvaluator`,
shared by both tasks; only the task metric itself (COCO mAP vs. mIoU)
differs per subclass.

Example:
    python tools/eval.py --config tasks/detection/config/photon2percept_det_bayer.yaml \\
        --checkpoint outputs/photon2percept_det_bayer/checkpoint_best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.common.config import apply_cli_overrides, load_config
from tasks.detection.evaluator import DetectionTaskEvaluator
from tasks.segmentation.evaluator import SegmentationTaskEvaluator

TASK_EVALUATORS = {
    'detection': DetectionTaskEvaluator,
    'segmentation': SegmentationTaskEvaluator,
}


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


def build_evaluator(config, args):
    """Instantiate the `*TaskEvaluator` subclass matching `config['task']`."""
    task = config['task']
    device = torch.device('cuda' if torch.cuda.is_available() else
                           ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    if task == 'detection':
        return DetectionTaskEvaluator(
            config, args.checkpoint, score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh, device=device,
        )
    if task == 'segmentation':
        return SegmentationTaskEvaluator(config, args.checkpoint, device=device)
    raise ValueError(f"Unknown task '{task}', expected one of {list(TASK_EVALUATORS)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    evaluator = build_evaluator(config, args)
    results = evaluator.evaluate(skip_efficiency=args.skip_efficiency)

    print("Task metrics:")
    for k, v in results['task_metrics'].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    if not args.skip_efficiency:
        img_size = tuple(config['model']['img_size'])
        print(f"Efficiency benchmark computed at input shape "
              f"(1, {config['model'].get('in_chans', 1)}, {img_size[0]}, {img_size[1]})")

    output_path = args.output or str(Path(args.checkpoint).with_suffix('.eval.json'))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
