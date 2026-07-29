#!/usr/bin/env python3
"""
Training CLI for Photon2Perception models.

Thin entry point: parses CLI args, resolves the config, picks the right
per-task `Trainer` subclass based on `config['task']`
(`tasks.detection.trainer.DetectionTrainer` /
`tasks.segmentation.trainer.SegmentationTrainer`), and calls `.fit()`. All
the actual training logic (dataloader construction, optimizer/scheduler,
the train/validate loop, checkpointing, mixed precision, logging) lives in
`photon2perception.engine.base_trainer.BaseTrainer`, shared by both tasks.

Example:
    python tools/train.py --config tasks/detection/config/photon2percept_det_bayer.yaml
    python tools/train.py --config tasks/detection/config/photon2percept_det_bayer.yaml \\
        --override training.epochs=5 data.batch_size=2 --output_dir ./outputs/debug
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.common.config import apply_cli_overrides, load_config
from photon2perception.engine.base_trainer import BaseTrainer
from tasks.detection.trainer import DetectionTrainer
from tasks.segmentation.trainer import SegmentationTrainer

TASK_TRAINERS = {
    'detection': DetectionTrainer,
    'segmentation': SegmentationTrainer,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Photon2Perception')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from a specific checkpoint')
    parser.add_argument('--auto_resume', action='store_true',
                         help='Resume from the latest checkpoint in output_dir/exp_name, if any')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                         help='Experiment name (subdirectory of output_dir); defaults to the config filename stem')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default=None,
                         help='Enable Weights & Biases logging (overrides config.logging.use_wandb)')
    parser.add_argument('--no_tensorboard', action='store_true',
                         help='Disable TensorBoard logging (overrides config.logging.use_tensorboard)')
    parser.add_argument('--wandb_project', type=str, default=None,
                         help='Overrides config.logging.wandb_project')
    parser.add_argument(
        '--override', nargs='+', default=None,
        help="Dotted-key config overrides, e.g. training.epochs=5 data.batch_size=2",
    )
    return parser.parse_args()


def build_trainer(config, args) -> BaseTrainer:
    """Instantiate the `Trainer` subclass matching `config['task']`."""
    trainer_cls = TASK_TRAINERS.get(config['task'])
    if trainer_cls is None:
        raise ValueError(f"Unknown task '{config['task']}', expected one of {list(TASK_TRAINERS)}")

    exp_name = args.exp_name or Path(args.config).stem
    output_dir = str(Path(args.output_dir) / exp_name)

    logging_cfg = config.get('logging', {})
    use_tensorboard = logging_cfg.get('use_tensorboard', True) and not args.no_tensorboard
    use_wandb = args.use_wandb if args.use_wandb is not None else logging_cfg.get('use_wandb', False)
    wandb_project = args.wandb_project or logging_cfg.get('wandb_project', 'photon2perception')

    return trainer_cls(
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        resume=args.resume,
        auto_resume=args.auto_resume,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )


def main():
    args = parse_args()

    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    trainer = build_trainer(config, args)
    trainer.logger.log(f"Config: {args.config}")
    trainer.fit()


if __name__ == '__main__':
    main()
