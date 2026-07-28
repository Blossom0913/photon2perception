"""
Unified experiment logging.

CLAUDE.md flags "no logging backend — only print() statements" as a known
gap. This module fixes that with a single `ExperimentLogger` that:
- Always logs to console + a plain-text file (zero dependencies).
- Optionally logs scalars to TensorBoard if `tensorboard` is installed.
- Optionally logs to Weights & Biases if `wandb` is installed AND configured.
- Is a no-op-safe: if optional backends aren't installed / configured, it
  silently falls back to console+file only, so this never becomes a hard
  dependency during e.g. an edge-device build.
- Is rank-aware: in DDP training, only rank 0 writes files / remote logs.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Console + file + (optional) TensorBoard/W&B logger.

    Args:
        log_dir: Directory to write `train.log` and TensorBoard event files.
        exp_name: Experiment name, used as the W&B run name if enabled.
        use_tensorboard: Try to enable TensorBoard SummaryWriter.
        use_wandb: Try to enable Weights & Biases logging.
        wandb_project: W&B project name (required if use_wandb=True).
        config: Config dict to log as run metadata (W&B) / dump to file.
        rank: Process rank in distributed training; only rank 0 writes.
    """

    def __init__(
        self,
        log_dir: str,
        exp_name: str = 'experiment',
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict] = None,
        rank: int = 0,
    ):
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.rank = rank
        self.is_main_process = (rank == 0)
        self._tb_writer = None
        self._wandb_run = None
        self._log_file = None

        if not self.is_main_process:
            return  # Non-main ranks stay fully silent to avoid interleaved logs.

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._log_file = open(os.path.join(log_dir, 'train.log'), 'a')

        if config is not None:
            with open(os.path.join(log_dir, 'resolved_config.json'), 'w') as f:
                json.dump(config, f, indent=2, default=str)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
            except ImportError:
                self._write_console("[logger] tensorboard not installed; skipping TB logging.")

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project or 'photon2perception',
                    name=exp_name,
                    config=config or {},
                )
            except ImportError:
                self._write_console("[logger] wandb not installed; skipping W&B logging.")
            except Exception as e:  # noqa: BLE001 - never let logging crash training
                self._write_console(f"[logger] wandb init failed ({e}); skipping W&B logging.")

    def _write_console(self, msg: str) -> None:
        print(msg, flush=True)
        if self._log_file is not None:
            self._log_file.write(msg + '\n')
            self._log_file.flush()

    def log(self, msg: str) -> None:
        """Log a free-form message with a timestamp prefix."""
        if not self.is_main_process:
            return
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self._write_console(f"[{timestamp}] {msg}")

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = '') -> None:
        """Log a dict of scalar metrics (loss components, LR, metrics, etc.).

        Args:
            scalars: Mapping from metric name to value.
            step: Global step / epoch counter used as the x-axis.
            prefix: Optional namespace prefix, e.g. 'train/' or 'val/'.
        """
        if not self.is_main_process:
            return
        formatted = ' '.join(f"{prefix}{k}={v:.4f}" for k, v in scalars.items())
        self.log(f"step={step} {formatted}")

        if self._tb_writer is not None:
            for k, v in scalars.items():
                self._tb_writer.add_scalar(f"{prefix}{k}", v, step)

        if self._wandb_run is not None:
            import wandb
            wandb.log({f"{prefix}{k}": v for k, v in scalars.items()}, step=step)

    def log_image(self, tag: str, image, step: int) -> None:
        """Log an image tensor/array (e.g. routing heatmap) if a backend supports it."""
        if not self.is_main_process:
            return
        if self._tb_writer is not None:
            self._tb_writer.add_image(tag, image, step, dataformats='HWC')
        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)

    def close(self) -> None:
        if not self.is_main_process:
            return
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
        if self._log_file is not None:
            self._log_file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AverageMeter:
    """Tracks a running average of a scalar (loss, metric) over an epoch."""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class MetricTracker:
    """Convenience container for multiple named AverageMeters."""

    def __init__(self):
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for name, value in metrics.items():
            if name not in self._meters:
                self._meters[name] = AverageMeter(name)
            self._meters[name].update(value, n)

    def averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self._meters.items()}

    def reset(self) -> None:
        for meter in self._meters.values():
            meter.reset()
