"""
Checkpoint save/load utilities.

Centralizes the checkpoint format used across train.py / eval.py / export
scripts so that:
- A checkpoint saved during training can be loaded for eval, export, or
  resume without format drift.
- DDP-wrapped models (`module.` prefix) and EMA weights are handled
  transparently.
- Loading is robust to architecture drift during research iteration
  (e.g. loading a checkpoint into a model with a few renamed/added params)
  via `strict=False` + a clear report of mismatched keys.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the 'module.' prefix added by DistributedDataParallel."""
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped in DDP/DataParallel."""
    return model.module if hasattr(model, 'module') else model


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save a full training checkpoint.

    Args:
        path: Destination file path (parent dirs created automatically).
        model: Model to save (DDP wrapper is automatically unwrapped).
        optimizer: Optimizer state, if resuming training.
        scheduler: LR scheduler state, if resuming training.
        scaler: AMP GradScaler state, if using mixed precision.
        epoch: Current epoch (for resume / logging).
        global_step: Current global training step.
        best_metric: Best validation metric seen so far (for model selection).
        config: The experiment config dict, embedded for reproducibility.
        extra: Any additional fields to store (merged into checkpoint dict).
    """
    Path(os.path.dirname(os.path.abspath(path))).mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    ckpt: Dict[str, Any] = {
        'model_state_dict': raw_model.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_metric': best_metric,
        'config': config,
    }
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if extra:
        ckpt.update(extra)

    # Write atomically: save to a temp file then rename, to avoid corrupt
    # checkpoints if training is killed mid-write (important for long,
    # unattended edge/cloud training runs).
    tmp_path = f"{path}.tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint`.

    Args:
        path: Checkpoint file path.
        model: If provided, model weights are loaded into it in-place.
        optimizer: If provided, optimizer state is restored (for resume).
        scheduler: If provided, scheduler state is restored (for resume).
        scaler: If provided, AMP scaler state is restored (for resume).
        map_location: torch.load map_location (use 'cpu' to avoid GPU OOM
            when just inspecting a checkpoint on a machine without the
            original GPU).
        strict: Passed to `load_state_dict`. Set False when loading into a
            model with minor architecture differences (reports mismatches
            instead of raising).
    Returns:
        The raw checkpoint dict (epoch, global_step, best_metric, config, ...).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required because our checkpoints contain non-tensor
    # objects (ConfigDict, optimizer state, etc.) that are not in PyTorch's
    # default safe-globals list.  The caller is responsible for only loading
    # checkpoints from trusted sources (own training runs, published releases).
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and 'model_state_dict' in ckpt:
        state_dict = strip_ddp_prefix(ckpt['model_state_dict'])
        raw_model = unwrap_model(model)
        missing, unexpected = raw_model.load_state_dict(state_dict, strict=strict)
        if not strict and (missing or unexpected):
            print(
                f"[checkpoint] Loaded '{path}' with strict=False. "
                f"missing_keys={list(missing)} unexpected_keys={list(unexpected)}"
            )

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    return ckpt


def load_weights_only(
    path: str,
    model: nn.Module,
    map_location: str = 'cpu',
    strict: bool = True,
    key: str = 'model_state_dict',
) -> nn.Module:
    """Load only model weights from a checkpoint (or a bare state_dict file).

    Convenience helper for eval/export scripts that never need
    optimizer/scheduler state. Handles both:
    - full training checkpoints (dict containing `key`)
    - bare `state_dict()` dumps (no wrapping dict)

    Args:
        path: Checkpoint or state_dict file path.
        model: Model to load weights into (mutated in place).
        map_location: torch.load map_location.
        strict: Passed to `load_state_dict`.
        key: The dict key holding the state_dict in a full checkpoint.
    Returns:
        The same `model`, for chaining, e.g. `model = load_weights_only(...)`.
    """
    # weights_only=False: see load_checkpoint() for rationale.
    raw = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = raw[key] if isinstance(raw, dict) and key in raw else raw
    state_dict = strip_ddp_prefix(state_dict)
    raw_model = unwrap_model(model)
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=strict)
    if not strict and (missing or unexpected):
        print(
            f"[checkpoint] Loaded weights from '{path}' with strict=False. "
            f"missing_keys={list(missing)} unexpected_keys={list(unexpected)}"
        )
    return model


def find_latest_checkpoint(directory: str, pattern: str = 'checkpoint_epoch_*.pth') -> Optional[str]:
    """Find the most recently modified checkpoint matching `pattern` in `directory`.

    Used by train.py's `--auto_resume` to pick up where a preempted job left
    off, which matters a lot for spot-instance / AutoDL style training.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None
    candidates = sorted(
        directory_path.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else None
