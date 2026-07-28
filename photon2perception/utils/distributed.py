"""
Distributed training helpers.

Centralizes the DDP setup/teardown boilerplate that was previously inlined
in tools/train.py, plus utilities for seeding and rank-aware operations
(e.g. only rank 0 should print/log/checkpoint).
"""

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedInfo:
    """Bundle of distributed-training identifiers used throughout the codebase."""
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = 'nccl') -> DistributedInfo:
    """Initialize `torch.distributed` if launched via torchrun, else single-process.

    Reads RANK / WORLD_SIZE / LOCAL_RANK from the environment (standard for
    torchrun / mpirun launches). If unset, returns a single-process
    DistributedInfo without touching `torch.distributed` at all, so the same
    training script runs unmodified on a laptop or a single-GPU edge dev box.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_distributed = world_size > 1
    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            actual_backend = backend
        else:
            # CPU-only distributed debugging (e.g. CI) must use gloo.
            actual_backend = 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=actual_backend)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    return DistributedInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device,
    )


def cleanup_distributed() -> None:
    """Tear down the process group, if initialized. Safe to call unconditionally."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch RNGs for reproducibility.

    Args:
        seed: Base seed.
        deterministic: If True, force cuDNN deterministic algorithms. This
            trades throughput for exact reproducibility — useful when
            debugging a training instability, not recommended for normal runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """All-reduce a dict of scalar tensors/floats across processes.

    Used to aggregate loss components / metrics computed independently on
    each rank into a single globally-consistent value for logging.
    """
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if world_size < 2:
        return input_dict

    names = sorted(input_dict.keys())
    values = torch.stack([
        torch.as_tensor(input_dict[k], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        for k in names
    ])
    dist.all_reduce(values)
    if average:
        values /= world_size
    return {k: v.item() for k, v in zip(names, values)}


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True
