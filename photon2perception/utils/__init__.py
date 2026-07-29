"""Utility functions: registries, checkpointing, logging, distributed helpers.

Note: config loading (`ConfigDict`/`load_config`/`save_config`/
`apply_cli_overrides`) lives in `photon2perception.common.config` -- it moved
out of `utils` because it's shared, task-agnostic infrastructure alongside
`common.dataset`/`common.loss`/`common.head`/`common.evaluation`, mirroring
the reference `planning_training_pipeline/common/config.py` layout.
"""

from .checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    load_weights_only,
    save_checkpoint,
    strip_ddp_prefix,
    unwrap_model,
)
from .distributed import (
    DistributedInfo,
    cleanup_distributed,
    is_main_process,
    reduce_dict,
    set_seed,
    setup_distributed,
)
from .logger import AverageMeter, ExperimentLogger, MetricTracker
from .registry import DATASETS, LOSSES, MODELS, TRANSFORMS, Registry

__all__ = [
    # registry
    'Registry', 'MODELS', 'DATASETS', 'TRANSFORMS', 'LOSSES',
    # checkpoint
    'save_checkpoint', 'load_checkpoint', 'load_weights_only',
    'find_latest_checkpoint', 'strip_ddp_prefix', 'unwrap_model',
    # logger
    'ExperimentLogger', 'AverageMeter', 'MetricTracker',
    # distributed
    'DistributedInfo', 'setup_distributed', 'cleanup_distributed',
    'set_seed', 'reduce_dict', 'is_main_process',
]
