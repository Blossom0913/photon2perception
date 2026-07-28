#!/usr/bin/env python3
"""
Feature exporter: raw source data (COCO/Cityscapes RGB images, or real RAW
files) -> training-ready tensors, cached to disk.

Why this exists
----------------
`tools/train.py` currently re-runs the full "load image -> resize ->
`UnprocessPipeline` (RGB -> synthetic Bayer RAW) -> Bayer-safe augmentation ->
normalize" chain on every `__getitem__` call, every epoch. That's the right
default (it's what gives free "infinite" augmentation variety on synthetic
data), but it means:
- Re-running training from scratch always pays the same fixed CPU cost for
  unprocessing, even when experimenting with model/optimizer changes only.
- There's no durable, inspectable artifact of "exactly what tensor did the
  model see" for debugging/reproducibility, matching the `.pb.txt` feature
  spec (`photon2perception/utils/feature_spec.py`) that describes the
  *shape* of that tensor.

This script performs that conversion once, up front, and writes fixed-size
shards of pre-processed tensors (`torch.save`d dicts, or bare `.npy` arrays)
to `feature_export.output_dir`, plus a `manifest.json` index. It intentionally
reuses the exact same dataset classes `tools/train.py` uses
(`CocoRawDetectionDataset`, `CityscapesRawSegmentationDataset`,
`BaseRAWDataset`) so an exported feature is byte-for-byte what training would
have produced for that sample (module RNG-driven on-the-fly augmentation,
which is intentionally *not* baked into the export -- augmentation still runs
per-epoch in `train.py`; what's cached here is the deterministic
unprocessing step: resize + RGB->Bayer + normalize).

Usage:
    python tools/export_features.py --config configs/detection/photon2percept_det_bayer.yaml --split train
    python tools/export_features.py --config configs/detection/photon2percept_det_bayer.yaml --split val --limit 32
    python tools/export_features.py --config configs/detection/photon2percept_det_bayer.yaml --emit_spec_only

See also: scripts/local_feature_exporter.sh (a thin CLI wrapper around this
script for local/AutoDL use).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.datasets.coco_raw_dataset import (
    CityscapesRawSegmentationDataset,
    CocoRawDetectionDataset,
)
from photon2perception.datasets.base_raw_dataset import BaseRAWDataset
from photon2perception.utils.config import apply_cli_overrides, load_config
from photon2perception.utils.feature_spec import (
    load_pbtxt,
    write_feature_specs_for_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export raw source data to training-ready tensors (feature export)",
    )
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--split', type=str, default='train', choices=('train', 'val'),
                         help='Which split to export (uses data.{train,val}_* / data.root_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                         help='Override config.feature_export.output_dir')
    parser.add_argument('--format', type=str, default=None, choices=('pt', 'npy'),
                         help='Override config.feature_export.format')
    parser.add_argument('--shard_size', type=int, default=None,
                         help='Override config.feature_export.shard_size (samples per shard file)')
    parser.add_argument('--limit', type=int, default=None,
                         help='Only export the first N samples (debug/smoke-test runs)')
    parser.add_argument('--emit_spec_only', action='store_true',
                         help='Only (re)generate the .pb.txt feature-spec files from the config '
                              'and exit, without touching any dataset/tensor export.')
    parser.add_argument('--override', nargs='+', default=None,
                         help="Dotted-key config overrides, e.g. data.batch_size=2")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Dataset construction (mirrors tools/train.py::build_dataloaders, single-split)
# ----------------------------------------------------------------------------

def build_export_dataset(config: Dict, split: str):
    data_cfg = config['data']
    task = config['task']
    dataset_type = data_cfg.get('type', 'coco' if task == 'detection' else 'cityscapes')
    if dataset_type == 'synthetic':
        dataset_type = 'coco' if task == 'detection' else 'cityscapes'

    img_size = tuple(data_cfg.get('img_scale', config['model']['img_size']))

    if dataset_type == 'coco':
        img_dir_key = f'{split}_img_dir'
        ann_key = f'{split}_ann_file'
        if not data_cfg.get(img_dir_key) or not data_cfg.get(ann_key):
            raise ValueError(
                f"config.data.{img_dir_key}/{ann_key} must be set to export the '{split}' split "
                f"of a 'coco'-type dataset."
            )
        return CocoRawDetectionDataset(
            root_dir=data_cfg[img_dir_key],
            ann_file=data_cfg[ann_key],
            img_size=img_size,
            cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
            normalize=data_cfg.get('normalize', True),
        ), 'detection'
    elif dataset_type == 'cityscapes':
        return CityscapesRawSegmentationDataset(
            root_dir=data_cfg['root_dir'],
            split=split,
            img_size=img_size,
            cfa_pattern=data_cfg.get('cfa_pattern', 'rggb'),
            normalize=data_cfg.get('normalize', True),
            num_classes=data_cfg['num_classes'],
        ), 'segmentation'
    elif dataset_type == 'real':
        return BaseRAWDataset(root_dir=data_cfg['root_dir'], split=split), task
    else:
        raise ValueError(f"Unknown data.type '{dataset_type}'")


# ----------------------------------------------------------------------------
# Shape verification against the .pb.txt feature spec
# ----------------------------------------------------------------------------

def verify_shape_against_pbtxt(image: torch.Tensor, inputs_pbtxt_path: str) -> None:
    """Assert `image`'s (C, H, W) shape matches the `bayer_raw` feature
    declared in `inputs_pbtxt_path` (batch dim excluded, since `image` here
    is a single un-batched sample)."""
    collection = load_pbtxt(inputs_pbtxt_path)
    spec = collection.get('bayer_raw')
    if spec is None:
        raise ValueError(f"'{inputs_pbtxt_path}' does not define a 'bayer_raw' feature")
    expected = spec.shape[1:]  # drop dynamic batch dim
    actual = tuple(image.shape)
    if len(expected) != len(actual) or any(e != -1 and e != a for e, a in zip(expected, actual)):
        raise AssertionError(
            f"Exported tensor shape {actual} does not match feature spec 'bayer_raw' "
            f"shape {spec.shape} (from {inputs_pbtxt_path}). Did model.img_size / "
            f"in_chans change without regenerating the .pb.txt files? Regenerate with "
            f"`python -m photon2perception.utils.feature_spec --config <cfg> --output_dir <dir>`."
        )


# ----------------------------------------------------------------------------
# Shard writing
# ----------------------------------------------------------------------------

def _target_to_serializable(target: Any) -> Any:
    """Detach/clone tensors inside a target (dict of tensors for detection,
    a single label-map tensor for segmentation) so shards are plain data,
    safe to `torch.save` without holding onto autograd graphs / dataset
    internal state."""
    if isinstance(target, torch.Tensor):
        return target.detach().clone()
    if isinstance(target, dict):
        return {k: _target_to_serializable(v) for k, v in target.items()}
    return target


def export_split(
    dataset,
    task: str,
    output_dir: str,
    split: str,
    fmt: str,
    shard_size: int,
    inputs_pbtxt_path: str,
    verify: bool,
    limit: int = None,
) -> Dict:
    """Iterate `dataset`, writing fixed-size shards of exported tensors to
    `output_dir/{split}_shard_XXXXX.{pt,npy}`, plus a manifest dict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(dataset) if limit is None else min(limit, len(dataset))
    manifest: Dict = {
        'split': split,
        'task': task,
        'format': fmt,
        'shard_size': shard_size,
        'num_samples': num_samples,
        'shards': [],
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    shard_images: List[torch.Tensor] = []
    shard_targets: List[Any] = []
    shard_ids: List[Any] = []
    shard_idx = 0
    verified_once = False

    def _flush_shard():
        nonlocal shard_idx, shard_images, shard_targets, shard_ids
        if not shard_images:
            return
        stacked = torch.stack(shard_images, dim=0)  # (S, C, H, W)
        shard_name = f'{split}_shard_{shard_idx:05d}.{fmt}'
        shard_path = out_dir / shard_name

        if fmt == 'pt':
            torch.save({
                'images': stacked,
                'targets': shard_targets,
                'image_ids': shard_ids,
            }, shard_path)
        else:  # npy: image tensor only (targets go to a sidecar .json, since
               # variable-length detection boxes don't fit a plain ndarray)
            import numpy as np
            np.save(shard_path, stacked.numpy())
            with open(out_dir / f'{split}_shard_{shard_idx:05d}.targets.json', 'w') as f:
                json.dump(
                    [_jsonify_target(t) for t in shard_targets], f,
                )

        manifest['shards'].append({
            'file': shard_name,
            'num_samples': len(shard_images),
            'image_ids': shard_ids,
        })
        shard_idx += 1
        shard_images = []
        shard_targets = []
        shard_ids = []

    for idx in range(num_samples):
        sample = dataset[idx]
        image = sample['image']
        target = sample.get('targets')
        image_id = sample.get('image_id', idx)

        if verify and not verified_once:
            verify_shape_against_pbtxt(image, inputs_pbtxt_path)
            verified_once = True

        shard_images.append(image)
        shard_targets.append(_target_to_serializable(target))
        shard_ids.append(image_id)

        if len(shard_images) >= shard_size:
            _flush_shard()

        if (idx + 1) % max(1, num_samples // 20 or 1) == 0 or idx == num_samples - 1:
            print(f"[export_features] {split}: {idx + 1}/{num_samples} samples processed")

    _flush_shard()

    manifest_path = out_dir / f'{split}_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[export_features] Wrote manifest: {manifest_path} "
          f"({len(manifest['shards'])} shards, {num_samples} samples)")
    return manifest


def _jsonify_target(target: Any) -> Any:
    if isinstance(target, torch.Tensor):
        return target.tolist()
    if isinstance(target, dict):
        return {k: _jsonify_target(v) for k, v in target.items()}
    return target


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    # Always (re)generate the .pb.txt feature spec first: it's cheap, and
    # every downstream step (shape verification, consumers of the spec)
    # depends on it reflecting the *current* config.
    spec_dir = config.get('features', {}).get('spec_dir', 'feature_specs')
    basename = config.get('data', {}).get('feature_spec_name', config.get('task', 'model'))
    inputs_pbtxt_path, outputs_pbtxt_path = write_feature_specs_for_config(config, spec_dir, basename)
    print(f"[export_features] Feature spec: {inputs_pbtxt_path}")
    print(f"[export_features] Feature spec: {outputs_pbtxt_path}")

    if args.emit_spec_only:
        return

    fe_cfg = config.get('feature_export', {})
    output_dir = args.output_dir or fe_cfg.get('output_dir', './data/features')
    fmt = args.format or fe_cfg.get('format', 'pt')
    shard_size = args.shard_size or fe_cfg.get('shard_size', 256)
    verify = fe_cfg.get('verify_against_pbtxt', True)

    dataset, task = build_export_dataset(config, args.split)
    print(f"[export_features] Exporting split='{args.split}' task='{task}' "
          f"dataset_size={len(dataset)} -> {output_dir} (format={fmt}, shard_size={shard_size})")

    export_split(
        dataset=dataset,
        task=task,
        output_dir=output_dir,
        split=args.split,
        fmt=fmt,
        shard_size=shard_size,
        inputs_pbtxt_path=inputs_pbtxt_path,
        verify=verify,
        limit=args.limit,
    )
    print("[export_features] Done.")


if __name__ == '__main__':
    main()
