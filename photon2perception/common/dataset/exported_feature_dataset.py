"""
Dataset that reads pre-exported feature shards written by
`tools/export_features.py`.

Why this exists
----------------
`tools/export_features.py` converts raw source data (COCO/Cityscapes RGB
images, real RAW files) into training-ready Bayer tensors once, up front,
and writes them as fixed-size shards (`{split}_shard_XXXXX.pt` +
`{split}_manifest.json`) under `config['feature_export']['output_dir']`.
`PreExportedFeatureDataset` is the read-side counterpart: a plain
`torch.utils.data.Dataset` that memory-maps/loads those shards lazily (one
shard resident in memory at a time) and returns samples in the exact same
`{'image', 'targets', 'image_id'}` schema `CocoRawDetectionDataset` /
`CityscapesRawSegmentationDataset` use, so it's a drop-in replacement
wherever those are used (`tools/train.py::build_dataloaders`,
`detection_collate_fn` / `segmentation_collate_fn`).

This intentionally does NOT re-run any augmentation or normalization --
that already happened at export time (see `export_features.py`'s module
docstring for what's baked in vs. left for per-epoch randomness). If you
need per-epoch Bayer-safe augmentation (random crop/flip/exposure jitter),
export raw (un-augmented) tensors and keep using the live dataset classes,
or extend this class to apply `raw_transforms` after loading.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PreExportedFeatureDataset(Dataset):
    """Reads `{split}_shard_*.pt` files + `{split}_manifest.json` produced by
    `tools/export_features.py --format pt`.

    Args:
        feature_dir: Directory containing the manifest + shard files
            (`config['feature_export']['output_dir']`).
        split: 'train' or 'val' -- selects `{split}_manifest.json`.
        cache_shards: If True (default), keep every loaded shard resident in
            memory for the lifetime of the dataset (fine for small/medium
            exported datasets on a training node with enough RAM); if False,
            only the most-recently-accessed shard is kept, trading repeated
            disk reads for lower peak memory on very large exports.
    """

    def __init__(self, feature_dir: str, split: str = 'train', cache_shards: bool = True):
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.cache_shards = cache_shards

        manifest_path = self.feature_dir / f'{split}_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No pre-exported feature manifest found at {manifest_path}. "
                f"Run `tools/export_features.py --config <cfg> --split {split}` "
                f"(or scripts/local_feature_exporter.sh) first, or set "
                f"config['feature_export']['enabled']=false to use the live "
                f"on-the-fly dataset instead."
            )
        with open(manifest_path, 'r') as f:
            self.manifest: Dict = json.load(f)

        if self.manifest.get('format', 'pt') != 'pt':
            raise ValueError(
                f"PreExportedFeatureDataset only supports format='pt' manifests "
                f"(got '{self.manifest.get('format')}'). The 'npy' export format "
                f"is intended for external/non-PyTorch consumers, not for "
                f"feeding back into tools/train.py."
            )

        # Flat index: global sample idx -> (shard_idx, offset_within_shard)
        self._index: List[tuple] = []
        for shard_idx, shard_info in enumerate(self.manifest['shards']):
            for offset in range(shard_info['num_samples']):
                self._index.append((shard_idx, offset))

        self._shard_cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]
        shard_file = self.manifest['shards'][shard_idx]['file']
        shard_path = self.feature_dir / shard_file
        # weights_only=False: shards contain plain tensors/dicts/python
        # primitives (no arbitrary classes), but detection targets are a
        # list of dicts of tensors, which torch's default safe-unpickler
        # rejects; these are exporter-produced artifacts from this same
        # trusted repo, not third-party checkpoints.
        data = torch.load(shard_path, map_location='cpu', weights_only=False)
        if not self.cache_shards:
            self._shard_cache.clear()
        self._shard_cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, offset = self._index[idx]
        shard = self._load_shard(shard_idx)
        return {
            'image': shard['images'][offset],
            'targets': shard['targets'][offset],
            'image_id': shard['image_ids'][offset],
        }


def try_build_preexported_dataset(
    feature_export_cfg: Dict[str, Any], split: str
) -> Optional[PreExportedFeatureDataset]:
    """Convenience helper for `tools/train.py`: build a
    `PreExportedFeatureDataset` for `split` if `feature_export.enabled` is
    true AND a manifest actually exists on disk; otherwise return None so
    the caller falls back to the live on-the-fly dataset.
    """
    if not feature_export_cfg or not feature_export_cfg.get('enabled', False):
        return None
    output_dir = feature_export_cfg.get('output_dir')
    if not output_dir:
        return None
    manifest_path = os.path.join(output_dir, f'{split}_manifest.json')
    if not os.path.isfile(manifest_path):
        return None
    return PreExportedFeatureDataset(output_dir, split=split)
