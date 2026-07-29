"""
Base dataset classes for RAW image perception tasks.

Provides dataset wrappers for:
- Real RAW datasets (PASCAL RAW, LOD, AODRaw)
- Synthetic RAW datasets (RGB→Bayer via unprocessing)
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Dict, Any
import rawpy
from PIL import Image
from .unprocessing import UnprocessPipeline, MultiConditionUnprocess


class BaseRAWDataset(Dataset):
    """
    Base class for RAW image datasets.

    Handles loading RAW files (via rawpy for .DNG/.NEF/.ARW etc. or
    via numpy for preprocessed .npy files) and provides a consistent
    interface for training.

    Args:
        root_dir: Dataset root directory
        split: 'train', 'val', or 'test'
        transforms: Optional data augmentation transforms
        input_format: 'bayer' (1ch RAW) or 'rgb' (3ch demosaiced)
        normalize: Whether to normalize pixel values
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transforms=None,
        input_format: str = 'bayer',
        normalize: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.input_format = input_format
        self.normalize = normalize

        self.data_list = self._load_annotations()
        self.classes = self._get_classes()

    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations. Override in subclass."""
        raise NotImplementedError

    def _get_classes(self) -> List[str]:
        """Get list of class names. Override in subclass."""
        raise NotImplementedError

    def _load_raw(self, path: str) -> np.ndarray:
        """Load a RAW image file."""
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.npy', '.npz'):
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data['arr_0'] if 'arr_0' in data else list(data.values())[0]
            return data.astype(np.float32)
        elif ext in ('.dng', '.nef', '.arw', '.cr2', '.cr3', '.rw2'):
            with rawpy.imread(path) as raw:
                # Return raw Bayer data as float32
                bayer = raw.raw_image_visible.astype(np.float32)
                # Normalize by white level
                bayer = bayer / raw.white_level
                return bayer
        else:
            # Assume standard image format
            img = Image.open(path)
            return np.array(img, dtype=np.float32) / 255.0

    def _prepare_input(self, raw_data: np.ndarray) -> torch.Tensor:
        """Convert raw numpy array to model input tensor."""
        if raw_data.ndim == 2:
            # Single-channel Bayer
            tensor = torch.from_numpy(raw_data).unsqueeze(0)  # (1, H, W)
        elif raw_data.ndim == 3:
            tensor = torch.from_numpy(raw_data)  # (C, H, W)
        else:
            raise ValueError(f"Unexpected data shape: {raw_data.shape}")
        return tensor.float()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        item = self.data_list[idx]

        # Load image
        img_path = item['image_path']
        raw_data = self._load_raw(img_path)
        image = self._prepare_input(raw_data)

        # Load targets
        targets = item.get('annotations', None)

        # Apply transforms
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        # Normalize
        if self.normalize:
            image = image * 2.0 - 1.0  # Scale to [-1, 1]

        return {'image': image, 'targets': targets, 'image_id': idx}


class SyntheticRAWDataset(BaseRAWDataset):
    """
    Dataset that generates Bayer RAW from RGB images via unprocessing.

    Takes an existing RGB dataset (e.g., COCO, Cityscapes) and
    converts images to Bayer RAW on-the-fly using the unprocessing pipeline.
    This enables training on RGB datasets without real RAW data.

    Args:
        rgb_dataset: Base RGB dataset (torch Dataset or list of image paths + annotations)
        unprocess: UnprocessPipeline instance
        cache_raw: Whether to cache generated RAW images to disk
        cache_dir: Directory for cached RAW images
    """

    def __init__(
        self,
        rgb_dataset,
        unprocess: UnprocessPipeline,
        cache_raw: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(root_dir='', **kwargs)
        self.rgb_dataset = rgb_dataset
        self.unprocess = unprocess
        self.cache_raw = cache_raw
        self.cache_dir = cache_dir

        if cache_raw and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _load_annotations(self):
        # Annotations come from the base RGB dataset
        return list(range(len(self.rgb_dataset)))

    def _get_classes(self):
        if hasattr(self.rgb_dataset, 'classes'):
            return self.rgb_dataset.classes
        return [f'class_{i}' for i in range(80)]  # COCO default

    def __getitem__(self, idx):
        # Get RGB sample
        rgb_sample = self.rgb_dataset[idx]
        rgb_image = rgb_sample['image']  # Assume tensor (3, H, W) in [0, 1]
        if isinstance(rgb_image, np.ndarray):
            rgb_image = torch.from_numpy(rgb_image).float()
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image / 255.0
        rgb_image = rgb_image.unsqueeze(0)  # (1, 3, H, W)

        # Convert to Bayer RAW
        bayer = self.unprocess(rgb_image.to(next(self.unprocess.parameters()).device))

        # Apply transforms
        if self.transforms is not None:
            bayer, targets = self.transforms(bayer.squeeze(0), rgb_sample.get('targets'))
        else:
            targets = rgb_sample.get('targets')

        if self.normalize:
            bayer = bayer * 2.0 - 1.0

        return {'image': bayer, 'targets': targets, 'image_id': idx}
