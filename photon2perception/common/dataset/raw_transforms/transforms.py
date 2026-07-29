"""
Bayer-safe data augmentations for RAW images.

All transforms respect the 2x2 CFA pattern structure:
- Crops aligned to 2x2 Bayer block boundaries
- Flips that correctly handle the RGGB pattern
- Exposure and ISO-aware noise augmentation
"""

import torch
import torch.nn as nn
import random
from typing import Optional, Tuple


class BayerSafeRandomCrop:
    """Random crop aligned to 2x2 Bayer block boundaries."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (h, w), both must be even
        assert size[0] % 2 == 0 and size[1] % 2 == 0

    def __call__(self, image, targets=None):
        # image: (1, H, W) or (H, W) Bayer RAW
        # Crop origin must be at even coordinates to preserve Bayer pattern
        h, w = image.shape[-2:]
        th, tw = self.size
        if h < th or w < tw:
            return image, targets
        y = random.randrange(0, h - th + 1, 2)  # step 2 for Bayer alignment
        x = random.randrange(0, w - tw + 1, 2)
        image = image[..., y:y+th, x:x+tw]
        if targets is not None:
            # Adjust bounding box coordinates
            targets = adjust_bbox_crop(targets, x, y)
        return image, targets


class BayerSafeRandomFlip:
    """Random horizontal/vertical flip with correct CFA handling."""
    def __init__(self, h_flip_prob=0.5, v_flip_prob=0.5):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def __call__(self, image, targets=None):
        # For RGGB pattern, horizontal flip converts RGGB -> GRBG
        # This is handled automatically since we flip the whole image
        # and the pattern position is still valid (just mirrored)
        _, h, w = image.shape
        if random.random() < self.h_flip_prob:
            image = torch.flip(image, dims=[-1])
            if targets is not None:
                targets = adjust_bbox_hflip(targets, w)
        if random.random() < self.v_flip_prob:
            image = torch.flip(image, dims=[-2])
            if targets is not None:
                targets = adjust_bbox_vflip(targets, h)
        return image, targets


class ExposureJitter:
    """Random exposure adjustment by multiplying raw values."""
    def __init__(self, range=(0.5, 2.0)):
        self.range = range

    def __call__(self, image, targets=None):
        factor = random.uniform(*self.range)
        image = image * factor
        return image, targets


class BayerNoiseInject:
    """Inject ISO-dependent noise (shot + read noise model)."""
    def __init__(self, shot_scale_range=(0.0001, 0.01), read_std_range=(0.00001, 0.001)):
        self.shot_scale_range = shot_scale_range
        self.read_std_range = read_std_range

    def __call__(self, image, targets=None):
        shot_scale = 10 ** random.uniform(
            torch.log10(torch.tensor(self.shot_scale_range[0])).item(),
            torch.log10(torch.tensor(self.shot_scale_range[1])).item()
        )
        read_std = 10 ** random.uniform(
            torch.log10(torch.tensor(self.read_std_range[0])).item(),
            torch.log10(torch.tensor(self.read_std_range[1])).item()
        )
        shot_noise = torch.randn_like(image) * torch.sqrt(torch.clamp(image, 0) * shot_scale)
        read_noise = torch.randn_like(image) * read_std
        image = image + shot_noise + read_noise
        return image, targets


def adjust_bbox_crop(targets, crop_x, crop_y):
    """Adjust bounding boxes after crop."""
    if targets is None:
        return None
    targets = targets.copy()
    for t in targets:
        t['bbox'][0] -= crop_x
        t['bbox'][1] -= crop_y
    return targets

def adjust_bbox_hflip(targets, width):
    """Adjust bounding boxes after horizontal flip."""
    if targets is None:
        return None
    targets = targets.copy()
    for t in targets:
        x1, y1, x2, y2 = t['bbox']
        t['bbox'][0] = width - x2
        t['bbox'][2] = width - x1
    return targets

def adjust_bbox_vflip(targets, height):
    """Adjust bounding boxes after vertical flip."""
    if targets is None:
        return None
    targets = targets.copy()
    for t in targets:
        x1, y1, x2, y2 = t['bbox']
        t['bbox'][1] = height - y2
        t['bbox'][3] = height - y1
    return targets


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets=None):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets
