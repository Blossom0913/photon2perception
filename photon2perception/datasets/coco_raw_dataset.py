"""
COCO/Cityscapes -> synthetic Bayer RAW dataset, without an mmdet/mmseg
dependency.

CLAUDE.md's "Known gaps" #2 states: "Dataset loading raises NotImplementedError
... either integrate COCO/Cityscapes via mmdet/mmseg dataset classes, or
implement a standalone loader." This module is the standalone-loader option:
it wraps `torchvision.datasets.CocoDetection` / a directory-based Cityscapes
reader, applies `UnprocessPipeline` to synthesize Bayer RAW from the RGB
images on the fly, and returns samples in the exact target schema the loss
functions expect:

  detection:    {'boxes': (G,4) float32 xyxy, 'labels': (G,) int64}
  segmentation: (H, W) int64 label map

This intentionally does NOT require `mmdet`/`mmsegmentation` to be
installed (those remain optional per CLAUDE.md's environment setup section);
only `pycocotools` (a lightweight, ubiquitous dependency of `torchvision`'s
own COCO utilities) is needed for detection.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .unprocessing import UnprocessPipeline, MultiConditionUnprocess
from .raw_transforms.transforms import Compose


class CocoRawDetectionDataset(Dataset):
    """COCO detection images, converted to synthetic Bayer RAW on-the-fly.

    Args:
        root_dir: Directory containing COCO images (e.g. 'train2017/').
        ann_file: Path to the COCO-format annotation JSON
            (e.g. 'annotations/instances_train2017.json').
        img_size: (H, W) to resize every image to before unprocessing (COCO
            images have varying resolutions; the model requires a fixed,
            patch_size-divisible input size).
        unprocess: An `UnprocessPipeline` (or `MultiConditionUnprocess`)
            instance used to synthesize Bayer RAW from RGB. If None, a
            default `UnprocessPipeline(pattern='rggb', add_noise=True)` is
            constructed.
        transforms: Optional Bayer-safe augmentation `Compose` pipeline
            (applied post-unprocessing, pre-normalization) from
            `raw_transforms.transforms`.
        normalize: If True, rescale Bayer values from [0, 1] to [-1, 1]
            (matches `BaseRAWDataset`'s convention so both dataset families
            feed the model identically-scaled inputs).
    """

    def __init__(
        self,
        root_dir: str,
        ann_file: str,
        img_size: Tuple[int, int] = (512, 512),
        unprocess: Optional[UnprocessPipeline] = None,
        transforms: Optional[Compose] = None,
        normalize: bool = True,
        cfa_pattern: str = 'rggb',
    ):
        try:
            from pycocotools.coco import COCO
        except ImportError as e:
            raise ImportError(
                "CocoRawDetectionDataset requires pycocotools. Install it with "
                "`pip install pycocotools` (it is a transitive dependency of "
                "torchvision's COCO utilities, so it's often already present)."
            ) from e

        self.root_dir = root_dir
        self.img_size = img_size
        self.normalize = normalize
        self.transforms = transforms
        self.unprocess = unprocess if unprocess is not None else UnprocessPipeline(
            pattern=cfa_pattern, add_noise=True,
        )

        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())
        # Filter out images with zero annotations to avoid degenerate
        # all-background training samples dominating small subsets.
        self.img_ids = [
            img_id for img_id in self.img_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        # COCO category ids are not contiguous (max ~90 for 80 classes);
        # remap to a dense [0, num_classes) range expected by the loss/head.
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_image(self, img_id: int) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(path).convert('RGB')
        orig_w, orig_h = image.size

        image = TF.resize(image, list(self.img_size))  # PIL resize wants (h, w) via list
        image = TF.to_tensor(image)  # (3, H, W) in [0, 1]
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        return image, scale_x, scale_y

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        image, scale_x, scale_y = self._load_image(img_id)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: (x, y, w, h) in original image coords
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[ann['category_id']])

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        # RGB -> synthetic Bayer RAW.
        with torch.no_grad():
            bayer = self.unprocess(image.unsqueeze(0)).squeeze(0)  # (1, H, W)

        if self.transforms is not None:
            # `transforms` expect a list-of-dicts target with 'bbox' keys
            # (see raw_transforms.transforms); adapt boxes<->that schema
            # around the call so augmentations (crop/flip) stay reusable.
            box_dicts = [{'bbox': list(b)} for b in boxes_t.tolist()]
            bayer, box_dicts = self.transforms(bayer, box_dicts)
            if box_dicts:
                boxes_t = torch.tensor([d['bbox'] for d in box_dicts], dtype=torch.float32)
            else:
                boxes_t = torch.zeros(0, 4, dtype=torch.float32)

        if self.normalize:
            bayer = bayer * 2.0 - 1.0

        return {
            'image': bayer,
            'targets': {'boxes': boxes_t, 'labels': labels_t},
            'image_id': img_id,
        }


class CityscapesRawSegmentationDataset(Dataset):
    """Cityscapes-style (leftImg8bit + gtFine) images, converted to synthetic
    Bayer RAW on the fly, for semantic segmentation.

    Expects the standard Cityscapes directory layout:
        {root_dir}/leftImg8bit/{split}/{city}/{city}_..._leftImg8bit.png
        {root_dir}/gtFine/{split}/{city}/{city}_..._gtFine_labelTrainIds.png

    Args:
        root_dir: Cityscapes dataset root (containing `leftImg8bit/`, `gtFine/`).
        split: 'train', 'val', or 'test'.
        img_size: (H, W) to resize to (segmentation labels are resized with
            nearest-neighbor interpolation to preserve class-index integrity).
        unprocess: `UnprocessPipeline` instance (default constructed if None).
        num_classes: Number of segmentation classes (19 for standard
            Cityscapes trainId scheme).
        ignore_index: Label value for "don't care" pixels (255 standard).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (512, 1024),
        unprocess: Optional[UnprocessPipeline] = None,
        normalize: bool = True,
        num_classes: int = 19,
        ignore_index: int = 255,
        cfa_pattern: str = 'rggb',
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.normalize = normalize
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.unprocess = unprocess if unprocess is not None else UnprocessPipeline(
            pattern=cfa_pattern, add_noise=True,
        )

        img_dir = os.path.join(root_dir, 'leftImg8bit', split)
        label_dir = os.path.join(root_dir, 'gtFine', split)
        self.samples: List[Tuple[str, str]] = []
        if os.path.isdir(img_dir):
            for city in sorted(os.listdir(img_dir)):
                city_img_dir = os.path.join(img_dir, city)
                city_label_dir = os.path.join(label_dir, city)
                if not os.path.isdir(city_img_dir):
                    continue
                for fname in sorted(os.listdir(city_img_dir)):
                    if not fname.endswith('_leftImg8bit.png'):
                        continue
                    label_fname = fname.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    self.samples.append((
                        os.path.join(city_img_dir, fname),
                        os.path.join(city_label_dir, label_fname),
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = TF.resize(image, list(self.img_size))
        image_t = TF.to_tensor(image)  # (3, H, W) in [0, 1]

        if os.path.isfile(label_path):
            label = Image.open(label_path)
            label = TF.resize(label, list(self.img_size), interpolation=TF.InterpolationMode.NEAREST)
            label_t = torch.from_numpy(
                __import__('numpy').array(label, dtype='int64')
            )
        else:
            label_t = torch.full(self.img_size, self.ignore_index, dtype=torch.int64)

        with torch.no_grad():
            bayer = self.unprocess(image_t.unsqueeze(0)).squeeze(0)

        if self.normalize:
            bayer = bayer * 2.0 - 1.0

        return {'image': bayer, 'targets': label_t, 'image_id': idx}


def detection_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for detection batches.

    Stacks images (all same size, since resizing happens in `__getitem__`)
    but keeps `targets` as a list of per-image dicts (standard for
    variable-length object detection annotations; matches
    `DetectionLoss.forward`'s expected `targets: List[Dict[str, Tensor]]`).
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = [item['targets'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    return {'image': images, 'targets': targets, 'image_id': image_ids}


def segmentation_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for segmentation batches (fixed-size label maps, so
    a plain stack works, unlike detection's variable-length boxes).
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    targets = torch.stack([item['targets'] for item in batch], dim=0)
    image_ids = [item['image_id'] for item in batch]
    return {'image': images, 'targets': targets, 'image_id': image_ids}
