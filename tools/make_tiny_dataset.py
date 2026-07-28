#!/usr/bin/env python3
"""
Generate a tiny, synthetic COCO-format or Cityscapes-format dataset for
smoke-testing the training pipeline end-to-end on CPU, without needing to
download any real dataset.

Why this exists
----------------
CLAUDE.md's "Working with this repo" section recommends exactly this
workflow for a genuine end-to-end smoke test: "generate a tiny COCO-format
dataset (a handful of small images + a matching annotations.json) and a
tiny model config ... in a scratch directory". This script is that
generator, used by `scripts/train_debug.sh` (and reusable standalone for
ad hoc debugging).

Usage:
    python tools/make_tiny_dataset.py --task detection --output_dir /tmp/tiny_coco --num_images 8
    python tools/make_tiny_dataset.py --task segmentation --output_dir /tmp/tiny_cityscapes --num_images 8
"""

import argparse
import json
import os
import random
from pathlib import Path


def make_tiny_coco(output_dir: str, num_images: int, img_h: int, img_w: int, num_classes: int, seed: int):
    import numpy as np
    from PIL import Image

    random.seed(seed)
    np.random.seed(seed)

    img_dir = Path(output_dir) / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    for i in range(num_images):
        fname = f'img_{i:04d}.jpg'
        arr = (np.random.rand(img_h, img_w, 3) * 255).astype('uint8')
        Image.fromarray(arr).save(img_dir / fname)
        images.append({'id': i, 'file_name': fname, 'width': img_w, 'height': img_h})

        # 1-3 random boxes per image, always inside bounds and non-degenerate.
        for _ in range(random.randint(1, 3)):
            bw = random.randint(max(4, img_w // 8), img_w // 3)
            bh = random.randint(max(4, img_h // 8), img_h // 3)
            x = random.randint(0, max(0, img_w - bw - 1))
            y = random.randint(0, max(0, img_h - bh - 1))
            annotations.append({
                'id': ann_id,
                'image_id': i,
                'category_id': random.randint(1, num_classes),
                'bbox': [x, y, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
            })
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': c, 'name': f'class_{c}'} for c in range(1, num_classes + 1)],
    }
    ann_path = Path(output_dir) / 'annotations.json'
    with open(ann_path, 'w') as f:
        json.dump(coco, f)

    print(f"[make_tiny_dataset] Wrote {num_images} images to {img_dir}")
    print(f"[make_tiny_dataset] Wrote COCO annotations to {ann_path}")
    return str(img_dir), str(ann_path)


def make_tiny_cityscapes(output_dir: str, num_images: int, img_h: int, img_w: int, num_classes: int, seed: int):
    import numpy as np
    from PIL import Image

    random.seed(seed)
    np.random.seed(seed)

    for split, n in (('train', num_images), ('val', max(2, num_images // 4))):
        img_dir = Path(output_dir) / 'leftImg8bit' / split / 'tinycity'
        label_dir = Path(output_dir) / 'gtFine' / split / 'tinycity'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            stem = f'tinycity_{i:04d}_000000'
            arr = (np.random.rand(img_h, img_w, 3) * 255).astype('uint8')
            Image.fromarray(arr).save(img_dir / f'{stem}_leftImg8bit.png')

            label = np.random.randint(0, num_classes, size=(img_h, img_w), dtype='uint8')
            Image.fromarray(label).save(label_dir / f'{stem}_gtFine_labelTrainIds.png')

        print(f"[make_tiny_dataset] Wrote {n} '{split}' images+labels under {img_dir.parent}")

    return str(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('detection', 'segmentation'), required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=8)
    parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--img_w', type=int, default=96)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.task == 'detection':
        make_tiny_coco(args.output_dir, args.num_images, args.img_h, args.img_w, args.num_classes, args.seed)
    else:
        make_tiny_cityscapes(args.output_dir, args.num_images, args.img_h, args.img_w, args.num_classes, args.seed)


if __name__ == '__main__':
    main()
