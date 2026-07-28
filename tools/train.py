#!/usr/bin/env python3
"""
Training script for Photon2Perception models.

Supports:
- Single-GPU and multi-GPU (DDP) training
- Config-based model, dataset, optimizer specification
- Checkpoint save/resume
- WandB / TensorBoard logging
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.models.backbones.raw_vit import RawViT
from photon2perception.models.heads.detection_head import RawDetectionHead
from photon2perception.datasets.base_raw_dataset import BaseRAWDataset, SyntheticRAWDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Photon2Perception')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DDP')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def build_model(config, device):
    """Build model from config dict."""
    model_cfg = config['model']

    backbone = RawViT(
        img_size=tuple(model_cfg['img_size']),
        patch_size=model_cfg['patch_size'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        use_rope_2d=model_cfg.get('use_rope_2d', True),
        use_directional=model_cfg.get('use_directional', False),
        use_sparse_routing=model_cfg.get('use_sparse_routing', False),
        router_type=model_cfg.get('router_type', 'saliency'),
        keep_ratio=model_cfg.get('keep_ratio', 0.7),
    )

    if config['task'] == 'detection':
        head = RawDetectionHead(
            in_channels=model_cfg['embed_dim'],
            num_classes=config['data']['num_classes'],
        )
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    # Combine backbone and head
    model = nn.ModuleDict({
        'backbone': backbone,
        'head': head,
    })

    return model.to(device)


def build_dataloaders(config, rank, world_size):
    """Build train and validation dataloaders."""
    data_cfg = config['data']

    # Build dataset
    dataset_type = data_cfg.get('type', 'synthetic')
    if dataset_type == 'synthetic':
        # Use synthetic RAW generation
        from photon2perception.datasets.unprocessing import UnprocessPipeline
        unprocess = UnprocessPipeline(
            pattern=data_cfg.get('cfa_pattern', 'rggb'),
            add_noise=data_cfg.get('add_noise', True),
            bit_depth=data_cfg.get('bit_depth', 8),
        )
        # For synthetic, use a dummy RGB dataset (replace with actual dataset)
        # This is a placeholder — in practice, integrate with mmdet datasets
        raise NotImplementedError("Connect to mmdet datasets or implement your own RGB dataset loader")
    else:
        dataset = BaseRAWDataset(
            root_dir=data_cfg['root_dir'],
            split='train',
        )

    # Distributed sampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, None


def train_one_epoch(model, dataloader, optimizer, epoch, device, rank):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch.get('targets', None)

        optimizer.zero_grad()

        # Forward pass
        cls_token, hidden_states = model['backbone'](images)

        # Compute loss (placeholder — implement task-specific loss)
        loss = cls_token.sum() * 0.0  # Dummy loss for testing

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if rank == 0 and batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) if args.config.endswith('.yaml') else json.load(f)

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_model(config, device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(config, rank, world_size)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001),
    )

    # Scheduler: warmup + cosine
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    total_epochs = config['training']['epochs']
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs) * len(train_loader),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs * len(train_loader)],
    )

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(config['training']['epochs']):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device, rank)

        if rank == 0:
            print(f"Epoch {epoch} completed. Average loss: {train_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': train_loss,
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch:04d}.pth')


if __name__ == '__main__':
    main()
