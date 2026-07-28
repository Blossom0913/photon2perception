"""
Segmentation head for RAW perception.

Lightweight semantic segmentation head adapted from RawSeg-Net design.
Takes features from a transformer backbone and produces per-pixel
class predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class RawSegmentationHead(nn.Module):
    """
    Semantic segmentation head for RAW token features.

    Design following RawSeg-Net pattern:
    - Feature pyramid aggregation
    - Multi-scale fusion
    - Lightweight decoder with only a few conv layers

    Args:
        in_channels: Input feature dimension from backbone
        num_classes: Number of segmentation classes
        hidden_dim: Internal feature dimension
        img_size: Output segmentation map size
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 19,
        hidden_dim: int = 256,
        img_size: tuple = (512, 1024),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        # Channel reduction
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ASPP-like module for multi-scale context
        self.aspp = ASPPModule(hidden_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) token features from backbone
            grid_h: Token grid height
            grid_w: Token grid width
        Returns:
            seg_logits: (B, num_classes, H_out, W_out)
        """
        B, N, D = features.shape

        # Reshape tokens to 2D feature map
        x = features.transpose(1, 2).reshape(B, D, grid_h, grid_w)
        # (B, D, grid_h, grid_w)

        # Channel reduction
        x = self.channel_reduce(x)  # (B, hidden_dim, grid_h, grid_w)

        # Multi-scale context
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)  # (B, num_classes, grid_h, grid_w)

        # Upsample to target resolution
        x = F.interpolate(
            x,
            size=self.img_size,
            mode='bilinear',
            align_corners=False,
        )

        return x


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.

    Captures multi-scale context using parallel dilated convolutions,
    following the DeepLab/ASPP design pattern.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inner_channels = out_channels // 4

        # 1x1 conv branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated conv branches
        dilations = [6, 12, 18]
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 3, padding=d, dilation=d),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])

        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(inner_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]

        # 1x1
        feat1 = self.conv1(x)

        # Dilated
        feat_dilated = []
        for conv in self.dilated_convs:
            feat_dilated.append(conv(x))

        # Global pooling
        feat_global = self.global_pool(x)
        feat_global = F.interpolate(
            feat_global, size=(h, w), mode='bilinear', align_corners=False
        )

        # Concatenate all branches
        out = torch.cat([feat1] + feat_dilated + [feat_global], dim=1)

        return self.out_conv(out)
