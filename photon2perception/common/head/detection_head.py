"""
Detection head for RAW perception.

Lightweight detection head following a RetinaNet-style design,
adapted for RAW token features from the RawViT backbone.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RawDetectionHead(nn.Module):
    """
    Detection head for RAW token features.

    Takes feature pyramid from the backbone and predicts
    class scores and bounding box offsets at multiple scales.

    Args:
        in_channels: Input feature dimension
        num_classes: Number of object classes (excluding background)
        num_anchors: Number of anchors per spatial location
        feat_channels: Internal feature dimension for head convolutions
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_anchors: int = 9,
        feat_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.feat_channels = feat_channels

        # Classification subnet
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_anchors * num_classes, 3, padding=1),
        )

        # Regression subnet
        self.reg_subnet = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_anchors * 4, 3, padding=1),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize classification bias for prior probability
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        if hasattr(self.cls_subnet[-1], 'bias') and self.cls_subnet[-1].bias is not None:
            nn.init.constant_(self.cls_subnet[-1].bias, bias_value)

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: List of feature maps from FPN, each (B, C, H, W)
        Returns:
            cls_scores: List of (B, num_anchors*num_classes, H, W)
            bbox_preds: List of (B, num_anchors*4, H, W)
        """
        cls_scores = []
        bbox_preds = []

        for feat in features:
            cls_scores.append(self.cls_subnet(feat))
            bbox_preds.append(self.reg_subnet(feat))

        return cls_scores, bbox_preds
