#!/usr/bin/env python3
"""
Visualization tools for qualitative analysis (Section 4.6).

Generates:
- Token routing heatmap overlays
- Attention distribution maps
- Degradation-scene selective focusing visualizations
- RAW-native vs RGB prediction comparisons
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple, List


def visualize_token_routing(
    image: torch.Tensor,
    routing_scores: torch.Tensor,
    grid_h: int,
    grid_w: int,
    save_path: str,
    title: str = 'Token Routing Heatmap',
):
    """
    Overlay routing scores on the input image as a heatmap.

    Shows which spatial regions the sparse router activates.

    Args:
        image: (1, H, W) or (3, H, W) input image in [0, 1]
        routing_scores: (N,) routing scores for each token
        grid_h, grid_w: Token grid dimensions
        save_path: Output image path
        title: Plot title
    """
    # Reshape scores to 2D grid
    scores_2d = routing_scores.reshape(grid_h, grid_w)

    # Prepare image for display
    if image.dim() == 3 and image.shape[0] == 1:
        img_display = image.squeeze(0).cpu().numpy()
        cmap_img = 'gray'
    elif image.dim() == 3 and image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).cpu().numpy()
        cmap_img = None
    else:
        img_display = image.squeeze().cpu().numpy()
        cmap_img = 'gray'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axes[0].imshow(img_display, cmap=cmap_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Routing heatmap
    im = axes[1].imshow(scores_2d.cpu().numpy(), cmap='hot', interpolation='nearest')
    axes[1].set_title('Routing Scores')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    from scipy.ndimage import zoom
    zoom_h = image.shape[-2] / grid_h
    zoom_w = image.shape[-1] / grid_w
    scores_upsampled = zoom(scores_2d.cpu().numpy(), (zoom_h, zoom_w), order=1)

    axes[2].imshow(img_display, cmap=cmap_img)
    heatmap = axes[2].imshow(
        scores_upsampled,
        cmap='jet',
        alpha=0.5,
        interpolation='bilinear',
    )
    axes[2].set_title('Routing Overlay')
    axes[2].axis('off')
    plt.colorbar(heatmap, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Routing visualization saved to {save_path}")


def visualize_attention_maps(
    attention_weights: torch.Tensor,
    grid_h: int,
    grid_w: int,
    save_path: str,
    layer_names: Optional[List[str]] = None,
    num_layers_to_show: int = 4,
):
    """
    Visualize attention distributions across transformer layers.

    Args:
        attention_weights: (num_layers, num_heads, N, N) attention weights
        grid_h, grid_w: Token grid
        save_path: Output path
        layer_names: Names for each layer
        num_layers_to_show: Number of layers to visualize
    """
    num_layers = attention_weights.shape[0]
    layers_to_show = min(num_layers, num_layers_to_show)
    step = max(1, num_layers // layers_to_show)
    selected_layers = list(range(0, num_layers, step))[:layers_to_show]

    fig, axes = plt.subplots(2, layers_to_show, figsize=(4 * layers_to_show, 8))

    for idx, layer_idx in enumerate(selected_layers):
        # Average over heads
        attn = attention_weights[layer_idx].mean(dim=0).cpu().numpy()  # (N, N)

        # Show attention from CLS token
        if axes.ndim == 1:
            ax_cls = axes[0] if layers_to_show == 1 else axes[0][idx]
            ax_spatial = axes[1] if layers_to_show == 1 else axes[1][idx]
        else:
            ax_cls = axes[0][idx]
            ax_spatial = axes[1][idx]

        # CLS token attention to spatial tokens
        cls_attn = attn[0, 1:]  # First row (CLS), exclude self
        cls_attn_2d = cls_attn.reshape(grid_h, grid_w)
        ax_cls.imshow(cls_attn_2d, cmap='viridis')
        ax_cls.set_title(f'Layer {layer_idx}: CLS Attention')
        ax_cls.axis('off')

        # Spatial attention pattern (average spatial attention)
        spatial_attn = attn[1:, 1:].mean(axis=1).reshape(grid_h, grid_w)
        ax_spatial.imshow(spatial_attn, cmap='viridis')
        ax_spatial.set_title(f'Layer {layer_idx}: Spatial Attention')
        ax_spatial.axis('off')

    plt.suptitle('Attention Map Analysis Across Layers')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention visualization saved to {save_path}")


def visualize_raw_vs_rgb_comparison(
    bayer_image: torch.Tensor,
    raw_prediction: torch.Tensor,
    rgb_image: torch.Tensor,
    rgb_prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    save_path: str,
):
    """
    Side-by-side comparison of RAW-native vs RGB predictions.

    Args:
        bayer_image: (1, H, W) Bayer RAW image
        raw_prediction: (H, W) or (num_classes, H, W) RAW model prediction
        rgb_image: (3, H, W) RGB image
        rgb_prediction: (H, W) or (num_classes, H, W) RGB model prediction
        ground_truth: (H, W) ground truth labels
        save_path: Output path
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Bayer input
    axes[0, 0].imshow(bayer_image.squeeze().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Bayer RAW Input')
    axes[0, 0].axis('off')

    # RGB input
    axes[1, 0].imshow(rgb_image.permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title('RGB Input')
    axes[1, 0].axis('off')

    # RAW prediction
    if raw_prediction.dim() == 3:
        raw_pred_display = raw_prediction.argmax(dim=0).cpu().numpy()
    else:
        raw_pred_display = raw_prediction.cpu().numpy()
    axes[0, 1].imshow(raw_pred_display, cmap='tab20')
    axes[0, 1].set_title('RAW Model Prediction')
    axes[0, 1].axis('off')

    # RGB prediction
    if rgb_prediction.dim() == 3:
        rgb_pred_display = rgb_prediction.argmax(dim=0).cpu().numpy()
    else:
        rgb_pred_display = rgb_prediction.cpu().numpy()
    axes[1, 1].imshow(rgb_pred_display, cmap='tab20')
    axes[1, 1].set_title('RGB Model Prediction')
    axes[1, 1].axis('off')

    # Error maps
    gt = ground_truth.cpu().numpy()
    raw_errors = (raw_pred_display != gt)
    rgb_errors = (rgb_pred_display != gt)

    axes[0, 2].imshow(raw_errors, cmap='Reds', vmin=0, vmax=1)
    axes[0, 2].set_title(f'RAW Errors: {raw_errors.mean():.2%}')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(rgb_errors, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title(f'RGB Errors: {rgb_errors.mean():.2%}')
    axes[1, 2].axis('off')

    plt.suptitle('RAW vs. RGB Perception Comparison')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved to {save_path}")


def visualize_routing_by_condition(
    bayer_image: torch.Tensor,
    routing_scores: torch.Tensor,
    condition: str,
    grid_h: int,
    grid_w: int,
    save_path: str,
):
    """
    Visualize how routing adapts to different degradation conditions.

    For low-light, noise, blur, occlusion scenes:
    - Overlay routing decisions on degraded inputs
    - Show that routing adapts to degradation (more tokens kept in
      challenging regions)

    Args:
        bayer_image: (1, H, W) input
        routing_scores: (N,) token routing scores
        condition: Degradation type
        grid_h, grid_w: Token grid
        save_path: Output path
    """
    from scipy.ndimage import zoom

    scores_2d = routing_scores.reshape(grid_h, grid_w).cpu().numpy()
    img = bayer_image.squeeze().cpu().numpy()

    zoom_h = img.shape[-2] / grid_h
    zoom_w = img.shape[-1] / grid_w
    scores_upsampled = zoom(scores_2d, (zoom_h, zoom_w), order=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Input ({condition})')
    axes[0].axis('off')

    axes[1].imshow(scores_2d, cmap='hot', interpolation='nearest')
    axes[1].set_title('Routing Pattern')
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(scores_upsampled, cmap='jet', alpha=0.5, interpolation='bilinear')
    axes[2].set_title(f'Routing Adaptation ({condition})')
    axes[2].axis('off')

    # Add statistics
    kept_fraction = (routing_scores > 0.5).float().mean().item()
    fig.suptitle(
        f'Condition: {condition} | '
        f'Tokens Kept: {kept_fraction:.1%} | '
        f'Mean Score: {routing_scores.mean().item():.3f}'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Quick self-test
if __name__ == '__main__':
    print("Testing visualization functions...")

    # Create dummy data
    dummy_img = torch.rand(1, 224, 224)
    dummy_routing = torch.rand(196)  # 14x14 grid
    dummy_save = Path('./outputs/test_viz.png')
    dummy_save.parent.mkdir(parents=True, exist_ok=True)

    visualize_token_routing(
        dummy_img, dummy_routing, grid_h=14, grid_w=14,
        save_path=str(dummy_save),
    )

    print("Visualization tests complete.")
