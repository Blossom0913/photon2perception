"""
RGB to Bayer RAW Unprocessing Pipeline.

Reverses the ISP pipeline to convert sRGB images back to Bayer RAW sensor data.
This is critical for training on datasets that only provide RGB images (COCO,
Cityscapes, ADE20K, etc.).

The unprocessing pipeline (inspired by Chan et al. 2025, AODRaw CVPR 2025,
and Brooks et al. "Unprocessing Images for Learned Raw Denoising", CVPR 2019):

    sRGB → Inverse Tone Mapping → Inverse Gamma → sRGB-to-CameraRGB (inverse CCM)
         → Inverse White Balance → Bayer Mosaicking → Add Noise → RAW

Steps:
1. Inverse gamma correction (sRGB gamma → linear)
2. Inverse color correction matrix (CCM) → camera RGB
3. Inverse white balance → sensor RAW values
4. Bayer mosaicking → RGGB pattern extraction
5. Optional: add shot + read noise

Reference:
- Chan et al. (2025): "Raw Camera Data Object Detectors"
- AODRaw (Li et al., CVPR 2025): synthetic ImageNet-RAW via unprocessing
- Brooks et al. (CVPR 2019): "Unprocessing Images for Learned Raw Denoising"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# Standard sRGB gamma
SRGB_GAMMA = 2.4
SRGB_THRESHOLD = 0.0031308
SRGB_LINEAR_SLOPE = 12.92


# Generic CCM matrices for common cameras
# These transform from camera RGB to sRGB (forward).
# We apply the inverse for unprocessing.
CCM_MATRICES = {
    # Generic / Nikon D3200-like (from RAW-Adapter / PASCAL RAW)
    'generic': torch.tensor([
        [1.8595, -0.6022, -0.2573],
        [-0.2539, 1.6389, -0.3850],
        [0.0489, -0.5685, 1.5196],
    ], dtype=torch.float32),
    # Canon EOS 5D Mark IV (from LOD dataset)
    'canon_5d4': torch.tensor([
        [2.0914, -0.7081, -0.3833],
        [-0.4879, 1.7266, -0.2387],
        [-0.0937, -0.4494, 1.5431],
    ], dtype=torch.float32),
    # Identity (no color correction)
    'identity': torch.eye(3, dtype=torch.float32),
}


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """
    Inverse sRGB gamma: convert sRGB to linear RGB.

    Uses the standard sRGB transfer function:
        if x <= 0.04045: x / 12.92
        else: ((x + 0.055) / 1.055) ^ 2.4

    Args:
        srgb: (..., 3) sRGB values in [0, 1]
    Returns:
        linear: (..., 3) linear RGB values
    """
    threshold = 0.04045
    linear = torch.where(
        srgb <= threshold,
        srgb / SRGB_LINEAR_SLOPE,
        torch.pow((srgb + 0.055) / 1.055, SRGB_GAMMA)
    )
    return torch.clamp(linear, 0.0, 1.0)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """
    Apply sRGB gamma: convert linear RGB to sRGB.

    Args:
        linear: (..., 3) linear RGB values in [0, 1]
    Returns:
        srgb: (..., 3) sRGB values
    """
    threshold = SRGB_THRESHOLD
    srgb = torch.where(
        linear <= threshold,
        linear * SRGB_LINEAR_SLOPE,
        1.055 * torch.pow(linear, 1.0 / SRGB_GAMMA) - 0.055
    )
    return torch.clamp(srgb, 0.0, 1.0)


def apply_ccm(rgb: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """
    Apply Color Correction Matrix to transform color spaces.

    Forward (cam -> sRGB): sRGB = CCM @ camRGB
    Inverse (sRGB -> cam): camRGB = inv(CCM) @ sRGB

    Args:
        rgb: (B, 3, H, W) input RGB values (standard PyTorch channel-first
            image layout, matching every other tensor in this pipeline).
        ccm: (3, 3) color correction matrix
    Returns:
        out: (B, 3, H, W) transformed RGB values
    """
    ccm = ccm.to(rgb.device).to(rgb.dtype)
    # Per-pixel matrix-vector product along the channel dim (dim=1):
    # out[b, i, h, w] = sum_j ccm[i, j] * rgb[b, j, h, w].
    # Implemented as an einsum rather than reshaping to channel-last +
    # `torch.matmul`, since that reshape/permute round-trip is both slower
    # and (as originally written here) easy to get wrong dimension-wise.
    out = torch.einsum('ij,bjhw->bihw', ccm, rgb)
    return out


def white_balance(
    rgb: torch.Tensor,
    wb_gains: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Apply or invert white balance gains.

    Forward: camRGB = wb_gains * sensor_raw
    Inverse: sensor_raw = camRGB / wb_gains

    Args:
        rgb: (B, 3, H, W) camera RGB values (channel-first).
        wb_gains: (3,) white balance gains (R, G, B).
        inverse: If True, divide by gains (inverse WB)
    Returns:
        out: (B, 3, H, W) balanced RGB values
    """
    # Broadcasting aligns shapes from the right, so a bare (3,) gains
    # tensor would erroneously line up against the W dimension of a
    # (B, 3, H, W) image. Reshape to (1, 3, 1, 1) to broadcast over the
    # channel dim instead.
    wb_gains = wb_gains.to(rgb.device).to(rgb.dtype).view(1, -1, 1, 1)
    if inverse:
        return rgb / (wb_gains + 1e-8)
    return rgb * wb_gains


def bayer_mosaic(rgb: torch.Tensor, pattern: str = 'rggb') -> torch.Tensor:
    """
    Create Bayer mosaic from demosaiced RGB image.

    For each 2x2 block in the RGGB pattern:
        R  G1
        G2 B

    Extract the corresponding channel value at each pixel position.

    Args:
        rgb: (B, 3, H, W) demosaiced linear camera RGB
        pattern: Bayer CFA pattern ('rggb', 'bggr', 'grbg', 'gbrg')
    Returns:
        bayer: (B, 1, H, W) mosaicked Bayer RAW image
    """
    B, C, H, W = rgb.shape
    if C != 3:
        raise ValueError(f"Expected 3-channel RGB, got {C} channels")

    # Channel mapping for each pattern
    pattern_maps = {
        'rggb': {
            (0, 0): 0,  # R at even row, even col
            (0, 1): 1,  # G1 at even row, odd col
            (1, 0): 2,  # G2 at odd row, even col
            (1, 1): 1,  # B at odd row, odd col (G channel)
        },
        'bggr': {
            (0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 1): 0,
        },
        'grbg': {
            (0, 0): 1, (0, 1): 0, (1, 0): 2, (1, 1): 1,
        },
        'gbrg': {
            (0, 0): 1, (0, 1): 2, (1, 0): 0, (1, 1): 1,
        },
    }

    pmap = pattern_maps[pattern.lower()]

    bayer = torch.zeros(B, 1, H, W, device=rgb.device, dtype=rgb.dtype)

    for dy in range(2):
        for dx in range(2):
            channel_idx = pmap[(dy, dx)]
            # Extract every other pixel starting at (dy, dx)
            bayer[:, 0, dy::2, dx::2] = rgb[:, channel_idx, dy::2, dx::2]

    return bayer


def bayer_demosaic_simple(bayer: torch.Tensor, pattern: str = 'rggb') -> torch.Tensor:
    """
    Simple bilinear demosaicing from Bayer RAW to RGB.

    This is a fast approximation for visualization purposes.
    For training, the model should operate directly on Bayer RAW.

    Args:
        bayer: (B, 1, H, W) Bayer RAW image
        pattern: CFA pattern
    Returns:
        rgb: (B, 3, H, W) demosaiced RGB
    """
    B, C, H, W = bayer.shape
    rgb = torch.zeros(B, 3, H, W, device=bayer.device, dtype=bayer.dtype)

    # R channel: at (0,0) positions
    rgb[:, 0, 0::2, 0::2] = bayer[:, 0, 0::2, 0::2]
    # B channel: at (1,1) positions
    rgb[:, 2, 1::2, 1::2] = bayer[:, 0, 1::2, 1::2]
    # G channel: at (0,1) and (1,0) positions
    rgb[:, 1, 0::2, 1::2] = bayer[:, 0, 0::2, 1::2]
    rgb[:, 1, 1::2, 0::2] = bayer[:, 0, 1::2, 0::2]

    # Bilinear interpolation for missing values
    # R at G and B positions
    rgb[:, 0, :-1:2, 1::2] = (rgb[:, 0, 0::2, 0::2][:, :, :, :-1] + rgb[:, 0, 0::2, 0::2][:, :, :, 1:]) / 2
    rgb[:, 0, 1::2, :-1:2] = (rgb[:, 0, 0::2, 0::2][:, :, :-1, :] + rgb[:, 0, 0::2, 0::2][:, :, 1:, :]) / 2

    # B at G and R positions
    rgb[:, 2, :-1:2, 1::2] = (rgb[:, 2, 1::2, 1::2][:, :, :, :-1] + rgb[:, 2, 1::2, 1::2][:, :, :, 1:]) / 2
    rgb[:, 2, 1::2, :-1:2] = (rgb[:, 2, 1::2, 1::2][:, :, :-1, :] + rgb[:, 2, 1::2, 1::2][:, :, 1:, :]) / 2

    return rgb


class UnprocessPipeline(nn.Module):
    """
    Differentiable RGB-to-Bayer unprocessing pipeline.

    Converts sRGB images (in [0, 1]) to simulated Bayer RAW sensor data.
    All operations are differentiable for potential gradient flow.

    Args:
        ccm: (3, 3) color correction matrix (sRGB ← camera RGB direction)
        wb_gains: (3,) white balance gains (R, G, B)
        pattern: CFA pattern
        add_noise: Whether to add synthetic sensor noise
        noise_params: (shot_noise_scale, read_noise_std) if add_noise
        bit_depth: Output bit depth (8 or 16)
    """

    def __init__(
        self,
        ccm: torch.Tensor = None,
        wb_gains: Tuple[float, float, float] = (2.0, 1.0, 1.5),
        pattern: str = 'rggb',
        add_noise: bool = False,
        noise_params: Optional[Tuple[float, float]] = None,
        bit_depth: int = 8,
    ):
        super().__init__()
        self.pattern = pattern.lower()
        self.add_noise = add_noise
        self.bit_depth = bit_depth
        self.max_val = 2 ** bit_depth - 1

        if ccm is None:
            ccm = CCM_MATRICES['generic']
        self.register_buffer('ccm', ccm.clone())
        self.register_buffer('ccm_inv', torch.inverse(ccm))

        self.register_buffer('wb_gains', torch.tensor(wb_gains, dtype=torch.float32))

        if noise_params is None:
            noise_params = (0.001, 0.0001)  # Default: small shot + read noise
        self.shot_noise_scale, self.read_noise_std = noise_params

    def forward(
        self,
        srgb: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            srgb: (B, 3, H, W) sRGB image in [0, 1]
            return_intermediates: If True, return dict with all intermediate results
        Returns:
            bayer: (B, 1, H, W) simulated Bayer RAW image in [0, max_val]
        """
        B, C, H, W = srgb.shape
        intermediates = {}

        # Step 1: Inverse sRGB gamma → linear RGB
        linear_rgb = srgb_to_linear(srgb)
        intermediates['linear_rgb'] = linear_rgb

        # Step 2: Inverse CCM → camera RGB
        camera_rgb = apply_ccm(linear_rgb, self.ccm_inv)
        # Clip to valid range (some values may go slightly negative due to matrix)
        camera_rgb = torch.clamp(camera_rgb, 0.0, 1.0)
        intermediates['camera_rgb'] = camera_rgb

        # Step 3: Inverse white balance → sensor RAW values
        sensor_raw = white_balance(camera_rgb, self.wb_gains, inverse=True)
        sensor_raw = torch.clamp(sensor_raw, 0.0, 1.0)
        intermediates['sensor_raw'] = sensor_raw

        # Step 4: Bayer mosaicking
        bayer = bayer_mosaic(sensor_raw, self.pattern)
        intermediates['bayer_clean'] = bayer

        # Step 5: Add noise (optional)
        if self.add_noise:
            # Shot noise: Poisson-like (signal-dependent)
            shot_noise = torch.randn_like(bayer) * torch.sqrt(
                torch.clamp(bayer, 0.0) * self.shot_noise_scale
            )
            # Read noise: Gaussian (signal-independent)
            read_noise = torch.randn_like(bayer) * self.read_noise_std
            # Dark current noise
            dark_current = torch.randn_like(bayer) * 0.0005

            bayer_noisy = bayer + shot_noise + read_noise + dark_current
            bayer = torch.clamp(bayer_noisy, 0.0, 1.0)
            intermediates['bayer_noisy'] = bayer

        # Step 6: Quantize to bit depth
        bayer = torch.round(bayer * self.max_val) / self.max_val

        if return_intermediates:
            return bayer, intermediates

        return bayer

    def set_ccm(self, ccm: torch.Tensor):
        """Update CCM and its inverse."""
        self.register_buffer('ccm', ccm.clone())
        self.register_buffer('ccm_inv', torch.inverse(ccm))

    def set_noise_params(self, shot_scale: float, read_std: float):
        """Update noise parameters (e.g., for simulating different ISOs)."""
        self.shot_noise_scale = shot_scale
        self.read_noise_std = read_std


class MultiConditionUnprocess(UnprocessPipeline):
    """
    Extended unprocessing that can simulate multiple conditions:
    - Normal light
    - Low light (reduced gain, increased noise)
    - Over-exposure (increased gain, highlight clipping)
    - Different ISOs (variable noise levels)

    Useful for robustness experiments in Section 4.4.
    """

    CONDITION_PARAMS = {
        'normal': {
            'exposure_gain': 1.0,
            'shot_noise_scale': 0.001,
            'read_noise_std': 0.0001,
            'clip_highlights': False,
        },
        'dark': {
            'exposure_gain': 0.1,
            'shot_noise_scale': 0.01,
            'read_noise_std': 0.002,
            'clip_highlights': False,
        },
        'over_exp': {
            'exposure_gain': 5.0,
            'shot_noise_scale': 0.0005,
            'read_noise_std': 0.00005,
            'clip_highlights': True,
        },
    }

    def __init__(
        self,
        condition: str = 'normal',
        ccm: torch.Tensor = None,
        pattern: str = 'rggb',
        bit_depth: int = 8,
    ):
        params = self.CONDITION_PARAMS[condition]
        super().__init__(
            ccm=ccm,
            pattern=pattern,
            add_noise=True,
            noise_params=(params['shot_noise_scale'], params['read_noise_std']),
            bit_depth=bit_depth,
        )
        self.condition = condition
        self.exposure_gain = params['exposure_gain']
        self.clip_highlights = params['clip_highlights']

    def forward(
        self,
        srgb: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Apply condition-specific processing."""
        # Adjust exposure
        srgb_adj = srgb * self.exposure_gain

        if self.clip_highlights:
            # Simulate highlight clipping
            srgb_adj = torch.clamp(srgb_adj, 0.0, 1.0)

        return super().forward(srgb_adj, return_intermediates)
