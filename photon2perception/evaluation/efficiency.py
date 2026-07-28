"""
Efficiency metrics for RAW perception models.

Measures:
- Latency (wall-clock inference time)
- FLOPs (floating point operations)
- GPU memory footprint
- Input bandwidth (bytes from sensor to processor)
- Memory bandwidth (activation sizes)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from contextlib import contextmanager


@contextmanager
def cuda_timer(name: str = ''):
    """Context manager for CUDA event timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)  # ms
    if name:
        print(f"{name}: {elapsed:.2f} ms")


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Measure inference latency of a model.

    Args:
        model: The model to benchmark
        input_shape: Input tensor shape (batch_size, channels, height, width)
        num_warmup: Number of warmup runs (not counted)
        num_runs: Number of measurement runs
        device: Device to run on
    Returns:
        Dict with 'mean_latency_ms', 'std_latency_ms', 'fps'
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # Convert to ms

    latencies = np.array(latencies)

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'fps': float(1000.0 / np.mean(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'batch_size': input_shape[0],
    }


def measure_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Measure GPU memory usage and model statistics.

    Args:
        model: The model
        input_shape: Input tensor shape
        device: Device
    Returns:
        Dict with memory metrics in MB
    """
    model.eval()
    model.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        _ = model(dummy_input)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    reserved_memory = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'peak_gpu_memory_mb': float(peak_memory),
        'reserved_gpu_memory_mb': float(reserved_memory),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': float(total_params * 4 / (1024 ** 2)),  # float32 = 4 bytes
    }


def compute_input_bandwidth(
    input_shape: Tuple[int, ...],
    input_format: str = 'bayer',
) -> Dict[str, float]:
    """
    Compute input data bandwidth from sensor to processor.

    Args:
        input_shape: (B, C, H, W) input dimensions
        input_format: 'bayer' (1 channel) or 'rgb' (3 channels)
    Returns:
        Dict with bandwidth metrics in bytes and MB
    """
    B, C, H, W = input_shape

    if input_format == 'bayer':
        # Single channel Bayer: H * W * 1 bytes (8-bit) or * 2 (16-bit)
        bytes_8bit = H * W * 1
        bytes_16bit = H * W * 2
    else:
        bytes_8bit = H * W * 3
        bytes_16bit = H * W * 6

    return {
        'input_bytes_per_image_8bit': float(bytes_8bit),
        'input_bytes_per_image_16bit': float(bytes_16bit),
        'input_mb_per_image_8bit': float(bytes_8bit / (1024 ** 2)),
        'input_mb_per_image_16bit': float(bytes_16bit / (1024 ** 2)),
        'savings_vs_rgb_percent': float((1.0 - 1.0 / 3.0) * 100),  # Bayer saves ~67%
    }


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
) -> Dict[str, float]:
    """
    Estimate FLOPs for the model.

    Uses torchprofile or fvcore if available, otherwise returns estimate.

    Args:
        model: The model
        input_shape: (B, C, H, W)
    Returns:
        Dict with FLOPs in GFLOPs
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
    except ImportError:
        try:
            from thop import profile
            dummy_input = torch.randn(*input_shape)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            total_flops = flops
        except ImportError:
            # Fallback: rough estimate based on ViT formula
            # This is approximate — install fvcore or thop for accurate results
            total_params = sum(p.numel() for p in model.parameters())
            # Rough estimate: ~2 * params per forward pass per token
            N = input_shape[2] * input_shape[3]  # Approximate tokens
            total_flops = 2 * total_params * N

    return {
        'gflops': float(total_flops / 1e9),
        'mflops': float(total_flops / 1e6),
    }


def full_efficiency_report(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    input_format: str = 'bayer',
    device: str = 'cuda',
) -> Dict:
    """
    Generate a complete efficiency report.
    """
    report = {}
    report['latency'] = measure_latency(model, input_shape, device=device)
    report['memory'] = measure_memory(model, input_shape, device=device)
    report['bandwidth'] = compute_input_bandwidth(input_shape, input_format)
    report['flops'] = estimate_flops(model, input_shape)
    return report
