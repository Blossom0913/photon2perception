#!/usr/bin/env python3
"""
Export Photon2Perception models to TorchScript and ONNX for deployment.

Why this exists
----------------
`PerceptionModel` (photon2perception.models.model_wrapper) already composes
backbone -> neck -> head into a single `forward(images) -> outputs` module
with a stable, flat signature, specifically so it can be traced/exported
without any glue-code rewriting (see that module's docstring). This script
is the actual export driver:

1. **TorchScript** (`torch.jit.trace`): a self-contained `.pt` file that can
   be loaded with `torch.jit.load` in a C++/Python runtime without the
   original Python source, and is the input format LibTorch / some mobile
   (PyTorch Mobile) / TorchServe pipelines expect.
2. **ONNX** (`torch.onnx.export`): a static computational graph consumable
   by onnxruntime, TensorRT, and (with additional vendor toolchains) edge
   NPUs such as Huawei CANN (Ascend) or Cambricon Neuware.

Both exports are followed by a **numerical parity check** against the
original eager PyTorch model on the same random input, so a silent
tracing/export bug (e.g. an `if` branch baked in at the wrong value, a
python scalar accidentally traced as a constant) is caught immediately
rather than discovered later on-device.

Sparse-routing / dynamic-control-flow notes
--------------------------------------------
Every dynamic-shape-looking op in this codebase was deliberately designed
to be trace/export-safe (see module docstrings in
`photon2perception/models/routing/router.py`,
`photon2perception/models/position_encoding/{rope_2d,directional}.py`, and
`photon2perception/models/necks/fpn_bridge.py` for the reasoning):
  - Routers (`SaliencyRouter` / `UncertaintyRouter` / `PhysicalPriorRouter`)
    *mask* tokens (multiply by 0/1) rather than gathering a variable-length
    subset, so the output shape is always `(B, N, D)` regardless of which
    tokens are "dropped" -- `topk(k, ...)` is used with `k` computed from a
    *static* N and a *constant* `keep_ratio`, so `k` is baked in as a
    constant at trace time (safe as long as input resolution doesn't change
    across calls, which is also required for the ViT patch grid itself).
  - RoPE2D's rotation angles are a fixed-shape buffer computed once at
    construction (from `grid_h`/`grid_w`, not from the input tensor).
  - DirectionalEnhance is pure Conv2d/LayerNorm with statically-known grid
    dims passed in as Python ints, not inferred from data.
This script still *asserts* `routing_active` compatibility (see
`PerceptionModel.routing_active`) before exporting, so a
`use_sparse_routing=True` config that would silently run *dense* at
inference (because `route_at_inference=False`) fails loudly instead of
producing an export that doesn't reflect the intended efficiency profile.

Detection post-processing (`photon2perception.common.head.postprocess.
postprocess_detections`) is intentionally **not** part of the exported
graph: it does score-thresholding + NMS, both classic data-dependent-shape
operations that are a poor fit for a static graph. Exported detection
models return raw per-level `(cls_scores, bbox_preds)` tensors; run
`postprocess_detections` (or `photon2perception/inference.py`, which wraps
it) after invoking the exported model.

Examples:
    # TorchScript
    python tools/export.py --config tasks/detection/config/photon2percept_det_bayer.yaml \\
        --checkpoint outputs/photon2percept_det_bayer/checkpoint_best.pth \\
        --format torchscript --output exported/det_bayer.pt

    # ONNX (opset 17, dynamic batch dim)
    python tools/export.py --config tasks/segmentation/config/photon2percept_seg_bayer.yaml \\
        --checkpoint outputs/photon2percept_seg_bayer/checkpoint_best.pth \\
        --format onnx --output exported/seg_bayer.onnx --opset 17 --dynamic_batch

    # Export both formats and skip randomly-initialized-weights warning
    python tools/export.py --config tasks/detection/config/photon2percept_det_bayer.yaml \\
        --format both --output exported/det_bayer
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from photon2perception.models.model_wrapper import PerceptionModel, build_perception_model
from photon2perception.utils.checkpoint import load_weights_only
from photon2perception.common.config import apply_cli_overrides, load_config

EXPORT_FORMATS = ('torchscript', 'onnx', 'both')

# NPU/edge-toolchain compatibility notes (read before targeting real hardware)
# ---------------------------------------------------------------------------
# Huawei Ascend (CANN / ATC toolchain):
#   - ATC (the ONNX->OM converter) has historically lagged on newer ONNX
#     opsets and on fused ops; prefer `--opset 11` or `--opset 13` and
#     `--attn_backend math` (unrolled softmax(QK^T)V instead of the fused
#     SDPA op, which ATC does not recognize as of CANN <= 7.x).
#   - GroupNorm (used in `SimpleFeaturePyramidNeck`) is supported since
#     CANN 6.0; older CANN versions may need GroupNorm folded to
#     InstanceNorm+affine or replaced with BatchNorm at export time.
#   - ConvTranspose2d (upsampling in the neck) is supported but ATC prefers
#     even, power-of-two strides/kernels (already the case here: k=2, s=2).
# Cambricon Neuware (CNNL / MagicMind toolchain):
#   - Similar story: no native fused-SDPA support -- use `--attn_backend math`.
#   - `torch.nn.functional.interpolate(mode='bilinear')` (used by
#     `RawSegmentationHead` and ASPP's global-pool branch) is supported by
#     MagicMind but only with `align_corners=False` (already this repo's
#     default) and static output size (already the case: `img_size` is a
#     fixed tuple, not computed from the input at runtime).
# NVIDIA Orin / Jetson (TensorRT):
#   - TensorRT >= 8.6 understands ONNX's `Attention`-adjacent patterns and
#     generally handles fused SDPA (opset 14+) fine; `--attn_backend sdpa`
#     (the default) is recommended here for the best kernel selection.
#   - `torchvision.ops.nms` (used only in postprocessing, not the exported
#     graph -- see module docstring above) has a native TensorRT plugin if
#     you choose to fuse post-processing into the engine later; out of
#     scope for this script by design.
# General:
#   - All three toolchains are far more reliable with `opset<=13` and
#     `dynamic_batch` disabled (fully static shapes) for a *first* bring-up;
#     only enable `--dynamic_batch` once the static-shape export is
#     confirmed working on-device.
NPU_COMPAT_NOTES = __doc__  # (kept for programmatic access / `--print_npu_notes`)


def parse_args():
    # `--print_npu_notes` is a standalone informational flag (like `--help`)
    # that shouldn't require `--config`/`--output` to also be supplied, so we
    # check for it directly against argv before argparse's required-args
    # validation would otherwise reject `python tools/export.py --print_npu_notes`.
    if '--print_npu_notes' in sys.argv:
        print(NPU_COMPAT_NOTES)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description='Export Photon2Perception models to TorchScript / ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Checkpoint to load weights from. If omitted, exports '
                              'randomly-initialized weights (useful for smoke-testing '
                              'the export pipeline itself, e.g. in CI).')
    parser.add_argument('--format', type=str, default='both', choices=EXPORT_FORMATS)
    parser.add_argument('--output', type=str, required=True,
                         help="Output path. For --format torchscript/onnx, used as-is "
                              "(extension added if missing). For --format both, used as "
                              "a stem and '.pt'/'.onnx' are appended.")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the export trace input')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--dynamic_batch', action='store_true',
                         help='Mark the batch dimension as dynamic in the ONNX graph '
                              '(TorchScript trace is always fixed-batch; use multiple '
                              'traces or torch.jit.script for dynamic TorchScript shapes)')
    parser.add_argument('--onnx_dynamo', action='store_true',
                         help="Use torch>=2.5's newer torch.export/dynamo-based ONNX "
                              "exporter (`torch.onnx.export(..., dynamo=True)`) instead "
                              "of the legacy TorchScript-tracing-based exporter. Requires "
                              "the optional 'onnxscript' package. The legacy exporter "
                              "(default here) has broader support across onnxruntime / "
                              "TensorRT / NPU toolchain versions as of this writing, so "
                              "it is the default; opt into dynamo once your target "
                              "toolchain is confirmed to support its output.")
    parser.add_argument('--attn_backend', type=str, default=None, choices=('sdpa', 'math', 'flash'),
                         help="Override the backbone's attention backend before export "
                              "(e.g. 'math' for NPU toolchains without a fused-SDPA op; "
                              "see NPU_COMPAT_NOTES in this file's docstring). Defaults "
                              "to whatever the config/checkpoint specifies.")
    parser.add_argument('--device', type=str, default='cpu',
                         help="Device to build/trace the model on. 'cpu' is recommended "
                              "for export portability (avoids baking CUDA-only kernel "
                              "choices into the traced graph).")
    parser.add_argument('--atol', type=float, default=1e-3,
                         help='Absolute tolerance for the eager-vs-exported parity check')
    parser.add_argument('--rtol', type=float, default=1e-3,
                         help='Relative tolerance for the eager-vs-exported parity check')
    parser.add_argument('--skip_verify', action='store_true',
                         help='Skip the post-export numerical parity check')
    parser.add_argument('--override', nargs='+', default=None,
                         help="Dotted-key config overrides, e.g. model.img_size=[256,256]")
    parser.add_argument('--print_npu_notes', action='store_true',
                         help='Print the Ascend/Cambricon/Orin compatibility notes and exit')
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Model construction / preparation
# ----------------------------------------------------------------------------

def build_export_model(config, checkpoint: str = None, attn_backend: str = None,
                        device: str = 'cpu') -> PerceptionModel:
    """Build a `PerceptionModel`, optionally load weights, and put it in a
    frozen, export-ready state.

    Export-readiness here means:
    - `.eval()` (disables dropout; also gates sparse routing per
      `route_at_inference`, see `PerceptionModel.routing_active`).
    - Attention backend overridden if `--attn_backend` was passed (e.g. to
      'math' for NPU toolchains that lack a fused-SDPA op).
    - A loud `AssertionError` if the model is configured for sparse routing
      but that routing would not actually execute in eval mode (i.e.
      `use_sparse_routing=True` and `route_at_inference=False`) -- exporting
      such a model would silently ship a "dense" graph despite the config
      claiming sparse routing, defeating the efficiency point of routing.
    """
    model = build_perception_model(config)

    if checkpoint:
        load_weights_only(checkpoint, model, map_location='cpu', strict=True)
        print(f"[export] Loaded weights from {checkpoint}")
    else:
        print("[export] WARNING: no --checkpoint given; exporting randomly-initialized "
              "weights. This is fine for smoke-testing the export pipeline, but the "
              "resulting artifact will not produce meaningful predictions.")

    if attn_backend is not None:
        model.backbone.set_attn_backend(attn_backend)
        print(f"[export] Overrode attention backend to '{attn_backend}'")

    model = model.to(device)
    model.eval()

    if model.backbone.use_sparse_routing and not model.routing_active:
        raise AssertionError(
            "Model is configured with use_sparse_routing=True but "
            "route_at_inference=False, so sparse routing will NOT execute in "
            "eval() mode -- exporting now would silently produce a dense graph "
            "that doesn't reflect the intended efficiency profile. Either set "
            "model.route_at_inference=true in the config (or via --override "
            "model.route_at_inference=true) to bake routing into the exported "
            "graph, or set model.use_sparse_routing=false if a dense export is "
            "actually what you want."
        )
    if model.backbone.use_sparse_routing:
        print("[export] Sparse routing IS active in the exported graph "
              f"(router_type={config['model'].get('router_type', 'saliency')}, "
              f"keep_ratio={config['model'].get('keep_ratio', 0.7)}).")

    return model


def make_dummy_input(config, batch_size: int, device: str) -> torch.Tensor:
    img_size = tuple(config['model']['img_size'])
    in_chans = config['model'].get('in_chans', 1)
    return torch.randn(batch_size, in_chans, *img_size, device=device)


def _flatten_outputs(outputs) -> List[torch.Tensor]:
    """Flatten a possibly-nested (list/tuple of list/tuple of) Tensor output
    structure into a flat list, for uniform parity-checking and ONNX
    output naming. `PerceptionModel.forward` returns:
      - detection:    (cls_scores: List[Tensor], bbox_preds: List[Tensor])
      - segmentation: seg_logits: Tensor
    """
    flat: List[torch.Tensor] = []

    def _recurse(x):
        if isinstance(x, torch.Tensor):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                _recurse(item)
        else:
            raise TypeError(f"Unexpected output leaf type {type(x)} in model output")

    _recurse(outputs)
    return flat


def _output_names(config, outputs) -> List[str]:
    task = config['task']
    flat = _flatten_outputs(outputs)
    if task == 'detection':
        num_levels = len(flat) // 2
        names = [f'cls_scores_{i}' for i in range(num_levels)]
        names += [f'bbox_preds_{i}' for i in range(num_levels)]
        return names
    return ['seg_logits'][: len(flat)] or [f'output_{i}' for i in range(len(flat))]


# ----------------------------------------------------------------------------
# TorchScript export
# ----------------------------------------------------------------------------

def export_torchscript(model: PerceptionModel, dummy_input: torch.Tensor, output_path: str) -> str:
    """Trace `model` and save a self-contained TorchScript module.

    Uses `torch.jit.trace` (not `torch.jit.script`) because:
    - Every submodule in this codebase (RoPE2D, routers, directional
      enhancement, the FPN neck) uses only tensor ops + Python-int-driven
      static reshapes, with no data-dependent Python control flow, so
      tracing captures the *exact* semantics for a fixed input shape.
    - `torch.jit.script` requires every branch to be TorchScript-typeable
      (no duck typing, restricted Python subset) which would need
      significant, purely mechanical rewrites of otherwise-fine eager code
      for no behavioral benefit here, since there's no genuine
      shape-dependent control flow to preserve across input sizes anyway
      (a fixed `img_size` is already required by `RawViT`/`RoPE2D`/the FPN
      neck's precomputed grid, so a scripted dynamic-shape graph wouldn't
      actually generalize across resolutions in this architecture).
    """
    output_path = str(Path(output_path).with_suffix('.pt'))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input, check_trace=True, strict=False)
    traced.save(output_path)
    print(f"[export] TorchScript module saved to {output_path}")
    return output_path


def verify_torchscript(model: PerceptionModel, ts_path: str, dummy_input: torch.Tensor,
                        atol: float, rtol: float) -> None:
    """Load the saved TorchScript module and compare its output against the
    live eager model on the same input, failing loudly on mismatch."""
    loaded = torch.jit.load(ts_path, map_location='cpu')
    loaded.eval()

    with torch.no_grad():
        eager_out = _flatten_outputs(model(dummy_input))
        ts_out = _flatten_outputs(loaded(dummy_input))

    _assert_allclose(eager_out, ts_out, atol, rtol, label='TorchScript')
    print(f"[export] TorchScript parity check passed (atol={atol}, rtol={rtol}).")


# ----------------------------------------------------------------------------
# ONNX export
# ----------------------------------------------------------------------------

def export_onnx(
    model: PerceptionModel,
    dummy_input: torch.Tensor,
    output_path: str,
    config: Dict,
    opset: int = 17,
    dynamic_batch: bool = False,
    use_dynamo: bool = False,
) -> str:
    """Export `model` to a static ONNX graph via `torch.onnx.export`.

    Input/output names are fixed and descriptive (rather than ONNX's
    default `onnx::Xxx_0` autogenerated names) so downstream onnxruntime/
    TensorRT/NPU-toolchain code can bind tensors by name reliably.

    `use_dynamo=False` (the default) forces the legacy TorchScript-tracing-
    based exporter (`torch.onnx.export(..., dynamo=False)`). As of torch
    2.5+, `dynamo=True` is the library default, but it requires the
    optional `onnxscript` package and its output has historically had
    narrower support across onnxruntime-versions-in-the-wild and vendor
    NPU toolchains (Ascend/Cambricon, see `NPU_COMPAT_NOTES`) than the
    battle-tested TorchScript-based path -- so we pin to the legacy
    exporter here for maximum deployment-target compatibility, and let
    `--onnx_dynamo` opt into the newer exporter once a specific target
    toolchain is confirmed to support it.
    """
    output_path = str(Path(output_path).with_suffix('.onnx'))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sample_outputs = model(dummy_input)
    output_names = _output_names(config, sample_outputs)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'images': {0: 'batch'}}
        dynamic_axes.update({name: {0: 'batch'} for name in output_names})

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=use_dynamo,
    )
    print(f"[export] ONNX graph saved to {output_path} (opset={opset}, "
          f"dynamic_batch={dynamic_batch}, dynamo={use_dynamo})")
    return output_path


def verify_onnx(model: PerceptionModel, onnx_path: str, dummy_input: torch.Tensor,
                 atol: float, rtol: float) -> None:
    """Run the exported ONNX graph through onnxruntime and compare against
    the eager PyTorch model on the same input.

    Requires `onnx` (for structural validation) and `onnxruntime` (for
    numerical execution); both are optional dependencies of this script --
    if either is missing, the check is skipped with a warning rather than
    failing the whole export (the artifact is still usable elsewhere, e.g.
    for a TensorRT/NPU toolchain that doesn't need onnxruntime at all).
    """
    try:
        import onnx
    except ImportError:
        print("[export] WARNING: 'onnx' not installed; skipping ONNX structural "
              "validation. Install with `pip install onnx` to enable.")
        return
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[export] ONNX structural validation (onnx.checker) passed.")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] WARNING: 'onnxruntime' not installed; skipping ONNX numerical "
              "parity check. Install with `pip install onnxruntime` to enable.")
        return

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: dummy_input.cpu().numpy()})

    with torch.no_grad():
        eager_out = _flatten_outputs(model(dummy_input))

    _assert_allclose(eager_out, [torch.from_numpy(o) for o in ort_outputs], atol, rtol, label='ONNX/onnxruntime')
    print(f"[export] ONNX/onnxruntime parity check passed (atol={atol}, rtol={rtol}).")


# ----------------------------------------------------------------------------
# Shared verification helper
# ----------------------------------------------------------------------------

def _assert_allclose(
    expected: List[torch.Tensor],
    actual: List[torch.Tensor],
    atol: float,
    rtol: float,
    label: str,
) -> None:
    if len(expected) != len(actual):
        raise AssertionError(
            f"[{label}] Output count mismatch: eager produced {len(expected)} tensors, "
            f"exported model produced {len(actual)}."
        )
    for i, (e, a) in enumerate(zip(expected, actual)):
        e_np = e.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        if e_np.shape != a_np.shape:
            raise AssertionError(
                f"[{label}] Output #{i} shape mismatch: eager={e_np.shape} vs exported={a_np.shape}"
            )
        if not np.allclose(e_np, a_np, atol=atol, rtol=rtol):
            max_abs_diff = float(np.max(np.abs(e_np - a_np)))
            raise AssertionError(
                f"[{label}] Output #{i} values diverge beyond tolerance "
                f"(atol={atol}, rtol={rtol}): max_abs_diff={max_abs_diff:.6g}"
            )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()  # handles --print_npu_notes (exits early) before this returns

    config = load_config(args.config)
    apply_cli_overrides(config, args.override)

    model = build_export_model(
        config, checkpoint=args.checkpoint, attn_backend=args.attn_backend, device=args.device,
    )
    dummy_input = make_dummy_input(config, batch_size=args.batch_size, device=args.device)

    # Sanity forward pass in eager mode before exporting anything, so a
    # broken config/checkpoint fails fast with a normal Python traceback
    # instead of a confusing tracer/exporter error.
    with torch.no_grad():
        _ = model(dummy_input)
    print(f"[export] Eager forward pass OK on input shape {tuple(dummy_input.shape)} "
          f"(task={config['task']}).")

    do_torchscript = args.format in ('torchscript', 'both')
    do_onnx = args.format in ('onnx', 'both')

    if args.format == 'both':
        ts_output = str(Path(args.output).with_suffix('.pt'))
        onnx_output = str(Path(args.output).with_suffix('.onnx'))
    else:
        ts_output = args.output
        onnx_output = args.output

    if do_torchscript:
        ts_path = export_torchscript(model, dummy_input, ts_output)
        if not args.skip_verify:
            verify_torchscript(model, ts_path, dummy_input, args.atol, args.rtol)

    if do_onnx:
        onnx_path = export_onnx(
            model, dummy_input, onnx_output, config,
            opset=args.opset, dynamic_batch=args.dynamic_batch, use_dynamo=args.onnx_dynamo,
        )
        if not args.skip_verify:
            verify_onnx(model, onnx_path, dummy_input, args.atol, args.rtol)

    print("[export] Done.")


if __name__ == '__main__':
    main()
