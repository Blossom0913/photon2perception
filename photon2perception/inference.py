"""
Unified inference interface for Photon2Perception models.

Why this exists
----------------
`tools/train.py` uses the eager `PerceptionModel` directly, and
`tools/export.py` produces TorchScript/ONNX artifacts from it -- but until
now there was no single entry point that downstream code (a demo script, a
serving container, a benchmark harness) could use *without caring* whether
the underlying model is:
  - the original eager PyTorch `nn.Module` (fastest iteration, needed for
    anything requiring autograd),
  - a `torch.jit.load`-ed TorchScript module (no Python source needed, good
    for LibTorch/C++ embedding or PyTorch Mobile),
  - an ONNX graph running under onnxruntime (portable across CPU/GPU/some
    NPU execution providers), or
  - a TensorRT engine (best latency on NVIDIA GPUs / Jetson Orin).

`PerceptionInferencer` (this module) hides that behind one
`predict(raw_image) -> results` call, always applying the *same*
pre-processing (normalization) and post-processing (anchor decode + NMS for
detection, argmax for segmentation) regardless of backend -- so a
benchmark or accuracy comparison across backends is apples-to-apples, and
switching backends is a one-line change (`backend='onnxruntime'` vs
`backend='pytorch'`).

Design notes
------------
- Pre/post-processing lives here, once, shared by all backends. This
  mirrors `tools/export.py`'s design choice to keep NMS *out* of the
  exported graph (data-dependent shapes) -- this module is exactly the
  "wrapper that reattaches post-processing" alluded to in that script's
  docstring and in `photon2perception/models/heads/postprocess.py`.
- Every backend other than 'pytorch' is an *optional* dependency
  (torch is obviously required for 'pytorch'/'torchscript'; 'onnxruntime'
  needs the `onnxruntime` package; 'tensorrt' needs `tensorrt` +
  `pycuda`/`cuda-python`). Import errors are deferred to
  `PerceptionInferencer.__init__` (only the requested backend's deps are
  imported), so `import photon2perception.inference` never fails just
  because e.g. TensorRT isn't installed on a laptop.
- All backends receive/return `numpy.ndarray`/`torch.Tensor` in the *same*
  logical layout (`(B, 1, H, W)` normalized Bayer RAW in, per-task
  predictions out), so results are directly comparable.

Example:
    from photon2perception.inference import PerceptionInferencer

    inferencer = PerceptionInferencer(
        config_path='configs/detection/photon2percept_det_bayer.yaml',
        backend='onnxruntime',
        weights_path='exported/det_bayer.onnx',
    )
    detections = inferencer.predict(raw_bayer_image)  # list of dicts (boxes/scores/labels)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .models.heads.postprocess import postprocess_detections
from .models.model_wrapper import PerceptionModel, build_perception_model
from .utils.checkpoint import load_weights_only
from .utils.config import ConfigDict, load_config

BACKENDS = ('pytorch', 'torchscript', 'onnxruntime', 'tensorrt')


# ----------------------------------------------------------------------------
# Backend abstraction
# ----------------------------------------------------------------------------

class _InferenceBackend:
    """Common interface every concrete backend implements: given a
    `(B, C, H, W)` numpy array, return the raw model outputs as a list of
    numpy arrays (flattened the same way `tools/export.py._flatten_outputs`
    does: detection -> `[cls_scores..., bbox_preds...]`, segmentation ->
    `[seg_logits]`).
    """

    def run(self, images: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend-specific resources (sessions/contexts). Safe
        to call multiple times; base implementation is a no-op."""
        pass


class _PyTorchBackend(_InferenceBackend):
    """Eager `PerceptionModel` backend -- the reference implementation all
    other backends are numerically checked against (see `tools/export.py`'s
    parity checks, which use exactly this path as ground truth).
    """

    def __init__(self, model: PerceptionModel, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run(self, images: np.ndarray) -> List[np.ndarray]:
        x = torch.from_numpy(images).to(self.device)
        outputs = self.model(x)
        return [t.detach().cpu().numpy() for t in _flatten(outputs)]


class _TorchScriptBackend(_InferenceBackend):
    """Loads a `.pt` file saved by `tools/export.py --format torchscript`."""

    def __init__(self, weights_path: str, device: torch.device):
        self.module = torch.jit.load(weights_path, map_location=device)
        self.module.eval()
        self.device = device

    @torch.no_grad()
    def run(self, images: np.ndarray) -> List[np.ndarray]:
        x = torch.from_numpy(images).to(self.device)
        outputs = self.module(x)
        return [t.detach().cpu().numpy() for t in _flatten(outputs)]


class _ONNXRuntimeBackend(_InferenceBackend):
    """Loads a `.onnx` graph saved by `tools/export.py --format onnx` and
    runs it through onnxruntime.

    Provider selection: tries CUDA first (if `device` requests it and the
    package build supports it), falling back to CPU. NPU-specific execution
    providers (e.g. `CANNExecutionProvider` for Huawei Ascend) can be passed
    explicitly via `providers`.
    """

    def __init__(self, weights_path: str, device: torch.device,
                 providers: Optional[List[str]] = None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "backend='onnxruntime' requires the 'onnxruntime' package "
                "(`pip install onnxruntime` or `onnxruntime-gpu`)."
            ) from e

        if providers is None:
            available = ort.get_available_providers()
            if device.type == 'cuda' and 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(weights_path, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

    def run(self, images: np.ndarray) -> List[np.ndarray]:
        images = np.ascontiguousarray(images.astype(np.float32))
        outputs = self.session.run(self._output_names, {self._input_name: images})
        return list(outputs)


class _TensorRTBackend(_InferenceBackend):
    """Loads a serialized TensorRT engine (`.engine`/`.plan`, built
    separately via `trtexec` or the TensorRT Python builder API from an
    ONNX graph produced by `tools/export.py`).

    Building the engine itself is intentionally out of scope for this
    module (it's a one-time, hardware-specific step best done with
    `trtexec --onnx=... --saveEngine=...` or a dedicated build script);
    this backend only handles *loading and running* an already-built
    engine, mirroring how the ONNX/TorchScript backends only load
    already-exported artifacts.
    """

    def __init__(self, weights_path: str, device: torch.device):
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401 - initializes the CUDA context
            import pycuda.driver as cuda
        except ImportError as e:
            raise ImportError(
                "backend='tensorrt' requires the 'tensorrt' and 'pycuda' packages, "
                "plus a TensorRT-capable NVIDIA GPU. Build the .engine file with "
                "`trtexec --onnx=model.onnx --saveEngine=model.engine` first."
            ) from e

        self._trt = trt
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(weights_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._stream = cuda.Stream()

    def run(self, images: np.ndarray) -> List[np.ndarray]:
        trt, cuda = self._trt, self._cuda
        images = np.ascontiguousarray(images.astype(np.float32))

        bindings = []
        device_buffers = []
        host_outputs = []
        output_bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self.context.set_input_shape(name, images.shape)
                d_buf = cuda.mem_alloc(images.nbytes)
                cuda.memcpy_htod_async(d_buf, images, self._stream)
                bindings.append(int(d_buf))
                device_buffers.append(d_buf)
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_out = np.empty(shape, dtype=dtype)
                d_buf = cuda.mem_alloc(host_out.nbytes)
                bindings.append(int(d_buf))
                device_buffers.append(d_buf)
                host_outputs.append(host_out)
                output_bindings.append((name, d_buf, host_out))

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])
        self.context.execute_async_v3(stream_handle=self._stream.handle)

        for _, d_buf, host_out in output_bindings:
            cuda.memcpy_dtoh_async(host_out, d_buf, self._stream)
        self._stream.synchronize()

        return [host_out for _, _, host_out in output_bindings]

    def close(self) -> None:
        # pycuda buffers are freed by GC; explicit context teardown isn't
        # strictly required but avoids leaking the execution context across
        # repeated engine loads within a long-lived process.
        self.context = None
        self.engine = None


def _flatten(outputs) -> List[torch.Tensor]:
    """Same flattening rule as `tools/export.py._flatten_outputs`, duplicated
    here (rather than imported from a `tools/` script, which is not meant to
    be imported as a library) to keep `photon2perception/` fully
    self-contained. Detection: (cls_scores_list, bbox_preds_list) ->
    flat list; segmentation: single tensor -> single-element list.
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


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------

class PerceptionInferencer:
    """Backend-agnostic inference wrapper around a Photon2Perception model.

    Args:
        config_path: Path to the experiment YAML (same one used for
            training/export -- provides `task`, `model.img_size`,
            `data.num_classes`, etc.).
        backend: One of `BACKENDS`:
            'pytorch'     -- eager `PerceptionModel` (needs `weights_path`
                              to be a training checkpoint, or None for
                              randomly-initialized weights).
            'torchscript' -- a `.pt` file from `tools/export.py --format
                              torchscript`.
            'onnxruntime' -- a `.onnx` file from `tools/export.py --format
                              onnx`.
            'tensorrt'    -- a `.engine`/`.plan` file built from that ONNX
                              graph via `trtexec` (see `_TensorRTBackend`).
        weights_path: Path to the checkpoint/TorchScript/ONNX/TensorRT
            artifact, as appropriate for `backend`. May be None only for
            backend='pytorch' (exports randomly-initialized weights,
            matching `tools/export.py`'s smoke-test convenience).
        device: 'cpu', 'cuda', 'cuda:0', or 'mps'. Only meaningful for
            'pytorch'/'torchscript' (device placement) and 'onnxruntime'
            (execution-provider selection); 'tensorrt' always runs on the
            local CUDA GPU by construction.
        score_thresh, nms_thresh, max_detections: Detection post-processing
            parameters, forwarded to
            `photon2perception.models.heads.postprocess.postprocess_detections`.
            Ignored for segmentation models.
        onnx_providers: Optional explicit onnxruntime execution provider
            list (e.g. `['CANNExecutionProvider', 'CPUExecutionProvider']`
            for Huawei Ascend); overrides the automatic CUDA/CPU selection.
    """

    def __init__(
        self,
        config_path: str,
        backend: str = 'pytorch',
        weights_path: Optional[str] = None,
        device: str = 'cpu',
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
        onnx_providers: Optional[List[str]] = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got '{backend}'")
        if backend != 'pytorch' and weights_path is None:
            raise ValueError(f"backend='{backend}' requires weights_path to be set")

        self.config: ConfigDict = load_config(config_path)
        self.task: str = self.config['task']
        self.img_size: Tuple[int, int] = tuple(self.config['model']['img_size'])
        self.in_chans: int = self.config['model'].get('in_chans', 1)
        self.num_classes: int = self.config['data']['num_classes']
        self.backend_name = backend
        self.device = torch.device(device)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections

        # `get_strides()` (needed to decode detection anchors) is only
        # available on the eager `PerceptionModel`, so we always build one
        # (cheap -- a few MB of randomly-initialized weights if unused) to
        # serve as the source of truth for strides/task metadata regardless
        # of which backend actually runs the forward pass.
        self._reference_model = build_perception_model(self.config)
        if backend == 'pytorch' and weights_path:
            load_weights_only(weights_path, self._reference_model, map_location='cpu', strict=True)
        self._reference_model.eval()
        self._strides = self._reference_model.get_strides() if self.task == 'detection' else None

        self._backend: _InferenceBackend = self._build_backend(backend, weights_path, onnx_providers)

    def _build_backend(self, backend: str, weights_path: Optional[str],
                        onnx_providers: Optional[List[str]]) -> _InferenceBackend:
        if backend == 'pytorch':
            return _PyTorchBackend(self._reference_model, self.device)
        if backend == 'torchscript':
            return _TorchScriptBackend(weights_path, self.device)
        if backend == 'onnxruntime':
            return _ONNXRuntimeBackend(weights_path, self.device, providers=onnx_providers)
        if backend == 'tensorrt':
            return _TensorRTBackend(weights_path, self.device)
        raise AssertionError(f"unreachable: unknown backend '{backend}'")  # BACKENDS check happens earlier

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess(self, raw_image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Normalize a raw Bayer image into the `(B, C, H, W)` float32 array
        the model expects.

        Args:
            raw_image: One of:
                - `(H, W)`: single-image, single-channel Bayer mosaic.
                - `(1, H, W)` or `(C, H, W)`: single image, channel-first.
                - `(B, C, H, W)`: already-batched.
            Values are assumed to already be in `[0, 1]` (typical for a
            RAW image normalized by its bit-depth max, e.g. `raw / 1023.0`
            for 10-bit) OR in `[-1, 1]` matching training-time
            normalization (`data.normalize: true` in the configs) --
            this method only handles shape/batching, not intensity
            rescaling, since the correct rescaling depends on the sensor's
            bit depth, which this generic inferencer doesn't know. Rescale
            before calling `predict()` if your raw values aren't already
            in the range the model was trained on.
        Returns:
            `(B, C, H, W)` float32 numpy array, resized to `self.img_size`
            if the input spatial dims don't already match (nearest-neighbor
            is deliberately avoided for RAW data; use bilinear, which is
            safe for the coarse patch-level tokenization this model does,
            even though it technically perturbs the raw Bayer phase
            alignment slightly at non-integer scale factors -- for exact
            phase alignment, pre-crop/pad to `img_size` yourself instead of
            relying on this resize).
        """
        if isinstance(raw_image, torch.Tensor):
            arr = raw_image.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_image)

        arr = arr.astype(np.float32)

        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        elif arr.ndim != 4:
            raise ValueError(f"Expected raw_image with 2, 3, or 4 dims, got shape {arr.shape}")

        _, _, h, w = arr.shape
        if (h, w) != self.img_size:
            t = torch.from_numpy(arr)
            t = torch.nn.functional.interpolate(t, size=self.img_size, mode='bilinear', align_corners=False)
            arr = t.numpy()

        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(self, raw_outputs: List[np.ndarray]) -> Any:
        """Convert flattened raw backend outputs into task-level predictions.

        Detection: runs anchor decode + score threshold + class-wise NMS via
        `postprocess_detections`, returning a list (len=batch) of
        `{'boxes': (K,4) xyxy, 'scores': (K,), 'labels': (K,)}` dicts.
        Segmentation: returns the `(B, H, W)` int64 argmax class map.
        """
        tensors = [torch.from_numpy(np.asarray(o)) for o in raw_outputs]

        if self.task == 'detection':
            num_levels = len(tensors) // 2
            cls_scores = tensors[:num_levels]
            bbox_preds = tensors[num_levels:]
            return postprocess_detections(
                cls_scores, bbox_preds, strides=self._strides, num_classes=self.num_classes,
                score_thresh=self.score_thresh, nms_thresh=self.nms_thresh,
                max_detections=self.max_detections, image_size=self.img_size,
            )

        seg_logits = tensors[0]
        return seg_logits.argmax(dim=1).numpy()

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(self, raw_image: Union[np.ndarray, torch.Tensor]) -> Any:
        """Run the full preprocess -> backend forward -> postprocess pipeline."""
        images = self.preprocess(raw_image)
        raw_outputs = self._backend.run(images)
        return self.postprocess(raw_outputs)

    def predict_raw(self, raw_image: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Like `predict`, but returns the raw (pre-postprocessing) tensors
        -- useful for cross-backend numerical parity checks (mirrors
        `tools/export.py`'s verification helpers) or custom postprocessing.
        """
        images = self.preprocess(raw_image)
        return self._backend.run(images)

    def benchmark(self, num_warmup: int = 5, num_runs: int = 20) -> Dict[str, float]:
        """Measure end-to-end (preprocess + backend forward + postprocess)
        latency for this inferencer's configured backend, using a random
        dummy input of `self.img_size`. For backend-forward-only timing
        (no postprocessing/NMS overhead), see
        `photon2perception.evaluation.efficiency.measure_latency`, which
        benchmarks the eager `PerceptionModel` directly.
        """
        dummy = np.random.randn(1, self.in_chans, *self.img_size).astype(np.float32)

        for _ in range(num_warmup):
            self.predict(dummy)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(dummy)
            latencies.append((time.perf_counter() - start) * 1000.0)

        latencies = np.array(latencies)
        return {
            'backend': self.backend_name,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'fps': float(1000.0 / np.mean(latencies)),
        }

    def close(self) -> None:
        """Release backend resources (ONNXRuntime session / TensorRT
        context). Safe to call multiple times or not at all for the
        'pytorch'/'torchscript' backends (pure-Python GC handles those)."""
        self._backend.close()

    def __enter__(self) -> "PerceptionInferencer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
