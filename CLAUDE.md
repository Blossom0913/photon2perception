# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the **photon2perception** research project — a full-stack PyTorch implementation of a brain-inspired, structure-preserving RAW image perception framework, alongside its paper manuscript (README.md), research idea sketch (prompts/idea.md), and an organized literature review of ~19 papers on RAW-domain computer vision. The codebase implements the complete pipeline end-to-end: Bayer RAW tokenization → multi-task perception heads → real detection/segmentation losses → training/validation loop → COCO mAP / mIoU evaluation → TorchScript/ONNX export → a unified multi-backend inference API, with experiment management for 36 planned experiments.

**Core research thesis:** Build a brain-inspired, efficient perception framework that operates directly on Bayer RAW tokens (preserving the 2D CFA structure), uses 2D RoPE for spatial position encoding with optional directional enhancement, and employs saliency/uncertainty-aware sparse routing to reduce latency, memory bandwidth, and compute while maintaining or improving multi-task perception performance.

## Repository structure

```
photon2perception/
├── README.md                          # Full paper outline (Introduction, Contributions, Experiments)
├── prompts/idea.md                    # Early-stage idea sketch and open questions
├── reference_pdf/                     # ~19 source papers (PDFs) on RAW perception, ISP, and related topics
│   └── summary_notes/                 # Chinese-language summaries of each paper
├── photon2perception/                 # Core Python package
│   ├── models/
│   │   ├── tokenization/              # BayerPatchEmbed, BayerFineTokenize
│   │   ├── position_encoding/         # RoPE2D, CFAwareRoPE2D, DirectionalEnhance
│   │   ├── routing/                   # SaliencyRouter, UncertaintyRouter, PhysicalPriorRouter
│   │   ├── backbones/                 # RawViT (RAW-adapted Vision Transformer; attn_backend sdpa/flash/math)
│   │   ├── necks/                     # SimpleFeaturePyramidNeck (ViTDet-style 1D tokens -> multi-scale FPN)
│   │   ├── heads/                     # RawDetectionHead, RawSegmentationHead, postprocess (NMS/argmax)
│   │   └── model_wrapper.py           # PerceptionModel + build_perception_model (single source of truth)
│   ├── datasets/                      # BaseRAWDataset, CocoRawDetectionDataset, CityscapesRawSegmentationDataset, UnprocessPipeline
│   │   └── raw_transforms/            # Bayer-safe augmentations
│   ├── losses/                        # DetectionLoss (focal+L1/GIoU), SegmentationLoss (CE+RMI)
│   ├── evaluation/                    # DetectionEvaluator (mAP), SegmentationEvaluator (mIoU), efficiency.py
│   ├── utils/                         # config, checkpoint, distributed (DDP), logger, registry
│   └── inference.py                   # PerceptionInferencer: pytorch/torchscript/onnxruntime/tensorrt backends
├── configs/                           # YAML config files (detection, segmentation)
├── tools/                             # train.py, eval.py, export.py, run_experiments.py, visualize.py
├── tests/                             # Unit tests: test_core.py (model/arch), test_infra.py (train/eval/export/infer pipeline)
├── scripts/                           # AutoDL setup & batch experiment scripts
│   ├── setup_autodl.sh                # One-click AutoDL environment setup
│   └── run_experiment_batch.sh        # Batch experiment runner (5 batches)
├── docs/                              # Guides and documentation
│   └── AUTODL_GUIDE.md                # Complete AutoDL usage guide
├── experiments/                       # Experiment tracking directory
├── reference_code/                    # Cloned reference repos (gitignored)
└── CLAUDE.md
```

## Code architecture

All code follows the **mmdetection/mmsegmentation plugin pattern** used by RAW-Adapter and AODRaw. The framework is designed as a standalone PyTorch package that can be integrated with OpenMMLab tools.

### Data flow

```
Bayer RAW (B,1,H,W)
  → BayerPatchEmbed (CFA-aware tokenization, patch_size=16)
  → CFA phase embeddings added
  → [Optional] DirectionalEnhance (gated residual)
  → [Optional] Sparse routing (saliency/uncertainty/physical)
  → RawViT blocks (2D RoPE applied to patch tokens only, CLS token excluded; attn_backend sdpa/flash/math)
  → CLS token + hidden states
  → PerceptionModel: Neck (SimpleFeaturePyramidNeck, detection only) → Task head
  → detection: (cls_scores, bbox_preds) per FPN level | segmentation: (B, num_classes, H, W) logits
```

`PerceptionModel` (`photon2perception/models/model_wrapper.py`) is the single `backbone -> neck -> head`
`nn.Module` used unmodified by `tools/train.py`, `tools/eval.py`, `tools/export.py`, and
`photon2perception/inference.py` — this is what keeps train/eval/export from drifting apart. Always
construct models via `build_perception_model(config)` rather than instantiating `RawViT`/heads directly.

### Key design decisions

1. **BayerPatchEmbed** requires even patch_size to capture complete 2×2 Bayer quads.
2. **2D RoPE** splits embedding into 4 equal parts: x-axis, y-axis, diagonal, anti-diagonal frequencies. It is applied once over the full `embed_dim` (all heads share the same rotation, per `apply_shared_rope_multihead` in `raw_vit.py`), and only to patch tokens — the CLS token is split off before rotation and re-concatenated after, since it has no 2D spatial position.
3. **Sparse routing** uses a gated combination of learned saliency + physical prior (local variance in RAW values). The `PhysicalPriorRouter` is the key differentiator from generic token pruning. **Important:** routing is only active during `.eval()`/inference if the backbone was built with `route_at_inference=True` (default `False`, for backward compatibility). Check `PerceptionModel.routing_active` before assuming an exported/deployed model actually runs sparse — `tools/export.py` asserts on this and fails loudly for a misconfigured `use_sparse_routing=True, route_at_inference=False` model, since that would silently export a dense graph.
4. **DirectionalEnhance** is gated with `tanh(gate)` and initialized at 0 (disabled at start).
5. **All experiments must report efficiency alongside accuracy** — latency, FLOPs, memory, and input bandwidth are first-class metrics (`photon2perception/evaluation/efficiency.py`, CUDA/MPS/CPU-safe — peak memory is CUDA-only and reported as 0.0 elsewhere rather than raising).
6. **Attention backend is switchable** (`attn_backend`: `'sdpa'` default / `'flash'` / `'math'`) via `RawViT.set_attn_backend()` — useful to force `'math'` before ONNX export/NPU targets that don't support the fused SDPA op, while using `'sdpa'`/`'flash'` for actual GPU training.

### Running the code

**On a GPU machine (AutoDL / server):**

```bash
# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Run the full test suite (82 tests: test_core.py = model/architecture,
# test_infra.py = train/eval/export/inference pipeline)
python -m pytest tests/ -v
```

**On a CPU-only machine (local development, incl. Apple Silicon/MPS):**

```bash
# Install CPU PyTorch (or use the Miniconda env described below)
conda install pytorch==2.1.0 torchvision==0.16.0 cpuonly -c pytorch
pip install -r requirements.txt

# Run tests — TestSanityCheck::test_overfit_tiny_batch (test_core.py) will
# fail on CPU (needs a GPU to converge in the test's step budget); all other
# tests, including the full test_infra.py infra suite, pass on CPU.
python -m pytest tests/ -v -k "not test_overfit_tiny_batch"
```

```bash
# List all experiments
python tools/run_experiments.py list

# Dry-run a single experiment
python tools/run_experiments.py dry-run --exp-id E01

# Train a model (supports CPU/MPS/single-GPU/DDP multi-GPU; auto-resume,
# mixed precision, checkpointing, and console+file[+TB/W&B] logging --
# see tools/train.py's module docstring)
python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml
python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml \
    --override training.epochs=5 data.batch_size=2 --output_dir ./outputs/debug

# Evaluate a checkpoint (COCO mAP / mIoU + efficiency report)
python tools/eval.py --config configs/detection/photon2percept_det_bayer.yaml \
    --checkpoint outputs/photon2percept_det_bayer/checkpoint_best.pth

# Export to TorchScript + ONNX (with numerical parity verification)
python tools/export.py --config configs/detection/photon2percept_det_bayer.yaml \
    --checkpoint outputs/photon2percept_det_bayer/checkpoint_best.pth \
    --format both --output exported/det_bayer

# Generate visualizations
python tools/visualize.py
```

**Inference (any backend, same API):**

```python
from photon2perception.inference import PerceptionInferencer

inferencer = PerceptionInferencer(
    config_path='configs/detection/photon2percept_det_bayer.yaml',
    backend='onnxruntime',  # or 'pytorch' / 'torchscript' / 'tensorrt'
    weights_path='exported/det_bayer.onnx',
)
detections = inferencer.predict(raw_bayer_image)  # list of dicts: boxes/scores/labels
stats = inferencer.benchmark()  # latency/FPS
```

### Environment setup

#### Anaconda (recommended for development)

```bash
# Create conda environment
conda create -n photon2perception python=3.10 -y
conda activate photon2perception

# Install PyTorch (CUDA 11.8 — adjust for your CUDA version)
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies (see requirements.txt for the full/authoritative list,
# including pycocotools for dataset loading and onnx/onnxruntime for export+inference)
pip install -r requirements.txt

# Optional: install mmdetection & mmsegmentation for full dataset/eval support
pip install openmim
mim install mmdet==3.3.0 mmsegmentation==1.2.0

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/ -v
```

**Conda environment export / restore:**

```bash
# Export environment (for reproducibility)
conda env export --no-builds > environment.yml

# Restore from file
conda env create -f environment.yml
```

**GPU requirements:** A single NVIDIA GPU with ≥8GB VRAM (Tesla V100, RTX 2080Ti, RTX 3090, A6000). Multi-GPU training supported via DDP. Paper baselines were run on: 1× Tesla V100 (RAW-Adapter, Dark-ISP), 4× RTX A6000 (Dr. RAW), 1× RTX 3090 (LIS, metadata RAW).

#### Docker

**Dockerfile** (`Dockerfile` at repo root):

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: OpenMMLab packages
RUN pip install --no-cache-dir openmim && \
    mim install mmdet==3.3.0 mmsegmentation==1.2.0

# Copy project
COPY . .

# Entry point
CMD ["python", "-m", "pytest", "tests/", "-v"]
```

**Build and run:**

```bash
# Build image
docker build -t photon2perception:latest .

# Run tests
docker run --gpus all photon2perception:latest

# Interactive development
docker run --gpus all -it -v $(pwd):/workspace photon2perception:latest bash

# Train a model
docker run --gpus all -v $(pwd):/workspace -v /path/to/data:/data \
    photon2perception:latest \
    python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml
```

**docker-compose.yml** (for multi-GPU setups):

```yaml
version: '3.8'
services:
  train:
    image: photon2perception:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    volumes:
      - .:/workspace
      - ./data:/data
      - ./outputs:/outputs
    command: python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml
    shm_size: '16gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
```

#### AutoDL (GPU cloud — recommended for training)

This project is designed to run on [AutoDL](https://www.autodl.com) for GPU training. See [docs/AUTODL_GUIDE.md](docs/AUTODL_GUIDE.md) for the complete walkthrough.

**Quick start on AutoDL:**

```bash
# 1. Clone code
cd /root/autodl-tmp
git clone <your-repo-url> photon2perception
cd photon2perception

# 2. One-click setup
bash scripts/setup_autodl.sh

# 3. Download COCO dataset to data/

# 4. Dry-run to validate
conda activate photon2perception
python tools/run_experiments.py dry-run --exp-id E01

# 5. Run experiments in batches
bash scripts/run_experiment_batch.sh batch1   # Quick validation (~2h)
bash scripts/run_experiment_batch.sh batch2   # Main results (~25h)
```

**Estimated GPU time:** ~55 hours on RTX 3090 for all 36 experiments (~140 CNY total).

**Key scripts:**
- [scripts/setup_autodl.sh](scripts/setup_autodl.sh) — One-click environment setup (system deps + conda + PyTorch + tests)
- [scripts/run_experiment_batch.sh](scripts/run_experiment_batch.sh) — Batch experiment runner with 5 predefined batches

**GPU selection guide:**

| GPU | VRAM | Price | Recommendation |
|-----|------|-------|----------------|
| RTX 3090 | 24 GB | ~2.5 CNY/h | **Primary choice** — best price/performance |
| RTX 4090 | 24 GB | ~4 CNY/h | Faster training, ~1.5-2× 3090 speed |
| A6000 | 48 GB | ~6 CNY/h | Large batches, high resolution, multi-GPU DDP |

### Reference repositories

| Paper | GitHub | Status |
|-------|--------|--------|
| RAW-Adapter (ECCV 2024) | `github.com/cuiziteng/ECCV_RAW_Adapter` | Full code |
| AODRaw (CVPR 2025) | `github.com/lzyhha/AODRaw` | Full code |
| Dr. RAW (NeurIPS 2025) | `github.com/WJ-Huang/Dr-RAW-...` | README only |
| RawNeRF (CVPR 2022) | `github.com/google-research/multinerf` | JAX-based |
| LIS (IJCV 2023) | `github.com/Linwei-Chen/LIS` | Full code |

## The paper outline (README.md)

The README is a structured paper draft with four sections:

1. **Introduction** (four-paragraph structure): problem background → limitations of existing RAW perception methods → proposed approach (Bayer-native tokenization + 2D RoPE + sparse routing) → contributions preview.
2. **Contributions** (three items): structure-preserving RAW framework, 2D RoPE + directional enhancement for RAW token representation, physics-driven sparse routing.
3. **Experiments** (§4.1–4.6): Setup, Main Results, Ablations, Robustness/Generalization, Efficiency Analysis, Qualitative Analysis.
4. **Key experimental axes:** detection (primary), segmentation (secondary), local feature / transfer tasks. Comparisons against RGB pipeline, demosaic RAW pipeline, learnable ISP/adapter methods, and direct RAW perception methods. Efficiency measured by latency, FLOPs, memory bandwidth, input byte count.

## Research idea (prompts/idea.md)

The original prompt asks for help designing a framework around:
- A single core contribution: brain-inspired intelligence for RAW perception, reducing latency and bandwidth ("photon to perception, brain inspired").
- Integrating 2D RoPE (DRoPE) for tokenizing images.
- Multi-task perception evaluation (not just detection).
- Literature planning and experimental design.

## Literature organization

Papers in `reference_pdf/` cover these themes:
- **Direct RAW perception:** Lin (RAW for robotic vision), Lu (object detection on Bayer, RawSeg), Li (RAW detection in diverse conditions), Chan (automotive RAW detection)
- **RAW-to-RGB / ISP:** RMFA-Net, Neural Photo-Finishing, DiffuseRAW, Dark-ISP, RAW-Adapter
- **RAW restoration & reconstruction:** Ke (training on RAW/HDR), Li (metadata-based RAW reconstruction), Fanous (biophotonic data)
- **Low-light & HDR:** RawNeRF (NeRF in the dark), Chen (instance segmentation in the dark), Guo (Dark-ISP)
- **Keypoints & features on RAW:** Lin (keypoint detection/description on Bayer)
- **Surveys & editorials:** general RAW-based image processing review, Niu & Zhang (CNN-based image processing)

Each paper has a corresponding Chinese summary in `reference_pdf/summary_notes/`.

## Infrastructure status (previously "Known gaps", now resolved)

The full pipeline — architecture, losses, dataset loading, training, validation, evaluation, export, and
inference — is implemented and covered by 82 passing unit tests (`tests/test_core.py` +
`tests/test_infra.py`) plus an end-to-end smoke test (tiny COCO-format dataset through
train → eval → export → inference on CPU). What used to be placeholder stubs is now:

1. **Real losses** (`photon2perception/losses/`): `DetectionLoss` (focal classification + L1/GIoU box
   regression, anchor-based) and `SegmentationLoss` (CrossEntropy + RMI, optional auxiliary head loss).
   Wired into `tools/train.py` via `build_loss(config)`.
2. **Real dataset loading** (`photon2perception/datasets/coco_raw_dataset.py`): `CocoRawDetectionDataset`
   and `CityscapesRawSegmentationDataset`, standalone loaders (no mmdet/mmseg dependency) that read
   COCO-format / Cityscapes-format annotations and synthesize Bayer RAW on-the-fly via `UnprocessPipeline`.
   Selected via `data.type: coco|cityscapes|real` in the YAML config (`synthetic` is a back-compat alias).
   Requires `pycocotools` for the COCO path (see requirements.txt).
3. **Backbone/head bridge solved via `PerceptionModel`** (`photon2perception/models/model_wrapper.py`):
   `SimpleFeaturePyramidNeck` (ViTDet-style Simple Feature Pyramid) converts RawViT's 1D token sequence
   into a multi-scale 2D feature pyramid for the detection head; the segmentation head consumes the
   last hidden state's patch tokens directly. `build_perception_model(config)` is the single
   config-dict-to-nn.Module constructor used by train/eval/export — do not hand-assemble
   backbone+neck+head elsewhere.
4. **Real validation loop**: `build_dataloaders` in `tools/train.py` returns a proper `val_loader`;
   `tools/eval.py` runs full COCO mAP (`DetectionEvaluator`, pycocotools-based with an approx-AP50
   fallback) or mIoU (`SegmentationEvaluator`) plus an efficiency report.
5. **Mixed precision wired**: `training.mixed_precision: true` wraps the forward pass in
   `torch.autocast` + `GradScaler` in `tools/train.py`.
6. **Unified logging**: `photon2perception/utils/logger.py`'s `ExperimentLogger` writes to
   console + a log file, with optional TensorBoard/W&B backends (`--use_wandb`), replacing bare `print()`.

### Things to double-check before a real (non-smoke) training run

- **Sparse routing at inference**: routing only runs during `.eval()`/export if the model config sets
  `route_at_inference: true` (default `false`). See "Key design decisions" #3 above — an experiment
  ablating sparse routing must set this explicitly, or the exported/evaluated model will silently run dense.
- **`torch.load` and checkpoints**: checkpoints embed a `ConfigDict` and other non-tensor state, so
  `photon2perception/utils/checkpoint.py` loads with `weights_only=False` — only load checkpoints you
  trust (own training runs / vetted releases).
- **ONNX export defaults to the legacy TorchScript-tracing exporter** (`dynamo=False`) for broader
  onnxruntime/vendor-NPU-toolchain compatibility; pass `--onnx_dynamo` to opt into the newer
  torch.export-based exporter once a target toolchain is confirmed to support it.
- **`data.batch_size` and full-scale configs**: `configs/{detection,segmentation}/*.yaml` use
  production-scale settings (`embed_dim: 768`, `img_size: [512, 512]`, real COCO paths under
  `data.train_img_dir`/`train_ann_file`). For local CPU iteration/smoke-testing, override with a tiny
  config (small `embed_dim`/`img_size`/`depth`, `data.batch_size=1-2`) rather than running the full
  config on CPU.

## Working with this repo

When asked to help with the research:
- Read the relevant paper summaries before proposing approaches — the literature review is extensive and directly informs the research direction.
- The paper outline in README.md is the authoritative structure; changes to the research direction should be reflected there.
- New paper summaries added to `reference_pdf/summary_notes/` should follow the existing naming convention: `论文概括：{English title} — {Chinese description}.md`.
- When discussing experimental design, reference the ablation structure in README.md §4.3.

When asked to change code:
- Run `python -m pytest tests/ -v` after any change under `photon2perception/` or `tools/` — the 82-test
  suite (`test_core.py` + `test_infra.py`) is fast (~10s on CPU) and covers both architecture correctness
  and the full train/eval/export/inference pipeline, so it catches regressions immediately.
- Always go through `build_perception_model(config)` (`photon2perception/models/model_wrapper.py`) to
  construct a model, rather than instantiating `RawViT`/heads/neck directly — this is what keeps
  `tools/train.py`, `tools/eval.py`, `tools/export.py`, and `photon2perception/inference.py` in sync.
- If you touch anything that affects the forward pass under tracing (control flow depending on a tensor
  value, `.item()`, Python-side shape branching), re-run the export smoke tests
  (`tests/test_infra.py::TestExportScript`) specifically — they exercise `tools/export.py` end-to-end via
  subprocess and assert TorchScript/ONNX numerical parity against eager mode.
- For a genuine end-to-end smoke test beyond unit tests (real `tools/train.py` → `tools/eval.py` →
  `tools/export.py` → `photon2perception.inference.PerceptionInferencer` CLI invocations), generate a
  tiny COCO-format dataset (a handful of small images + a matching `annotations.json`) and a tiny model
  config (small `embed_dim`/`depth`/`img_size`, `data.batch_size=1-2`, `training.epochs=1`) in a scratch
  directory — this is fast on CPU and validates the actual CLI entry points, not just importable functions.
