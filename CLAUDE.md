# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is the **photon2perception** research project — a full-stack PyTorch implementation of a brain-inspired, structure-preserving RAW image perception framework, alongside its paper manuscript (README.md), research idea sketch (prompts/idea.md), and an organized literature review of ~19 papers on RAW-domain computer vision. The codebase contains 29 Python modules implementing the complete pipeline from Bayer RAW tokenization through multi-task perception heads, with experiment management for 36 planned experiments.

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
│   │   ├── backbones/                 # RawViT (RAW-adapted Vision Transformer)
│   │   └── heads/                     # RawDetectionHead, RawSegmentationHead
│   ├── datasets/                      # BaseRAWDataset, SyntheticRAWDataset, UnprocessPipeline
│   │   └── raw_transforms/            # Bayer-safe augmentations
│   └── evaluation/                    # Efficiency metrics, benchmark runner
├── configs/                           # YAML config files (detection, segmentation)
├── tools/                             # train.py, run_experiments.py, visualize.py
├── tests/                             # Unit tests (test_core.py)
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
  → RawViT blocks (with 2D RoPE in attention)
  → CLS token + hidden states
  → Task head (detection or segmentation)
```

### Key design decisions

1. **BayerPatchEmbed** requires even patch_size to capture complete 2×2 Bayer quads.
2. **2D RoPE** splits embedding into 4 equal parts: x-axis, y-axis, diagonal, anti-diagonal frequencies.
3. **Sparse routing** uses a gated combination of learned saliency + physical prior (local variance in RAW values). The `PhysicalPriorRouter` is the key differentiator from generic token pruning.
4. **DirectionalEnhance** is gated with `tanh(gate)` and initialized at 0 (disabled at start).
5. **All experiments must report efficiency alongside accuracy** — latency, FLOPs, memory, and input bandwidth are first-class metrics.

### Running the code

**On a GPU machine (AutoDL / server):**

```bash
# Install dependencies (Python 3.8+)
pip install torch torchvision rawpy pyyaml matplotlib scipy

# Run tests (all 19 should pass)
python -m pytest tests/test_core.py -v
```

**On a CPU-only machine (local development):**

```bash
# Install CPU PyTorch
conda install pytorch==2.1.0 torchvision==0.16.0 cpuonly -c pytorch

# Run tests — TestSanityCheck::test_overfit_tiny_batch will fail on CPU,
# the other 18 tests should pass
python -m pytest tests/test_core.py -v -k "not test_overfit_tiny_batch"
```

# List all experiments
python tools/run_experiments.py list

# Dry-run a single experiment
python tools/run_experiments.py dry-run --exp-id E01

# Train a model
python tools/train.py --config configs/detection/photon2percept_det_bayer.yaml

# Generate visualizations
python tools/visualize.py
```

### Environment setup

#### Anaconda (recommended for development)

```bash
# Create conda environment
conda create -n photon2perception python=3.10 -y
conda activate photon2perception

# Install PyTorch (CUDA 11.8 — adjust for your CUDA version)
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install rawpy pyyaml matplotlib scipy fvcore pytest

# Optional: install mmdetection & mmsegmentation for full dataset/eval support
pip install openmim
mim install mmdet==3.3.0 mmsegmentation==1.2.0

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/test_core.py -v
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
CMD ["python", "-m", "pytest", "tests/test_core.py", "-v"]
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

## Known gaps — must fix before real training

The core model architecture (BayerPatchEmbed, RoPE2D, RawViT, routing, detection/segmentation heads) is fully implemented and tested. However, the training pipeline has placeholder stubs that need to be completed before running actual experiments:

1. **Training loss is a dummy** ([tools/train.py:155](tools/train.py#L155)): `loss = cls_token.sum() * 0.0`. Must implement the actual detection loss (Focal loss + L1 regression) and segmentation loss (CrossEntropy + RMI).
2. **Dataset loading raises NotImplementedError** ([tools/train.py:113](tools/train.py#L113)): The synthetic dataset path needs a real RGB dataset loader — either integrate COCO/Cityscapes via mmdet/mmseg dataset classes, or implement a standalone loader.
3. **Detection head expects 2D features** but RawViT backbone outputs 1D CLS token + hidden states. Need a feature reshaping/upsampling bridge (or use dense prediction heads like Segmenter/DPT style).
4. **No validation loop**: `build_dataloaders` returns `None` for val_loader.
5. **No mixed precision wiring**: Config has `mixed_precision: false` but `train.py` never wraps forward pass in `torch.cuda.amp.autocast`. Enabling AMP would reduce VRAM by ~30%.
6. **No logging backend**: Despite docstring claims, there is no WandB/TensorBoard integration — only `print()` statements.

## Working with this repo

When asked to help with the research:
- Read the relevant paper summaries before proposing approaches — the literature review is extensive and directly informs the research direction.
- The paper outline in README.md is the authoritative structure; changes to the research direction should be reflected there.
- New paper summaries added to `reference_pdf/summary_notes/` should follow the existing naming convention: `论文概括：{English title} — {Chinese description}.md`.
- When discussing experimental design, reference the ablation structure in README.md §4.3.
