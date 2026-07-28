#!/bin/bash
# ============================================================================
# Photon2Perception — 训练冒烟测试脚本 (train_debug.sh)
# ============================================================================
# 作用：
#   端到端验证训练代码是否能正常跑通，而不依赖任何真实数据集或 GPU：
#     1. 用 tools/make_tiny_dataset.py 生成一份极小的合成 COCO 检测数据集
#        （和可选的合成 Cityscapes 分割数据集）到临时 scratch 目录；
#     2. 用一份被大幅缩小的模型/训练配置（小 embed_dim/depth/img_size，
#        1~2 个 epoch，batch_size=1~2）通过 --override 跑一遍
#        tools/train.py，覆盖 dataloader -> model -> loss -> optimizer ->
#        checkpoint 的完整链路；
#     3. 检查 checkpoint / 日志文件确实被写出，用退出码反映"训练代码能否
#        正常运行"这一问题的答案。
#   这与 tests/ 下的单元测试互补：单元测试验证各模块的正确性，本脚本验证
#   *实际 CLI 入口* (`python tools/train.py ...`) 在真实文件系统 I/O 下
#   可以跑通，参照 CLAUDE.md "Working with this repo" 一节的建议。
#
# 用法：
#   bash scripts/train_debug.sh                  # 只跑检测任务冒烟测试
#   bash scripts/train_debug.sh --task both       # 检测 + 分割都跑
#   bash scripts/train_debug.sh --task segmentation
#   bash scripts/train_debug.sh --keep_scratch    # 保留生成的临时数据/输出目录，便于排查
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="photon2perception"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $1"; }

usage() {
    cat <<EOF
用法: bash scripts/train_debug.sh [选项]

可选参数:
  --task <detection|segmentation|both>   要冒烟测试的任务 (默认: detection)
  --scratch_dir <path>                   临时数据/输出目录 (默认: 系统临时目录下自动创建)
  --keep_scratch                         结束后不删除临时目录（便于排查失败原因）
  --epochs <N>                           冒烟测试跑的 epoch 数 (默认: 1)
  -h, --help                             显示本帮助

示例:
  bash scripts/train_debug.sh
  bash scripts/train_debug.sh --task both --keep_scratch
EOF
}

TASK="detection"
SCRATCH_DIR=""
KEEP_SCRATCH=""
EPOCHS="1"

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --scratch_dir) SCRATCH_DIR="$2"; shift 2 ;;
        --keep_scratch) KEEP_SCRATCH="1"; shift 1 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "未知参数: $1"; usage; exit 1 ;;
    esac
done

case "$TASK" in
    detection|segmentation|both) ;;
    *) log_error "无效的 --task '$TASK'（可选: detection, segmentation, both）"; exit 1 ;;
esac

cd "$PROJECT_DIR"

# ---------- Python 环境探测 ----------
PYTHON_CMD="python"
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q "$CONDA_ENV"; then
    log_info "检测到 conda 环境 '$CONDA_ENV'，使用 conda run 执行"
    PYTHON_CMD="conda run -n $CONDA_ENV --no-capture-output python"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    log_info "使用当前已激活的 conda 环境: $CONDA_DEFAULT_ENV"
elif [ -n "$VIRTUAL_ENV" ]; then
    log_info "使用当前已激活的 venv: $VIRTUAL_ENV"
else
    log_warn "未检测到 conda/venv 环境，使用系统 'python'（可能导致依赖缺失）"
fi

# ---------- 临时目录 ----------
if [ -z "$SCRATCH_DIR" ]; then
    SCRATCH_DIR="$(mktemp -d /tmp/p2p_train_debug.XXXXXX)"
fi
DATA_DIR="$SCRATCH_DIR/data"
OUTPUT_DIR="$SCRATCH_DIR/outputs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
log_info "临时目录: $SCRATCH_DIR"

cleanup() {
    if [ -z "$KEEP_SCRATCH" ]; then
        log_info "清理临时目录 $SCRATCH_DIR"
        rm -rf "$SCRATCH_DIR"
    else
        log_info "保留临时目录（--keep_scratch）: $SCRATCH_DIR"
    fi
}
trap cleanup EXIT

PASS_COUNT=0
FAIL_COUNT=0

# ---------- 检测任务冒烟测试 ----------
run_detection_smoke() {
    log_step "=== 检测任务冒烟测试 ==="
    local coco_dir="$DATA_DIR/tiny_coco"

    log_info "生成合成 COCO 数据集 -> $coco_dir"
    $PYTHON_CMD tools/make_tiny_dataset.py --task detection --output_dir "$coco_dir" \
        --num_images 8 --img_h 64 --img_w 96 --num_classes 3

    log_info "运行 tools/train.py (tiny 检测配置, ${EPOCHS} epoch)"
    if $PYTHON_CMD tools/train.py \
        --config configs/detection/photon2percept_det_bayer.yaml \
        --exp_name debug_detection \
        --output_dir "$OUTPUT_DIR" \
        --no_tensorboard \
        --override \
            data.type=coco \
            data.train_img_dir="$coco_dir/images" \
            data.train_ann_file="$coco_dir/annotations.json" \
            data.num_classes=3 \
            data.batch_size=2 \
            data.num_workers=0 \
            'model.img_size=[64,96]' \
            'data.img_scale=[64,96]' \
            model.embed_dim=64 \
            model.depth=1 \
            model.num_heads=2 \
            training.epochs="$EPOCHS" \
            training.warmup_epochs=0 \
            training.save_interval=1 \
            training.val_interval=1 \
        > "$SCRATCH_DIR/detection_debug.log" 2>&1; then
        if [ -f "$OUTPUT_DIR/debug_detection/checkpoint_last.pth" ]; then
            log_info "检测任务冒烟测试通过 ✓ (checkpoint 已生成)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            log_error "检测任务训练命令成功退出，但未找到预期的 checkpoint_last.pth"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        log_error "检测任务冒烟测试失败 ✗，日志见: $SCRATCH_DIR/detection_debug.log"
        tail -n 30 "$SCRATCH_DIR/detection_debug.log" || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# ---------- 分割任务冒烟测试 ----------
run_segmentation_smoke() {
    log_step "=== 分割任务冒烟测试 ==="
    local cityscapes_dir="$DATA_DIR/tiny_cityscapes"

    log_info "生成合成 Cityscapes 数据集 -> $cityscapes_dir"
    $PYTHON_CMD tools/make_tiny_dataset.py --task segmentation --output_dir "$cityscapes_dir" \
        --num_images 8 --img_h 64 --img_w 96 --num_classes 4

    log_info "运行 tools/train.py (tiny 分割配置, ${EPOCHS} epoch)"
    if $PYTHON_CMD tools/train.py \
        --config configs/segmentation/photon2percept_seg_bayer.yaml \
        --exp_name debug_segmentation \
        --output_dir "$OUTPUT_DIR" \
        --no_tensorboard \
        --override \
            data.type=cityscapes \
            data.root_dir="$cityscapes_dir" \
            data.num_classes=4 \
            data.batch_size=2 \
            data.num_workers=0 \
            'model.img_size=[64,96]' \
            'data.img_scale=[64,96]' \
            'model.seg_output_size=[64,96]' \
            model.embed_dim=64 \
            model.depth=1 \
            model.num_heads=2 \
            training.epochs="$EPOCHS" \
            training.warmup_epochs=0 \
            training.lr_schedule=constant \
            training.save_interval=1 \
            training.val_interval=1 \
        > "$SCRATCH_DIR/segmentation_debug.log" 2>&1; then
        if [ -f "$OUTPUT_DIR/debug_segmentation/checkpoint_last.pth" ]; then
            log_info "分割任务冒烟测试通过 ✓ (checkpoint 已生成)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            log_error "分割任务训练命令成功退出，但未找到预期的 checkpoint_last.pth"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        log_error "分割任务冒烟测试失败 ✗，日志见: $SCRATCH_DIR/segmentation_debug.log"
        tail -n 30 "$SCRATCH_DIR/segmentation_debug.log" || true
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

echo ""
log_step "开始训练冒烟测试 (task=$TASK, epochs=$EPOCHS)"
echo ""

case "$TASK" in
    detection) run_detection_smoke ;;
    segmentation) run_segmentation_smoke ;;
    both) run_detection_smoke; run_segmentation_smoke ;;
esac

log_step "=== 冒烟测试汇总 ==="
log_info "通过: $PASS_COUNT | 失败: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
