#!/bin/bash
# ============================================================================
# Photon2Perception — 本地/单机训练启动脚本 (train_local.sh)
# ============================================================================
# 作用：
#   在本地开发机或单机 GPU 服务器上启动一次完整的 tools/train.py 训练，
#   自动探测可用的 conda 环境与计算设备（CUDA / Apple Silicon MPS / CPU），
#   并按需启动 TensorBoard 便于实时查看训练曲线。
#
# 与 scripts/run_experiment_batch.sh 的区别：
#   run_experiment_batch.sh 面向 AutoDL 云端批量跑 36 个论文实验；
#   train_local.sh 面向单次、交互式的本地/服务器训练（默认使用当前项目自带
#   的 configs/*.yaml，不依赖 AutoDL 的固定目录布局 /root/autodl-tmp/...）。
#
# 用法：
#   bash scripts/train_local.sh --config configs/detection/photon2percept_det_bayer.yaml
#   bash scripts/train_local.sh --config configs/detection/photon2percept_det_bayer.yaml \
#       --exp_name my_run --output_dir ./outputs --auto_resume
#   bash scripts/train_local.sh --config configs/detection/photon2percept_det_bayer.yaml \
#       --override training.epochs=20 data.batch_size=4 --no_tensorboard_server
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
用法: bash scripts/train_local.sh --config <config.yaml> [选项]

必需参数:
  --config <path>          实验配置文件

可选参数:
  --exp_name <name>        实验名（输出子目录名），默认取配置文件名
  --output_dir <path>      输出根目录 (默认: ./outputs)
  --resume <ckpt_path>     从指定 checkpoint 恢复训练
  --auto_resume            自动从 output_dir/<exp_name> 下最新 checkpoint 恢复
  --seed <N>               随机种子 (默认: 42)
  --gpu <id>                指定使用的 CUDA 设备号 (设置 CUDA_VISIBLE_DEVICES)
  --use_wandb               启用 Weights & Biases 日志
  --no_tensorboard          训练时禁用 TensorBoard 记录
  --no_tensorboard_server   不自动启动本地 TensorBoard 查看服务（仅记录，不起服务）
  --tensorboard_port <N>   TensorBoard 服务端口 (默认: 6006)
  --override k=v [...]     透传给 tools/train.py 的配置覆盖项 (dotted-key)
  -h, --help                显示本帮助

示例:
  bash scripts/train_local.sh --config configs/detection/photon2percept_det_bayer.yaml
  bash scripts/train_local.sh --config configs/segmentation/photon2percept_seg_bayer.yaml \\
      --exp_name seg_run1 --auto_resume
  bash scripts/train_local.sh --config configs/detection/photon2percept_det_bayer.yaml \\
      --override training.epochs=20 data.batch_size=4
EOF
}

CONFIG=""
EXP_NAME=""
OUTPUT_DIR="./outputs"
RESUME=""
AUTO_RESUME=""
SEED="42"
GPU_ID=""
USE_WANDB=""
NO_TENSORBOARD=""
NO_TB_SERVER=""
TB_PORT="6006"
OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --auto_resume) AUTO_RESUME="1"; shift 1 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --use_wandb) USE_WANDB="1"; shift 1 ;;
        --no_tensorboard) NO_TENSORBOARD="1"; shift 1 ;;
        --no_tensorboard_server) NO_TB_SERVER="1"; shift 1 ;;
        --tensorboard_port) TB_PORT="$2"; shift 2 ;;
        --override)
            shift 1
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                OVERRIDES+=("$1")
                shift 1
            done
            ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    log_error "缺少必需参数 --config"
    usage
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    log_error "配置文件不存在: $CONFIG"
    exit 1
fi

cd "$PROJECT_DIR"
[ -z "$EXP_NAME" ] && EXP_NAME="$(basename "$CONFIG" .yaml)"

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

# ---------- 计算设备探测 ----------
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    log_info "指定 GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
fi

DEVICE_INFO=$($PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    print(f'cuda x{torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    print('mps (Apple Silicon)')
else:
    print('cpu')
" 2>/dev/null || echo "unknown")
log_info "计算设备: $DEVICE_INFO"

# ---------- TensorBoard 服务 ----------
TB_PID=""
if [ -z "$NO_TENSORBOARD" ] && [ -z "$NO_TB_SERVER" ]; then
    if $PYTHON_CMD -c "import tensorboard" >/dev/null 2>&1; then
        TB_LOGDIR="$OUTPUT_DIR/$EXP_NAME"
        mkdir -p "$TB_LOGDIR"
        log_info "启动 TensorBoard: http://localhost:$TB_PORT (logdir=$TB_LOGDIR)"
        $PYTHON_CMD -m tensorboard.main --logdir "$TB_LOGDIR" --port "$TB_PORT" \
            > "$OUTPUT_DIR/$EXP_NAME.tensorboard.log" 2>&1 &
        TB_PID=$!
        log_info "TensorBoard PID: $TB_PID (日志: $OUTPUT_DIR/$EXP_NAME.tensorboard.log)"
    else
        log_warn "未安装 tensorboard，跳过自动启动 TensorBoard 服务（训练仍会写入事件文件，"
        log_warn "之后可手动运行: tensorboard --logdir $OUTPUT_DIR/$EXP_NAME/tensorboard）"
    fi
fi

cleanup() {
    if [ -n "$TB_PID" ]; then
        log_info "停止 TensorBoard (PID $TB_PID)"
        kill "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------- 组装训练命令 ----------
CMD_ARGS=(tools/train.py --config "$CONFIG" --exp_name "$EXP_NAME" --output_dir "$OUTPUT_DIR" --seed "$SEED")
[ -n "$RESUME" ] && CMD_ARGS+=(--resume "$RESUME")
[ -n "$AUTO_RESUME" ] && CMD_ARGS+=(--auto_resume)
[ -n "$USE_WANDB" ] && CMD_ARGS+=(--use_wandb)
[ -n "$NO_TENSORBOARD" ] && CMD_ARGS+=(--no_tensorboard)
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    CMD_ARGS+=(--override "${OVERRIDES[@]}")
fi

log_step "开始训练"
log_info "配置文件: $CONFIG"
log_info "实验名: $EXP_NAME"
log_info "输出目录: $OUTPUT_DIR/$EXP_NAME"
echo ""

START_TIME=$(date +%s)
$PYTHON_CMD "${CMD_ARGS[@]}"
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log_info "训练完成 ✓ (耗时 $((DURATION / 60))m$((DURATION % 60))s)"
    log_info "checkpoint: $OUTPUT_DIR/$EXP_NAME/checkpoint_last.pth"
    log_info "训练日志: $OUTPUT_DIR/$EXP_NAME/train.log"
else
    log_error "训练失败 ✗ (exit code: $EXIT_CODE)，详见: $OUTPUT_DIR/$EXP_NAME/train.log"
fi
exit $EXIT_CODE
