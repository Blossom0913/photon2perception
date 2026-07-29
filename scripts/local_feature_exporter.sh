#!/bin/bash
# ============================================================================
# Photon2Perception — 本地特征导出脚本 (local_feature_exporter.sh)
# ============================================================================
# 作用：
#   调用 tools/export_features.py，把原始数据（COCO/Cityscapes RGB 图像，
#   或真实 RAW 文件）转换成训练可直接使用的 tensor（Bayer 化 + resize +
#   归一化），分片写入磁盘（<output_dir>/{split}_shard_XXXXX.pt +
#   {split}_manifest.json），并（重新）生成描述输入/输出张量 shape 的
#   `.pb.txt` 特征说明文件（见 photon2perception/utils/feature_spec.py）。
#
# 用法：
#   bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml
#   bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml --split val
#   bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml --spec_only
#   bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml \
#       --limit 32 --output_dir /tmp/feat_debug   # 冒烟测试：只导出前 32 条样本
#
# 环境：
#   优先使用名为 photon2perception 的 conda 环境（若存在），否则回退到
#   当前 shell 已激活的 Python 环境（例如已手动 `conda activate` 或使用venv）。
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
用法: bash scripts/local_feature_exporter.sh --config <config.yaml> [选项]

必需参数:
  --config <path>       实验配置文件 (tasks/detection/config/*.yaml 或 tasks/segmentation/config/*.yaml)

可选参数:
  --split <train|val>   要导出的数据集划分 (默认: train)
  --output_dir <path>   导出目录 (默认: 读取 config.feature_export.output_dir)
  --format <pt|npy>     导出张量格式 (默认: 读取 config.feature_export.format)
  --shard_size <N>      每个分片文件包含的样本数 (默认: 读取 config.feature_export.shard_size)
  --limit <N>           只导出前 N 条样本，便于快速冒烟测试
  --spec_only           只（重新）生成 .pb.txt 特征说明文件，不导出任何张量
  --override k=v [...]  透传给 tools/export_features.py 的配置覆盖项 (dotted-key)
  -h, --help            显示本帮助

示例:
  bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml
  bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml --split val
  bash scripts/local_feature_exporter.sh --config tasks/detection/config/photon2percept_det_bayer.yaml --spec_only
EOF
}

CONFIG=""
SPLIT="train"
OUTPUT_DIR=""
FORMAT=""
SHARD_SIZE=""
LIMIT=""
SPEC_ONLY=""
OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --format) FORMAT="$2"; shift 2 ;;
        --shard_size) SHARD_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --spec_only) SPEC_ONLY="1"; shift 1 ;;
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

# ---------- 组装命令 ----------
CMD_ARGS=(tools/export_features.py --config "$CONFIG" --split "$SPLIT")
[ -n "$OUTPUT_DIR" ] && CMD_ARGS+=(--output_dir "$OUTPUT_DIR")
[ -n "$FORMAT" ] && CMD_ARGS+=(--format "$FORMAT")
[ -n "$SHARD_SIZE" ] && CMD_ARGS+=(--shard_size "$SHARD_SIZE")
[ -n "$LIMIT" ] && CMD_ARGS+=(--limit "$LIMIT")
[ -n "$SPEC_ONLY" ] && CMD_ARGS+=(--emit_spec_only)
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    CMD_ARGS+=(--override "${OVERRIDES[@]}")
fi

log_step "开始特征导出"
log_info "配置文件: $CONFIG"
log_info "数据划分: $SPLIT"
[ -n "$SPEC_ONLY" ] && log_info "模式: 仅生成 .pb.txt 特征说明文件"
echo ""

START_TIME=$(date +%s)
$PYTHON_CMD "${CMD_ARGS[@]}"
EXIT_CODE=$?
END_TIME=$(date +%s)

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log_info "特征导出完成 ✓ (耗时 $((END_TIME - START_TIME))s)"
else
    log_error "特征导出失败 ✗ (exit code: $EXIT_CODE)"
fi
exit $EXIT_CODE
