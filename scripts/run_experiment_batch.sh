#!/bin/bash
# ============================================================================
# Photon2Perception — AutoDL 批量实验启动脚本
# ============================================================================
# 用法：
#   bash scripts/run_experiment_batch.sh              # 交互式选择批次
#   bash scripts/run_experiment_batch.sh batch1       # 直接跑第 1 批
#   bash scripts/run_experiment_batch.sh all          # 跑所有实验（慎用！）
# ============================================================================

set -e

PROJECT_DIR="/root/autodl-tmp/photon2perception"
CONDA_ENV="photon2perception"
LOG_DIR="$PROJECT_DIR/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $1"; }

# ============================================================================
# 实验批次定义
# ============================================================================

# 第 1 批：快速验证（~2 GPU 小时）— 消融实验中跑得最快的几个
BATCH1_EXPS=(
    "E11a" "E11b" "E11c"   # 输入表示消融
    "E12a" "E12b" "E12c" "E12d"  # 位置编码消融
)

# 第 2 批：主实验 + 核心消融（~25 GPU 小时）
BATCH2_EXPS=(
    "E01" "E02" "E03" "E04"   # 主对比实验
    "E05" "E06" "E07" "E08" "E09"  # 不同条件下的主实验
    "E13a" "E13b" "E13c" "E13d"  # 稀疏路由消融
    "E14a" "E14b" "E14c"       # CFA 消融
    "E15a" "E15b"              # 元数据消融
)

# 第 3 批：鲁棒性测试（~20 GPU 小时）
BATCH3_EXPS=(
    "E16" "E17" "E18" "E19" "E20"  # 退化鲁棒性
    "E21" "E22" "E23" "E24"        # 跨域泛化
    "E25"                          # OOD 稳定性
)

# 第 4 批：效率分析（~5 GPU 小时，主要是推理）
BATCH4_EXPS=(
    "E26" "E27" "E28" "E29" "E30" "E31"
)

# 第 5 批：定性分析 + Pareto（~3 GPU 小时）
BATCH5_EXPS=(
    "E32" "E33" "E34" "E35" "E36"
    "E10"  # Pareto 前沿（依赖 E01-E04）
)

# ============================================================================
# 工具函数
# ============================================================================

check_environment() {
    log_step "检查环境..."

    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "项目目录不存在: $PROJECT_DIR"
        log_error "请先运行 scripts/setup_autodl.sh"
        exit 1
    fi

    # 检查 conda 环境
    if ! conda env list | grep -q "$CONDA_ENV"; then
        log_error "Conda 环境 $CONDA_ENV 不存在"
        log_error "请先运行 scripts/setup_autodl.sh"
        exit 1
    fi

    # 检查 GPU
    GPU_COUNT=$(conda run -n "$CONDA_ENV" python -c "import torch; print(torch.cuda.device_count())")
    if [ "$GPU_COUNT" -eq 0 ]; then
        log_error "没有可用的 GPU！"
        exit 1
    fi
    log_info "检测到 $GPU_COUNT 个 GPU"

    # 检查数据目录
    if [ ! -d "$PROJECT_DIR/data" ] || [ -z "$(ls -A $PROJECT_DIR/data 2>/dev/null)" ]; then
        log_warn "数据目录为空，请先下载数据集到 $PROJECT_DIR/data/"
        log_warn "至少需要 COCO 或 PASCAL RAW 数据集"
        read -p "  是否继续? (y/n): " cont
        if [ "$cont" != "y" ]; then
            exit 1
        fi
    fi
}

run_single_experiment() {
    local exp_id=$1
    local gpu_id=${2:-0}

    log_info "启动实验 $exp_id (GPU $gpu_id)..."

    conda run -n "$CONDA_ENV" python tools/run_experiments.py run \
        --exp-id "$exp_id" \
        --gpus "$gpu_id" \
        --output "$PROJECT_DIR/outputs" \
        > "$LOG_DIR/${exp_id}_${TIMESTAMP}.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_info "实验 $exp_id 完成 ✓"
        echo "$exp_id PASS" >> "$LOG_DIR/batch_results_${TIMESTAMP}.txt"
    else
        log_error "实验 $exp_id 失败 ✗ (exit code: $exit_code)"
        echo "$exp_id FAIL (exit $exit_code)" >> "$LOG_DIR/batch_results_${TIMESTAMP}.txt"
    fi

    return $exit_code
}

run_batch() {
    local batch_name=$1
    shift
    local exp_list=("$@")

    log_step "========== 开始批次: $batch_name =========="
    log_info "共 ${#exp_list[@]} 个实验"
    log_info "日志目录: $LOG_DIR"
    echo ""

    local passed=0
    local failed=0
    local start_time=$(date +%s)

    for exp_id in "${exp_list[@]}"; do
        log_step "--- $exp_id ---"
        if run_single_experiment "$exp_id" 0; then
            ((passed++))
        else
            ((failed++))
        fi
        echo ""

        # 每个实验后检查显存是否释放
        conda run -n "$CONDA_ENV" python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'  显存已清理: {torch.cuda.memory_allocated(0)//1024**2} MB 占用')
" 2>/dev/null || true
    done

    local end_time=$(date +%s)
    local duration=$(( (end_time - start_time) / 3600 ))
    local minutes=$(( (end_time - start_time) % 3600 / 60 ))

    echo ""
    log_step "========== 批次完成: $batch_name =========="
    log_info "通过: $passed / 失败: $failed"
    log_info "耗时: ${duration}h ${minutes}m"
    echo ""
}

# ============================================================================
# 主入口
# ============================================================================

main() {
    mkdir -p "$LOG_DIR"

    echo ""
    echo "============================================"
    echo " Photon2Perception — 批量实验启动"
    echo "============================================"
    echo ""

    check_environment

    local batch="${1:-}"
    shift 2>/dev/null || true

    # 如果没有指定批次，交互式选择
    if [ -z "$batch" ]; then
        echo "可选批次:"
        echo "  batch1  — 快速验证（输入 & PE 消融，~2h, 7 个实验）"
        echo "  batch2  — 主实验 + 核心消融（~25h, 19 个实验）"
        echo "  batch3  — 鲁棒性测试（~20h, 10 个实验）"
        echo "  batch4  — 效率分析（~5h, 6 个实验）"
        echo "  batch5  — 定性分析 + Pareto（~3h, 6 个实验）"
        echo "  all     — 全部 48 个实验（~55h, 慎用！）"
        echo ""
        read -p "请选择批次 [batch1]: " batch
        batch="${batch:-batch1}"
    fi

    cd "$PROJECT_DIR"

    case "$batch" in
        batch1)
            run_batch "第1批: 快速验证" "${BATCH1_EXPS[@]}"
            ;;
        batch2)
            run_batch "第2批: 主实验 + 消融" "${BATCH2_EXPS[@]}"
            ;;
        batch3)
            run_batch "第3批: 鲁棒性测试" "${BATCH3_EXPS[@]}"
            ;;
        batch4)
            run_batch "第4批: 效率分析" "${BATCH4_EXPS[@]}"
            ;;
        batch5)
            run_batch "第5批: 定性分析" "${BATCH5_EXPS[@]}"
            ;;
        all)
            log_warn "即将运行全部 48 个实验，预计耗时 ~55 GPU 小时"
            log_warn "请确认你已了解这个操作！"
            read -p "  确认? (输入 yes 继续): " confirm
            if [ "$confirm" != "yes" ]; then
                log_info "已取消"
                exit 0
            fi
            run_batch "第1批: 快速验证" "${BATCH1_EXPS[@]}"
            run_batch "第2批: 主实验 + 消融" "${BATCH2_EXPS[@]}"
            run_batch "第3批: 鲁棒性测试" "${BATCH3_EXPS[@]}"
            run_batch "第4批: 效率分析" "${BATCH4_EXPS[@]}"
            run_batch "第5批: 定性分析" "${BATCH5_EXPS[@]}"
            ;;
        *)
            log_error "未知批次: $batch"
            log_info "可选: batch1, batch2, batch3, batch4, batch5, all"
            exit 1
            ;;
    esac

    echo ""
    log_info "所有批次完成！结果汇总见: $LOG_DIR/batch_results_${TIMESTAMP}.txt"
}

main "$@"
