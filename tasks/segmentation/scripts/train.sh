#!/bin/bash
# ============================================================================
# tasks/segmentation/scripts/train.sh
# ============================================================================
# 分割任务的训练快捷脚本：固定使用本任务自带的
# tasks/segmentation/config/photon2percept_seg_bayer.yaml，是对仓库级
# scripts/train_local.sh 的任务专属薄封装（本脚本不重复实现 conda/设备探测
# /TensorBoard 启动等通用逻辑，直接委托给 scripts/train_local.sh）。
#
# 用法：
#   bash tasks/segmentation/scripts/train.sh
#   bash tasks/segmentation/scripts/train.sh --exp_name my_run --auto_resume
#   bash tasks/segmentation/scripts/train.sh --override training.epochs=20 data.batch_size=4
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$TASK_DIR/../.." && pwd)"

exec bash "$PROJECT_DIR/scripts/train_local.sh" \
    --config "$TASK_DIR/config/photon2percept_seg_bayer.yaml" \
    "$@"
