#!/bin/bash
# ============================================================================
# Photon2Perception — AutoDL 一键环境配置脚本
# ============================================================================
# 用法：在 AutoDL Jupyter Lab 终端中执行：
#   bash scripts/setup_autodl.sh
#
# 前置条件：
#   - AutoDL 实例已开机，选的是 PyTorch 2.1.0 + CUDA 11.8 + Miniconda 镜像
#   - 数据盘已挂载到 /root/autodl-tmp
# ============================================================================

set -e  # 遇到错误立即退出

echo "============================================"
echo " Photon2Perception — AutoDL 环境配置"
echo "============================================"
echo ""

# ---------- 1. 系统依赖 ----------
echo "[1/6] 安装系统依赖 (libraw, OpenGL)..."
apt-get update -qq
apt-get install -y -qq libraw-dev libgl1-mesa-glx libglib2.0-0
echo "  完成: libraw-dev, libgl1-mesa-glx, libglib2.0-0 已安装"
echo ""

# ---------- 2. Conda 环境 ----------
echo "[2/6] 创建 Conda 环境 (Python 3.10)..."
if conda env list | grep -q "photon2perception"; then
    echo "  环境 photon2perception 已存在，跳过创建"
else
    conda create -n photon2perception python=3.10 -y -q
    echo "  完成: conda 环境 photon2perception 已创建"
fi
echo ""

# ---------- 3. PyTorch (CUDA 版) ----------
echo "[3/6] 安装 PyTorch 2.1.0 + CUDA 11.8..."
# 激活环境并安装（用 conda run 避免 source 问题）
conda run -n photon2perception pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
echo "  完成: PyTorch 2.1.0 + CUDA 11.8"
echo ""

# ---------- 4. Python 依赖 ----------
echo "[4/6] 安装 Python 项目依赖..."
conda run -n photon2perception pip install --no-cache-dir \
    rawpy>=0.17.0 \
    numpy>=1.21.0 \
    pyyaml>=6.0 \
    matplotlib>=3.5.0 \
    scipy>=1.7.0 \
    fvcore>=0.1.5 \
    pytest>=7.0.0
echo "  完成: 所有 Python 依赖已安装"
echo ""

# ---------- 5. OpenMMLab（可选）----------
echo "[5/6] 安装 OpenMMLab (可选, 用于完整数据集和评估)..."
read -p "  是否安装 mmdet + mmseg? (y/n, 默认 n): " install_mm
if [ "$install_mm" = "y" ]; then
    conda run -n photon2perception pip install openmim
    conda run -n photon2perception mim install mmdet==3.3.0 mmsegmentation==1.2.0
    echo "  完成: mmdet 3.3.0 + mmseg 1.2.0 已安装"
else
    echo "  跳过 OpenMMLab 安装"
fi
echo ""

# ---------- 6. 创建数据目录 ----------
echo "[6/6] 创建数据与输出目录..."
mkdir -p /root/autodl-tmp/photon2perception/data
mkdir -p /root/autodl-tmp/photon2perception/outputs
mkdir -p /root/autodl-tmp/photon2perception/checkpoints
echo "  完成: 数据目录已创建"
echo ""

# ---------- 验证 ----------
echo "============================================"
echo " 验证安装"
echo "============================================"
conda run -n photon2perception python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_mem // 1024**3} GB)')
"
echo ""

# ---------- 运行测试 ----------
echo "============================================"
echo " 运行单元测试"
echo "============================================"
cd /root/autodl-tmp/photon2perception
conda run -n photon2perception python -m pytest tests/test_core.py -v --tb=short 2>&1 || true
echo ""

echo "============================================"
echo " 环境配置完成！"
echo ""
echo " 激活环境:  conda activate photon2perception"
echo " 项目目录:  /root/autodl-tmp/photon2perception"
echo " 数据目录:  /root/autodl-tmp/photon2perception/data"
echo " 输出目录:  /root/autodl-tmp/photon2perception/outputs"
echo ""
echo " 下一步:"
echo "  1. 下载数据集到 data/"
echo "  2. python tools/run_experiments.py list  # 查看实验"
echo "  3. python tools/run_experiments.py dry-run --exp-id E01  # 验证配置"
echo "============================================"
