# Photon2Perception — AutoDL 使用指南

本指南涵盖从零开始在 AutoDL (autodl.com) 上配置环境、运行实验、下载结果的完整流程。

---

## 目录

1. [AutoDL 是什么](#1-autodl-是什么)
2. [购买与配置实例](#2-购买与配置实例)
3. [上传代码](#3-上传代码)
4. [配置环境](#4-配置环境)
5. [下载数据集](#5-下载数据集)
6. [运行实验](#6-运行实验)
7. [管理实例与省钱技巧](#7-管理实例与省钱技巧)
8. [下载结果](#8-下载结果)
9. [常见问题](#9-常见问题)

---

## 1. AutoDL 是什么

AutoDL（[autodl.com](https://www.autodl.com)）是国内流行的 GPU 云计算平台，提供按小时/按天租赁的 NVIDIA GPU 实例。

**关键概念：**

| 概念 | 说明 |
|------|------|
| **实例（容器）** | 你的 GPU 运行环境，类似于一台装了 GPU 的 Linux 虚拟机 |
| **系统盘** | 实例的操作系统和软件盘，通常 50GB 免费 |
| **数据盘** | 持久化存储，关机后数据不丢失，按容量月付（~2 CNY/10GB/月） |
| **关机 vs 销毁** | 关机 = 停 GPU 计费，数据盘中数据保留；销毁 = 删除实例，数据盘数据也会丢失 |

---

## 2. 购买与配置实例

### 2.1 选择 GPU

推荐从 **RTX 3090 (24GB)** 开始——性价比最高，24GB 显存足够跑 ViT-Base 模型训练。

| GPU | 显存 | 参考价格 | 适合场景 |
|-----|------|---------|---------|
| **RTX 3090** | 24 GB | ~2.5 CNY/小时 | **推荐主力** |
| RTX 4090 | 24 GB | ~4 CNY/小时 | 需要更快训练时 |
| A6000 | 48 GB | ~6 CNY/小时 | 大 batch、高分辨率、DDP 多卡 |
| A100 | 40/80 GB | ~10-15 CNY/小时 | 极致大批量实验 |

> 本项目的 *全部 36 个实验* 在单张 RTX 3090 上预计耗时 **~55 小时**，费用约 **140 CNY**。

### 2.2 创建实例步骤

1. 登录 [AutoDL 官网](https://www.autodl.com)，完成实名认证和充值
2. 进入「容器实例」→「租用新实例」
3. **选择 GPU**：筛选「RTX 3090」，选择有空闲实例的地区
4. **选择基础镜像**：
   - 在「社区镜像」或「官方镜像」中选择 **PyTorch 2.1.0 + CUDA 11.8 + Miniconda**
   - 如果没有精确匹配，选择 PyTorch ≥ 2.0 + CUDA ≥ 11.8 的镜像也可以
5. **配置数据盘**：
   - 新建数据盘，建议 **100 GB**（COCO ~25GB + Cityscapes ~12GB + checkpoints ~30GB + 余量）
   - 或者如果已有数据盘，直接挂载
6. 点击「立即创建」，等待实例启动（通常 1-3 分钟）

### 2.3 连接实例

实例启动后，有以下几种连接方式：

- **Jupyter Lab**（推荐）：浏览器中打开，有终端、文件管理、代码编辑器
- **SSH**：`ssh -p <端口> root@<IP地址>`（在实例详情页查看连接信息）
- **网页终端**：AutoDL 控制台的「终端」按钮

> **推荐首次配置用 Jupyter Lab**，界面友好，复制粘贴方便。

---

## 3. 上传代码

### 方案 A：Git 拉取（推荐）

这是最高效的方式，支持增量和版本管理：

```bash
# 在 AutoDL 终端中执行
cd /root/autodl-tmp

# 从 GitHub 或 Gitee 克隆（Gitee 国内下载更快）
git clone https://github.com/<你的用户名>/photon2perception.git
# 或 Gitee:
# git clone https://gitee.com/<你的用户名>/photon2perception.git

cd photon2perception
```

**本地推送代码到仓库：**

```powershell
# 在本地 Windows PowerShell 中
cd e:\2026Fall\photon2perception
git init
git add .
git commit -m "Initial commit: full PyTorch implementation"
git remote add origin https://github.com/<你的用户名>/photon2perception.git
git push -u origin main
```

> 如果 GitHub 太慢，推荐使用 **Gitee**（码云），国内访问很快。在 gitee.com 创建私有仓库，设置方法同 GitHub。

### 方案 B：直接上传

AutoDL Jupyter Lab 左侧有文件管理器，可以拖拽上传文件。适合小文件和临时修改。整个项目约 300KB（不包含数据和 checkpoints），上传很快。

---

## 4. 配置环境

### 一键配置（推荐）

```bash
cd /root/autodl-tmp/photon2perception
bash scripts/setup_autodl.sh
```

这个脚本会自动完成：
1. 安装系统依赖（libraw, OpenGL）
2. 创建 conda 环境 (Python 3.10)
3. 安装 PyTorch 2.1.0 + CUDA 11.8
4. 安装所有 Python 依赖
5. 询问是否安装 OpenMMLab (mmdet + mmseg)
6. 创建数据/输出目录
7. 验证安装并运行单元测试

### 手动配置（如果脚本出问题）

```bash
# 系统依赖
apt-get update && apt-get install -y libraw-dev libgl1-mesa-glx libglib2.0-0

# Conda 环境
conda create -n photon2perception python=3.10 -y
conda activate photon2perception

# PyTorch
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Python 依赖
pip install rawpy numpy pyyaml matplotlib scipy fvcore pytest

# 验证
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest tests/test_core.py -v
```

---

## 5. 下载数据集

### 最小可行数据集：COCO 2017

使用 COCO 2017 训练集 + 本项目自带的 RGB→Bayer unprocessing pipeline 生成合成 RAW 数据。

```bash
cd /root/autodl-tmp/photon2perception/data

# 下载 COCO 2017 训练集（~19 GB）
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip
unzip annotations_trainval2017.zip
```

### 真实 RAW 数据集

| 数据集 | 用途 | 下载链接 | 大小 |
|--------|------|---------|------|
| PASCAL RAW | 目标检测 | [官方链接](https://cedar.buffalo.edu/~srihari/pascalraw/) | ~25 GB |
| LOD | 低光检测 | 联系作者或从论文主页获取 | ~10 GB |
| Cityscapes | 语义分割 | [官网](https://www.cityscapes-dataset.com/)（需注册） | ~12 GB |
| ADE20K | 语义分割 | [官网](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | ~5 GB |

> 优先下载 COCO —— 它可以用 unprocessing pipeline 转成合成 RAW，不需要 CCM/WB 等相机参数。

---

## 6. 运行实验

### 6.1 验证配置（dry-run）

在正式训练之前，先 dry-run 确认配置正确：

```bash
conda activate photon2perception

# 查看所有实验
python tools/run_experiments.py list

# 按分组查看
python tools/run_experiments.py list --group ablation

# Dry-run 单个实验
python tools/run_experiments.py dry-run --exp-id E01
```

### 6.2 启动训练

```bash
# 方法 1: 跑单个实验
python tools/run_experiments.py run --exp-id E11a --gpus 0

# 方法 2: 跑整个分组
python tools/run_experiments.py run --group ablation --gpus 0

# 方法 3: 使用批量脚本（推荐）
bash scripts/run_experiment_batch.sh batch1
```

### 6.3 实验分批策略

```
第 1 批（batch1）→ 快速验证 pipeline，~2 小时
  目的：确认训练能正常运行，产生合理的 loss 曲线
  实验：E11a-E11c（输入消融）, E12a-E12d（位置编码消融）

第 2 批（batch2）→ 主实验，~25 小时
  目的：跑完所有核心对比实验和消融
  实验：E01-E09（主对比）, E13a-E15b（路由/CFA/元数据消融）

第 3 批（batch3）→ 鲁棒性，~20 小时
  实验：E16-E25（退化鲁棒性 + 跨域泛化）

第 4 批（batch4）→ 效率分析，~5 小时
  实验：E26-E31（推理基准测试）

第 5 批（batch5）→ 定性 + Pareto，~3 小时
  实验：E32-E36（可视化）, E10（Pareto 前沿）
```

### 6.4 监控训练状态

```bash
# 查看日志
tail -f outputs/logs/E01_*.log

# 查看 GPU 使用情况
nvidia-smi

# 持续监控 GPU（每 2 秒刷新）
watch -n 2 nvidia-smi
```

---

## 7. 管理实例与省钱技巧

### 7.1 关机 vs 销毁

| 操作 | GPU 计费 | 数据盘 | 适用场景 |
|------|---------|--------|---------|
| **关机** | 停止 | 保留 | 晚上不训练、周末休息时 |
| **销毁（无数据盘）** | 停止 | **丢失！** | 用完即弃的测试实例 |
| **销毁（有数据盘）** | 停止 | 保留 | 换实例类型时，数据盘可重新挂载 |

> **黄金法则：不训练时关机，数据盘保留。** 关机后只收数据盘存储费（~2 CNY/10GB/月）。

### 7.2 省钱技巧

1. **晚上关机**：假如一天训练 12 小时，关机 12 小时，GPU 费省一半
2. **选对时间**：凌晨和周末有时有空闲折扣
3. **先用小数据验证**：第 1 批实验用 COCO 的 10% 子集先跑通，确认没问题再跑全量
4. **使用混合精度**：在 config 中设 `mixed_precision: true`，减少 ~30% 显存和时间
5. **批量运行**：用 `run_experiment_batch.sh` 自动排队，避免手动操作耽误的 GPU 空闲时间

### 7.3 实例续租提醒

AutoDL 实例按租期（小时/天/周）计费。建议：
- 首次设置：租 2-4 小时，配置环境 + 下载数据
- 跑实验：按天租，预估当天能跑完的实验量

---

## 8. 下载结果

### 方法 1：Jupyter Lab 文件管理

AutoDL Jupyter Lab 左侧文件面板 → 右键 `outputs/` 目录 → 「下载为压缩包」→ 浏览器下载到本地。

### 方法 2：scp 命令行

```powershell
# 在本地 Windows PowerShell 中
# AutoDL 实例详情页可以找到 SSH 连接信息：IP 地址和端口
scp -r -P <端口号> root@<IP地址>:/root/autodl-tmp/photon2perception/outputs/ .\outputs\
```

### 方法 3：AutoDL 网盘

AutoDL 提供网盘功能，可以直接在实例间共享文件。

### 定期备份

建议每跑完一批实验就把结果下载到本地：

```bash
# 在本地
scp -r -P <端口> root@<IP>:/root/autodl-tmp/photon2perception/outputs/ ./outputs_backup_$(date +%Y%m%d)/
```

---

## 9. 常见问题

### Q: conda 创建环境很慢？

```
# 改用 mamba（更快的 conda 替代品）
conda install mamba -n base -c conda-forge
mamba create -n photon2perception python=3.10 -y
```

### Q: pip install 很慢？

```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

### Q: CUDA out of memory？

1. 减小 batch size：编辑 config 中的 `data.batch_size`
2. 启用混合精度：`mixed_precision: true`
3. 减小输入尺寸：`img_size: [384, 384]`
4. 或者换更大的 GPU（如 A6000 48GB）

### Q: rawpy 导入失败？

```
apt-get install -y libraw-dev
pip install rawpy --no-cache-dir
```

### Q: 如何用多张 GPU？

```bash
# DDP 多卡训练
torchrun --nproc_per_node=<GPU数量> tools/train.py --config <config>

# run_experiments.py 也支持多 GPU
python tools/run_experiments.py run --gpus 0,1,2,3 --group main
```

### Q: 实验结果怎么看？

```bash
# 实验输出目录结构
outputs/
├── E01/
│   ├── config.yaml          # 该实验的实际配置
│   ├── stdout.log           # 训练日志
│   ├── stderr.log           # 错误日志
│   └── checkpoint_epoch_*.pth  # 模型权重
├── logs/
│   └── batch_results_*.txt  # 批量实验结果汇总
└── experiment_matrix.csv    # 实验矩阵 CSV
```

---

## 快速参考卡片

```
# 登录 AutoDL，创建 RTX 3090 实例

# ====== 一次性设置 ======
cd /root/autodl-tmp
git clone <your-repo-url> photon2perception
cd photon2perception
bash scripts/setup_autodl.sh
# 下载 COCO 到 data/

# ====== 每次训练前 ======
conda activate photon2perception
cd /root/autodl-tmp/photon2perception
git pull  # 拉取最新代码

# ====== 启动实验 ======
bash scripts/run_experiment_batch.sh batch1

# ====== 监控 ======
nvidia-smi
tail -f outputs/logs/*.log

# ====== 结束后 ======
# 关机实例（在 AutoDL 网页控制台操作）
# 下载结果到本地
```
