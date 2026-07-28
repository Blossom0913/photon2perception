
## Introduction（四段结构）

**第一段：问题背景与重要性。**  
Camera RAW 保留了传感器在成像链路最前端的线性物理信息，在弱光、高动态范围、运动模糊和局部遮挡等场景下，往往比经过 ISP 后的 sRGB 图像保留更多可用于感知的细节。已有研究已经反复表明，低光感知任务中 RAW 的高 bit-depth 与物理一致性具有明显优势，因此将感知直接建立在 RAW 域上，而不是先将其压缩为 RGB 再进行识别，是一个具有明确物理意义和实际价值的方向。

**第二段：现有工作的局限。**  
然而，现有 RAW 感知方法大多沿着两条路线发展：一类是以 learnable ISP 或轻量适配模块为中心，试图把 RAW 调整到预训练 RGB backbone 更易处理的形式；另一类是针对单一任务构建 RAW 特定模型，但通常仍然保持稠密计算范式，并没有从输入结构、token 组织和资源分配机制上重构感知过程。换言之，当前方法虽然证明了 RAW 的价值，但还没有把 RAW 的二维传感器结构和高层 perception 任务之间的关系真正“做成体系”。

**第三段：本文方法。**  
基于这一观察，我们提出一种面向 RAW 感知的结构保持型框架：直接在 Bayer RAW 上进行 token 化，显式保留 CFA 的二维空间结构，并在此基础上引入 2D 位置编码建模 token 间的空间关系；同时加入可选的方向性增强，以进一步适配 RAW 网格中的局部模式；最后，通过 saliency-aware 或 uncertainty-aware 的稀疏路由机制，把计算集中到更有信息量的区域。该设计的核心不是单纯追求“更少字节”，而是在不依赖完整 ISP 的前提下，降低推理时延与内存/计算开销，同时维持甚至提升感知质量。2D RoPE 已经在视觉 Transformer 中被证明适合二维图像 token，而我们将其进一步放入 RAW 传感器结构这一更强先验的场景中。

**第四段：贡献与实验预告。**  
因此，本文的目标不是构建一个仅对单一任务有效的 RAW 模型，而是提出一个可迁移、可扩展、并且具有物理解释性的 RAW-to-perception 框架。我们将从多任务感知、低光与退化鲁棒性、以及效率—性能权衡三个维度进行系统评估，验证该框架在 detection、segmentation 等高层任务中的通用性，并考察其在不同噪声、模糊、曝光和跨相机条件下的稳健性。最终，我们希望证明：RAW 的价值不只在于“保真成像”，更在于为高层感知提供一种更高信息密度、更适合稀疏计算的输入表征。

## Contributions（三条）

1. **我们提出了一个面向 RAW 感知的结构保持型框架。**  
    该框架直接处理 Bayer RAW token，显式保留二维 CFA 结构，而不是依赖完整 ISP 先将输入转为 RGB。它将 RAW 的物理信息与高层感知任务直接连接起来。
    
2. **我们设计了一个 2D 位置编码驱动的 RAW token 表示，并引入可选的方向性增强。**  
    该设计以 2D RoPE 为主干，使模型更好地利用图像的空间组织；方向性模块作为可选增强，用于刻画 RAW 网格中的局部方向模式，但不把它设为方法成败的唯一支柱。
    
3. **我们提出了一种物理驱动的稀疏路由机制，以提升效率并保持鲁棒性。**  
    与一般 token pruning 不同，该路由以 saliency、uncertainty 和 RAW 物理退化为依据，目标是在降低 latency、memory bandwidth 和 compute 的同时，维持多任务 perception 性能。
    

## Experiments（小节提纲）

### 4.1 Experimental Setup

- 数据集与任务设置：以 RAW 相关高层感知任务为主，优先选择检测作为主任务，再补充一个 dense task（例如 segmentation 或 instance segmentation），以及一个局部特征任务或迁移验证任务。
    
- 输入形式：RAW Bayer、demosaic RAW、RGB 三种输入统一比较。
    
- 评估指标：任务指标（AP、mIoU 等）+ 效率指标（latency、FLOPs、显存占用、memory bandwidth、输入字节量）。
    

### 4.2 Main Results

- 与 RGB pipeline、demosaic RAW pipeline、learnable ISP/adapter 类方法、以及直接 RAW 感知方法进行比较。
    
- 报告在正常光、低光、模糊、遮挡、曝光变化和跨相机条件下的结果。
    
- 强调性能—效率 Pareto 前沿，而不是只报单点 SOTA。
    

### 4.3 Ablation Studies

- **输入表示消融**：RAW tokenization vs demosaic vs RGB。
    
- **位置编码消融**：无位置编码 / 2D absolute PE / 2D RoPE / 2D RoPE + directional enhancement。
    
- **稀疏路由消融**：dense attention / heuristic pruning / saliency-aware routing / uncertainty-aware routing。
    
- **CFA-aware 设计消融**：是否显式建模 Bayer phase、是否共享相位嵌入、是否对不同相位单独编码。
    
- **metadata 消融**：是否加入 ISO、曝光、black level、white balance 等 sensor metadata。
    

### 4.4 Robustness and Generalization

- 低光、噪声、模糊、过曝、天气退化下的稳健性测试。
    
- 跨相机、跨 ISO、跨曝光、跨分辨率泛化测试。
    
- OOD benchmark 上的稳定性分析。
    

### 4.5 Efficiency Analysis

- 统计不同输入和不同模块配置下的端到端 latency。
    
- 区分 sensor-to-processor bandwidth、on-chip memory bandwidth 和 compute cost。
    
- 分析稀疏路由带来的真实收益，并给出 token 数量、激活区域比例与性能之间的关系。
    

### 4.6 Qualitative Analysis

- 可视化 token 路由位置、注意力分布和退化场景下的选择性聚焦行为。
    
- 展示 RAW-native 表示在暗区、高噪区域、边缘细节上的优势。