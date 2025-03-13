# GPT-2 (124M) 在 PyTorch 中的重现

本仓库致力于重现 Andrej Karpathy 在其 YouTube 视频 ["让我们重现 GPT-2 (124M)"](https://www.youtube.com/watch?v=MutbZKX2jAE) 中展示的 GPT-2 (124M) 模型。与 OpenAI 原始使用 TensorFlow 实现的 GPT-2 不同，本项目采用 PyTorch，以更简洁和现代的方式从头构建和训练模型。

本项目的目标是记录我逐步重现 GPT-2 (124M) 模型的完整经历，包括遇到的挑战、获得的洞见和解决方案。所有的实验、代码和笔记都通过 Jupyter 笔记本共享，以确保透明度和教育价值。

## 项目动机

Andrej Karpathy 的视频为从零开始构建 GPT-2 (124M) 提供了一个全面的指南。受他的方法启发，我的目标是：
- 深入理解 GPT-2 架构的内部工作原理。
- 将实现从 TensorFlow 适配到 PyTorch。
- 记录整个过程，以便他人可以从我的经历中学习。

## 仓库内容

- **`notebooks/`**: 包含记录构建和训练 GPT-2 (124M) 模型逐步过程的 Jupyter 笔记本。
- **`src/`**: 模型架构、训练循环和实用工具（例如分词、数据加载）的源代码。
- **`data/`**: 下载和准备数据集（例如 FineWeb 或 OpenWebText）的脚本或说明。
- **`config/`**: 模型超参数和训练设置的配置文件。
- **`results/`**: 训练运行的日志、检查点和生成输出。
- `README.md`: 本文件，提供概览和使用说明。

## 开始使用

### 前提条件
- Python 3.8+
- PyTorch 2.0+（支持 CUDA 的 GPU 训练）
- 其他依赖：`numpy`, `transformers`, `datasets`, `tiktoken`, `tqdm`

## 进展

以下是项目的当前状态，以清单形式跟踪：

- [ ] **模型架构**: 在 PyTorch 中实现 GPT-2 (124M) 架构。
- [ ] **数据准备**: 对数据集（例如 FineWeb 或 OpenWebText）进行分词和预处理。
- [ ] **训练设置**: 使用 PyTorch 设置训练循环，包括损失函数和优化器。
- [ ] **训练与评估**: 训练模型并使用基准测试（如 HellaSwag）评估性能。
- [ ] **文本生成**: 生成样本文本并分析输出质量。

每个里程碑都将在 Jupyter 笔记本中详细记录。

## 仓库结构

```plaintext
仓库结构如下：
GPT-2-Reproduce/
│
├── notebooks/                  # 记录过程的 Jupyter 笔记本
│   ├── 01_data_preprocessing.ipynb    # 数据集准备步骤
│   ├── 02_model_implementation.ipynb  # 模型架构实现
│   ├── 03_training.ipynb              # 训练循环和实验
│   └── 04_evaluation_and_generation.ipynb  # 评估和文本生成
│
├── src/                        # 核心源代码
│   ├── model.py                # GPT-2 模型定义
│   ├── train.py                # 包含主循环的训练脚本
│   ├── utils.py                # 实用函数（例如数据加载、日志记录）
│   └── generate.py             # 文本生成脚本
│
├── data/                       # 数据处理
│   ├── prepare.py              # 下载和预处理数据的脚本
│   └── README.md               # 数据集设置说明
│
├── config/                     # 配置文件
│   └── train_gpt2.py           # 训练超参数
│
├── results/                    # 输出目录
│   ├── logs/                   # 训练日志（例如损失曲线）
│   ├── checkpoints/            # 保存的模型权重
│   └── samples/                # 生成的文本样本
│
├── requirements.txt            # Python 依赖列表
├── README.md                   # 项目概览（本文件）
└── LICENSE                     # MIT 许可证文件
```

此结构将代码、文档和输出分开，以确保清晰且易于使用。
