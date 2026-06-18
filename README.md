# LSTM vs Transformer vs BERT 序列对比系统

## 项目简介

本项目实现了一个完整的LSTM、Transformer和BERT对比实验框架，用于情感分析任务。系统在RTX 4060 GPU上运行，包含混合精度训练、TensorBoard监控、超参数对比等功能。

## 技术要点

### 核心技术实现
1. **Bi-LSTM模型**: 2层双向LSTM，embedding维度300，隐藏层256
2. **Transformer模型**: 4层Transformer Encoder，8头注意力，前馈网络1024维
3. **BERT模型**: HuggingFace预训练bert-base-uncased微调
4. **混合精度训练**: torch.cuda.amp自动混合精度
5. **学习率调度**: Warm-up + Cosine Decay
6. **可视化监控**: TensorBoard全程监控

### 数据集
- IMDB电影评论情感分析（50000条）
- 二分类任务（正面/负面）

## 环境要求

### 硬件
- NVIDIA RTX 4060 (8GB VRAM) 或同等配置
- 至少16GB系统内存

### 软件
- Python 3.10+
- CUDA 12.x
- PyTorch 2.x

## 快速开始

### 1. 一键运行
```bash
chmod +x run.sh
./run.sh