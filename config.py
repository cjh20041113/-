"""
配置文件 - LSTM vs Transformer对比实验
"""

import os
import torch


class Config:
    """全局配置"""

    # ========== 基础配置 ==========
    SEED = 42  # 随机种子，保证可复现性
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 数据配置 ==========
    DATASET_NAME = 'imdb'  # 使用IMDB数据集（英文）
    MAX_SEQ_LEN = 256  # 最大序列长度
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Windows下设为0

    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ========== LSTM模型配置 ==========
    LSTM_CONFIG = {
        'vocab_size': None,  # 动态设置
        'embed_dim': 300,
        'hidden_size': 256,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.5,
        'bidirectional': True
    }

    # ========== Transformer模型配置 ==========
    TRANSFORMER_CONFIG = {
        'vocab_size': None,  # 动态设置
        'embed_dim': 256,
        'num_heads': 4,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3,
        'dim_feedforward': 512,
        'max_seq_len': MAX_SEQ_LEN
    }

    # ========== BERT模型配置 ==========
    # 使用本地英文BERT模型
    BERT_MODEL_PATH = './bert-base-uncased'  # 你的英文BERT路径
    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_MAX_SEQ_LEN = 128
    BERT_BATCH_SIZE = 16
    BERT_LEARNING_RATE = 2e-5

    # ========== 训练配置 ==========
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 学习率调度配置
    WARMUP_EPOCHS = 2

    # 早停配置
    EARLY_STOPPING_PATIENCE = 3

    # ========== 混合精度训练配置 ==========
    USE_AMP = True  # 是否使用混合精度训练

    # ========== 路径配置 ==========
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    FIGURE_DIR = os.path.join(BASE_DIR, 'figures')

    # 创建必要的目录
    for dir_path in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("实验配置信息")
        print("=" * 50)
        print(f"设备: {cls.DEVICE}")
        print(f"数据集: {cls.DATASET_NAME}")
        print(f"数据目录: {cls.DATA_DIR}")
        print(f"BERT模型路径: {cls.BERT_MODEL_PATH}")
        print(f"批大小: {cls.BATCH_SIZE}")
        print(f"最大序列长度: {cls.MAX_SEQ_LEN}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"使用混合精度: {cls.USE_AMP}")
        print("=" * 50)


def set_seed(seed: int):
    """设置所有随机种子，保证可复现性"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")