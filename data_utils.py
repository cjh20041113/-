"""
数据处理工具 - 加载、预处理、数据增强
适配本地IMDB数据集 + 本地英文BERT模型
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from config import Config, set_seed

set_seed(Config.SEED)


def load_imdb_from_local():
    """
    从本地data目录加载IMDB数据集
    """
    from datasets import Dataset as HFDataset

    print("正在从本地加载IMDB数据集...")
    print(f"数据目录: {Config.DATA_DIR}")

    try:
        train_dataset = HFDataset.from_file(os.path.join(Config.DATA_DIR, "imdb-train.arrow"))
        test_dataset = HFDataset.from_file(os.path.join(Config.DATA_DIR, "imdb-test.arrow"))

        texts = list(train_dataset["text"]) + list(test_dataset["text"])
        labels = list(train_dataset["label"]) + list(test_dataset["label"])

        print(f"成功加载 {len(texts)} 条数据")
        print(f"训练集: {len(train_dataset)} 条")
        print(f"测试集: {len(test_dataset)} 条")

        return texts, labels

    except Exception as e:
        print(f"从arrow文件加载失败: {e}")
        print("警告: 无法加载本地数据集，使用演示数据")
        return create_demo_data()


def create_demo_data(num_samples=5000):
    """创建演示数据（备用方案）"""
    import random

    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful',
                      'fantastic', 'awesome', 'brilliant', 'perfect', 'love']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor',
                      'disappointing', 'boring', 'worst', 'hate', 'waste']

    texts = []
    labels = []

    for i in range(num_samples):
        label = random.randint(0, 1)

        if label == 1:
            words = positive_words[:random.randint(3, 8)]
            template = "This movie is {} and {}. I really {} it. The plot is {}."
        else:
            words = negative_words[:random.randint(3, 8)]
            template = "This movie is {} and {}. I really {} it. The plot is {}."

        if len(words) >= 2:
            text = template.format(words[0], words[1],
                                  'loved' if label == 1 else 'hated',
                                  words[2] if len(words) > 2 else words[0])
        else:
            text = f"This movie is {words[0]}."

        texts.append(text)
        labels.append(label)

    return texts, labels


def build_vocab(texts, max_size=20000):
    """构建词表"""
    from collections import Counter

    print("正在构建词表...")

    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)

    word_counts = Counter(all_words)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_size - 2):
        vocab[word] = len(vocab)

    print(f"词表大小: {len(vocab)}")
    return vocab


def text_to_indices(text, vocab, max_len):
    """将文本转换为索引序列"""
    words = text.lower().split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in words[:max_len]]

    if len(indices) < max_len:
        indices.extend([vocab['<PAD>']] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    return indices


class SentimentDataset(Dataset):
    """情感分析数据集类（用于LSTM和Transformer）"""

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        indices = text_to_indices(text, self.vocab, self.max_len)

        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTDataset(Dataset):
    """BERT数据集类（使用本地英文BERT）"""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_data_loaders():
    """
    获取数据加载器（包含LSTM、Transformer、BERT）
    使用本地英文BERT模型
    """
    print("\n" + "=" * 50)
    print("数据加载与预处理")
    print("=" * 50)

    # 加载数据
    texts, labels = load_imdb_from_local()

    print(f"加载完成: {len(texts)} 条样本")

    # 分析类别分布
    label_counts = Counter(labels)
    print(f"\n类别分布:")
    for label, count in sorted(label_counts.items()):
        label_name = "正面" if label == 1 else "负面"
        print(f"  {label_name}({label}): {count} ({count/len(labels)*100:.1f}%)")

    # 划分数据集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        train_size=Config.TRAIN_RATIO,
        random_state=Config.SEED,
        stratify=labels
    )

    val_ratio_in_temp = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=1 - val_ratio_in_temp,
        random_state=Config.SEED,
        stratify=temp_labels
    )

    print(f"\n最终数据集划分:")
    print(f"  训练集: {len(train_texts)} 条")
    print(f"  验证集: {len(val_texts)} 条")
    print(f"  测试集: {len(test_texts)} 条")

    # 训练集类别分布
    train_label_counts = Counter(train_labels)
    print(f"\n训练集类别分布:")
    for label, count in sorted(train_label_counts.items()):
        label_name = "正面" if label == 1 else "负面"
        print(f"  {label_name}: {count} ({count/len(train_labels)*100:.1f}%)")

    # 构建词表
    vocab = build_vocab(train_texts)

    Config.LSTM_CONFIG['vocab_size'] = len(vocab)
    Config.TRANSFORMER_CONFIG['vocab_size'] = len(vocab)

    # ========== LSTM和Transformer数据集 ==========
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, Config.MAX_SEQ_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, Config.MAX_SEQ_LEN)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, Config.MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # ========== BERT数据集（使用本地英文模型）==========
    print(f"\n正在加载本地英文BERT模型: {Config.BERT_MODEL_PATH}/")

    # 从本地文件夹加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_PATH)
    print(f"✅ BERT分词器加载完成，词表大小: {tokenizer.vocab_size}")

    bert_train_dataset = BERTDataset(train_texts, train_labels, tokenizer, Config.BERT_MAX_SEQ_LEN)
    bert_val_dataset = BERTDataset(val_texts, val_labels, tokenizer, Config.BERT_MAX_SEQ_LEN)
    bert_test_dataset = BERTDataset(test_texts, test_labels, tokenizer, Config.BERT_MAX_SEQ_LEN)

    bert_train_loader = DataLoader(
        bert_train_dataset,
        batch_size=Config.BERT_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    bert_val_loader = DataLoader(
        bert_val_dataset,
        batch_size=Config.BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    bert_test_loader = DataLoader(
        bert_test_dataset,
        batch_size=Config.BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    print(f"\n数据加载完成！")
    print(f"LSTM/Transformer 批大小: {Config.BATCH_SIZE}")
    print(f"BERT 批大小: {Config.BERT_BATCH_SIZE}")
    print(f"BERT模型路径: {Config.BERT_MODEL_PATH}")

    return {
        'lstm': (train_loader, val_loader, test_loader),
        'transformer': (train_loader, val_loader, test_loader),
        'bert': (bert_train_loader, bert_val_loader, bert_test_loader),
        'vocab': vocab,
        'tokenizer': tokenizer
    }


def get_timestamp():
    """获取时间戳"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")