"""
模型定义 - LSTM、Transformer、BERT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertForSequenceClassification, BertConfig


class LSTMSentiment(nn.Module):
    """Bi-LSTM情感分类模型"""

    def __init__(self, config):
        super(LSTMSentiment, self).__init__()

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            bidirectional=config['bidirectional']
        )

        lstm_output_dim = config['hidden_size'] * (2 if config['bidirectional'] else 1)

        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(lstm_output_dim, config['num_classes'])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSentiment(nn.Module):
    def __init__(self, config):
        super(TransformerSentiment, self).__init__()

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim'],
            padding_idx=0
        )

        self.positional_encoding = PositionalEncoding(
            config['embed_dim'],
            config['max_seq_len'],
            dropout=config['dropout']
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )

        self.layer_norm = nn.LayerNorm(config['embed_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config['embed_dim'], config['num_classes'])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).float()

        embedded = self.embedding(input_ids)
        embedded = embedded * math.sqrt(self.embedding.embedding_dim)
        embedded = self.positional_encoding(embedded)
        embedded = self.layer_norm(embedded)

        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=(attention_mask == 0)
        )

        pooled = transformer_out.mean(dim=1)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)

        return output


def get_bert_model(num_classes=2):
    """获取本地英文BERT模型"""
    from transformers import BertForSequenceClassification, BertConfig

    bert_local_path = "./bert-base-uncased"

    print(f"从本地加载英文BERT模型: {bert_local_path}")

    # 正确加载模型
    config = BertConfig.from_pretrained(bert_local_path)
    config.num_labels = num_classes

    model = BertForSequenceClassification.from_pretrained(
        bert_local_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 打印模型信息确认
    print(f"✅ BERT模型加载完成")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   分类头: {model.classifier}")

    return model


# 为了兼容config导入
from config import Config