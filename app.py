"""
情感分析Web应用
支持LSTM、Transformer、BERT三种模型的实时预测
"""

import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
import json
import math
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ========== 配置 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 模型路径
MODEL_DIR = "./checkpoints"
BERT_MODEL_PATH = "./bert-base-uncased"
BERT_FINETUNED_PATH = "./checkpoints/bert"  # 微调后的BERT模型路径

# ========== 定义模型结构 ==========
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=300, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.5, bidirectional=True):
        super(LSTMSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
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
    def __init__(self, vocab_size=20000, embed_dim=256, num_heads=4, num_layers=2,
                 num_classes=2, dropout=0.3, dim_feedforward=512, max_seq_len=256):
        super(TransformerSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).float()
        embedded = self.embedding(input_ids)
        embedded = embedded * math.sqrt(self.embedding.embedding_dim)
        embedded = self.positional_encoding(embedded)
        embedded = self.layer_norm(embedded)
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=(attention_mask == 0))
        pooled = transformer_out.mean(dim=1)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)
        return output


# ========== 全局模型变量 ==========
models = {}
tokenizer = None
vocab_size = 20000


def load_models():
    """加载所有训练好的模型"""
    global models, tokenizer

    print("正在加载模型...")

    # 加载BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)

    # 加载LSTM模型
    try:
        lstm_model = LSTMSentiment(vocab_size=vocab_size).to(DEVICE)
        checkpoint_path = os.path.join(MODEL_DIR, "lstm", "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
            lstm_model.eval()
            models['lstm'] = lstm_model
            print("✅ LSTM模型加载成功")
        else:
            print("❌ LSTM模型文件不存在")
    except Exception as e:
        print(f"❌ LSTM模型加载失败: {e}")

    # 加载Transformer模型
    try:
        transformer_model = TransformerSentiment(vocab_size=vocab_size).to(DEVICE)
        checkpoint_path = os.path.join(MODEL_DIR, "transformer", "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            transformer_model.eval()
            models['transformer'] = transformer_model
            print("✅ Transformer模型加载成功")
        else:
            print("❌ Transformer模型文件不存在")
    except Exception as e:
        print(f"❌ Transformer模型加载失败: {e}")

    # 加载微调后的BERT模型
    try:
        # 微调后的模型保存在 checkpoints/bert/best_model.pth
        # 这是一个完整的checkpoint文件，不是HuggingFace格式
        bert_checkpoint_path = os.path.join(BERT_FINETUNED_PATH, "best_model.pth")

        if os.path.exists(bert_checkpoint_path):
            print(f"找到微调BERT模型: {bert_checkpoint_path}")

            # 首先加载基础BERT模型
            bert_model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL_PATH,
                num_labels=2,
                ignore_mismatched_sizes=True
            ).to(DEVICE)

            # 加载微调后的权重
            checkpoint = torch.load(bert_checkpoint_path, map_location=DEVICE, weights_only=False)

            # 检查checkpoint的结构
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 加载权重
            bert_model.load_state_dict(state_dict, strict=False)
            bert_model.eval()
            models['bert'] = bert_model
            print("✅ BERT微调模型加载成功")
            print(f"   最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
        else:
            print(f"❌ BERT微调模型文件不存在: {bert_checkpoint_path}")
            # 尝试加载基础模型
            bert_model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL_PATH, num_labels=2
            ).to(DEVICE)
            bert_model.eval()
            models['bert'] = bert_model
            print("✅ BERT基础模型加载成功（未微调）")
    except Exception as e:
        print(f"❌ BERT模型加载失败: {e}")
        import traceback
        traceback.print_exc()


def text_to_indices(text, max_len=256):
    """将文本转换为索引（用于LSTM和Transformer）"""
    words = text.lower().split()
    indices = []
    for word in words[:max_len]:
        # 简单哈希映射
        idx = hash(word) % (vocab_size - 2) + 2
        indices.append(idx)

    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    return torch.tensor([indices], dtype=torch.long).to(DEVICE)


def predict_with_lstm(text):
    """使用LSTM模型预测"""
    if 'lstm' not in models:
        return None, None
    with torch.no_grad():
        input_ids = text_to_indices(text)
        outputs = models['lstm'](input_ids)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence


def predict_with_transformer(text):
    """使用Transformer模型预测"""
    if 'transformer' not in models:
        return None, None
    with torch.no_grad():
        input_ids = text_to_indices(text)
        outputs = models['transformer'](input_ids)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence


def predict_with_bert(text):
    """使用BERT模型预测"""
    if 'bert' not in models or tokenizer is None:
        return None, None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = models['bert'](**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence


def predict_all(text):
    """使用所有模型进行预测"""
    results = {}

    pred, conf = predict_with_lstm(text)
    if pred is not None:
        results['lstm'] = {'prediction': pred, 'confidence': conf,
                          'sentiment': '正面' if pred == 1 else '负面'}

    pred, conf = predict_with_transformer(text)
    if pred is not None:
        results['transformer'] = {'prediction': pred, 'confidence': conf,
                                  'sentiment': '正面' if pred == 1 else '负面'}

    pred, conf = predict_with_bert(text)
    if pred is not None:
        results['bert'] = {'prediction': pred, 'confidence': conf,
                          'sentiment': '正面' if pred == 1 else '负面'}

    return results


# ========== 内嵌HTML ==========
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>情感分析系统 - LSTM vs Transformer vs BERT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .main-content { display: flex; gap: 30px; flex-wrap: wrap; }
        .input-section {
            flex: 1; min-width: 350px; background: white; border-radius: 20px;
            padding: 25px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .input-section h2 { color: #333; margin-bottom: 20px; border-left: 4px solid #667eea; padding-left: 15px; }
        textarea {
            width: 100%; height: 200px; padding: 15px; font-size: 16px;
            border: 2px solid #e0e0e0; border-radius: 12px; resize: vertical;
            font-family: inherit; transition: border-color 0.3s;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        .example-buttons { margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap; }
        .example-btn {
            background: #f0f0f0; border: none; padding: 8px 16px; border-radius: 20px;
            cursor: pointer; font-size: 14px; transition: all 0.3s;
        }
        .example-btn:hover { background: #667eea; color: white; }
        .predict-btn {
            width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 14px; font-size: 18px;
            font-weight: bold; border-radius: 12px; cursor: pointer; margin-top: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .predict-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102,126,234,0.4); }
        .results-section { flex: 2; min-width: 500px; }
        .model-card {
            background: white; border-radius: 20px; padding: 20px; margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2); transition: transform 0.3s;
        }
        .model-card:hover { transform: translateY(-5px); }
        .model-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0;
        }
        .model-name.lstm { color: #2E86AB; font-size: 1.4rem; font-weight: bold; }
        .model-name.transformer { color: #A23B72; font-size: 1.4rem; font-weight: bold; }
        .model-name.bert { color: #F18F01; font-size: 1.4rem; font-weight: bold; }
        .model-accuracy { background: #f0f0f0; padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; color: #666; }
        .result-content { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; }
        .sentiment-badge { font-size: 1.5rem; font-weight: bold; padding: 10px 25px; border-radius: 50px; }
        .sentiment-badge.positive { background: #d4edda; color: #155724; }
        .sentiment-badge.negative { background: #f8d7da; color: #721c24; }
        .confidence { text-align: right; }
        .confidence-value { font-size: 1.8rem; font-weight: bold; color: #333; }
        .confidence-label { font-size: 0.85rem; color: #666; }
        .progress-bar { width: 200px; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .progress-fill.positive { background: #28a745; width: 0%; height: 100%; transition: width 0.5s ease; }
        .progress-fill.negative { background: #dc3545; width: 0%; height: 100%; transition: width 0.5s ease; }
        .loading { text-align: center; padding: 40px; color: white; }
        .spinner {
            border: 3px solid #f3f3f3; border-top: 3px solid #667eea;
            border-radius: 50%; width: 40px; height: 40px;
            animation: spin 1s linear infinite; margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .footer { text-align: center; color: white; margin-top: 30px; padding: 20px; opacity: 0.8; }
        @media (max-width: 900px) { .main-content { flex-direction: column; } .results-section { min-width: auto; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 情感分析系统</h1>
            <p>LSTM vs Transformer vs BERT - 三模型实时对比</p>
        </div>
        <div class="main-content">
            <div class="input-section">
                <h2>📝 输入文本</h2>
                <textarea id="inputText" placeholder="请输入要分析的英文评论...&#10;&#10;例如：&#10;This movie is absolutely fantastic! I love it!&#10;The acting was terrible and the plot was boring."></textarea>
                <div class="example-buttons">
                    <button class="example-btn" onclick="setExample('positive')">👍 正面示例</button>
                    <button class="example-btn" onclick="setExample('negative')">👎 负面示例</button>
                    <button class="example-btn" onclick="setExample('neutral')">🤔 中性示例</button>
                </div>
                <button class="predict-btn" onclick="predict()">🔍 开始分析</button>
            </div>
            <div class="results-section" id="resultsSection">
                <div class="loading" id="loading" style="display: none;">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px;">分析中...</p>
                </div>
                <div class="model-card" id="lstmCard" style="display: none;">
                    <div class="model-header"><span class="model-name lstm">🧠 LSTM</span><span class="model-accuracy">准确率: 87.29%</span></div>
                    <div class="result-content">
                        <div class="sentiment-badge" id="lstmSentiment">-</div>
                        <div class="confidence">
                            <div class="confidence-value" id="lstmConfidence">-</div>
                            <div class="confidence-label">置信度</div>
                            <div class="progress-bar"><div class="progress-fill" id="lstmProgress"></div></div>
                        </div>
                    </div>
                </div>
                <div class="model-card" id="transformerCard" style="display: none;">
                    <div class="model-header"><span class="model-name transformer">⚡ Transformer</span><span class="model-accuracy">准确率: 85.15%</span></div>
                    <div class="result-content">
                        <div class="sentiment-badge" id="transformerSentiment">-</div>
                        <div class="confidence">
                            <div class="confidence-value" id="transformerConfidence">-</div>
                            <div class="confidence-label">置信度</div>
                            <div class="progress-bar"><div class="progress-fill" id="transformerProgress"></div></div>
                        </div>
                    </div>
                </div>
                <div class="model-card" id="bertCard" style="display: none;">
                    <div class="model-header"><span class="model-name bert">🤗 BERT (微调)</span><span class="model-accuracy">准确率: 87.63%</span></div>
                    <div class="result-content">
                        <div class="sentiment-badge" id="bertSentiment">-</div>
                        <div class="confidence">
                            <div class="confidence-value" id="bertConfidence">-</div>
                            <div class="confidence-label">置信度</div>
                            <div class="progress-bar"><div class="progress-fill" id="bertProgress"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="footer"><p>基于 IMDB 数据集训练 | RTX 4060 GPU 加速 | 深度学习期末综合实践项目</p></div>
    </div>
    <script>
        const examples = {
            positive: "This movie is absolutely fantastic! The acting was superb and the story kept me on the edge of my seat. I would definitely recommend it to everyone!",
            negative: "This movie was a complete disaster. The acting was terrible, the plot made no sense, and I regret wasting my time and money on it.",
            neutral: "This movie is okay. It has some good parts and some bad parts. I'm not sure if I would recommend it or not."
        };
        function setExample(type) { document.getElementById('inputText').value = examples[type]; predict(); }
        async function predict() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) { alert('请输入文本内容'); return; }
            document.getElementById('loading').style.display = 'block';
            document.getElementById('lstmCard').style.display = 'none';
            document.getElementById('transformerCard').style.display = 'none';
            document.getElementById('bertCard').style.display = 'none';
            try {
                const response = await fetch('/predict', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const results = await response.json();
                document.getElementById('loading').style.display = 'none';
                if (results.lstm) updateResultCard('lstm', results.lstm);
                if (results.transformer) updateResultCard('transformer', results.transformer);
                if (results.bert) updateResultCard('bert', results.bert);
            } catch (error) { console.error('预测失败:', error); document.getElementById('loading').style.display = 'none'; alert('预测失败'); }
        }
        function updateResultCard(model, result) {
            const sentimentEl = document.getElementById(`${model}Sentiment`);
            const confidenceEl = document.getElementById(`${model}Confidence`);
            const progressEl = document.getElementById(`${model}Progress`);
            const card = document.getElementById(`${model}Card`);
            const isPositive = result.sentiment === '正面';
            const confidencePercent = (result.confidence * 100).toFixed(1);
            sentimentEl.textContent = result.sentiment;
            sentimentEl.className = `sentiment-badge ${isPositive ? 'positive' : 'negative'}`;
            confidenceEl.textContent = `${confidencePercent}%`;
            progressEl.style.width = `${confidencePercent}%`;
            progressEl.className = `progress-fill ${isPositive ? 'positive' : 'negative'}`;
            card.style.display = 'block';
        }
        window.onload = () => { document.getElementById('inputText').value = examples.positive; predict(); };
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """首页 - 返回内嵌HTML"""
    return HTML_PAGE


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请输入文本'})
    results = predict_all(text)
    return jsonify(results)


@app.route('/models_info')
def models_info():
    """获取模型信息"""
    info = {
        'available_models': list(models.keys()),
        'model_details': {
            'lstm': {'name': 'LSTM (长短时记忆网络)', 'accuracy': '87.29%'},
            'transformer': {'name': 'Transformer', 'accuracy': '85.15%'},
            'bert': {'name': 'BERT (微调模型)', 'accuracy': '87.63%'}
        }
    }
    return jsonify(info)


if __name__ == '__main__':
    load_models()
    PORT = 7890
    print("\n" + "=" * 50)
    print("情感分析Web应用已启动")
    print("=" * 50)
    print(f"访问地址: http://127.0.0.1:{PORT}")
    print(f"可用模型: {list(models.keys())}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=PORT)