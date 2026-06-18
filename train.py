"""
训练脚本 - 支持混合精度训练、TensorBoard监控、学习率调度
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config, set_seed
from data_utils import get_data_loaders, get_timestamp
from models import LSTMSentiment, TransformerSentiment, get_bert_model


class Trainer:
    """训练器类，封装训练逻辑"""

    def __init__(self, model, model_name, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 根据模型类型设置不同的学习率
        if model_name == 'bert':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.BERT_LEARNING_RATE)
            self.scheduler = None
        elif model_name == 'transformer':
            # Transformer使用更小的学习率
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-4,  # Transformer专用学习率
                weight_decay=config.WEIGHT_DECAY
            )
            # Transformer也使用学习率调度
            total_steps = len(train_loader) * config.NUM_EPOCHS
            warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:  # LSTM
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
            total_steps = len(train_loader) * config.NUM_EPOCHS
            warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 混合精度训练
        self.use_amp = config.USE_AMP and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard
        log_dir = os.path.join(config.LOG_DIR, f"{model_name}_{get_timestamp()}")
        self.writer = SummaryWriter(log_dir)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0
        self.patience_counter = 0

        # GPU监控
        self.gpu_memory_usage = []

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} [Train]')

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            if self.model_name == 'bert':
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    if self.model_name == 'bert':
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    else:
                        logits = self.model(input_ids)
                    loss = self.criterion(logits, labels)

                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()
                # 梯度裁剪（对Transformer重要）
                if self.model_name == 'transformer':
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.model_name == 'bert':
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    logits = self.model(input_ids)
                loss = self.criterion(logits, labels)

                loss.backward()
                # 梯度裁剪
                if self.model_name == 'transformer':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar(f'{self.model_name}/train_loss_step', loss.item(), global_step)

            # 每100步记录GPU内存使用
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(self.device) / 1024 ** 2
                self.gpu_memory_usage.append(gpu_mem)
                self.writer.add_scalar(f'{self.model_name}/gpu_memory_MB', gpu_mem, global_step)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        return avg_loss, epoch_time

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1} [Val]')

            for batch in pbar:
                if self.model_name == 'bert':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(input_ids)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                num_batches += 1

                pbar.set_postfix({'acc': f'{(correct / total) * 100:.2f}%'})

        avg_loss = total_loss / num_batches
        accuracy = correct / total

        # 记录到TensorBoard
        self.writer.add_scalar(f'{self.model_name}/val_loss', avg_loss, epoch)
        self.writer.add_scalar(f'{self.model_name}/val_accuracy', accuracy, epoch)

        if self.scheduler is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'{self.model_name}/learning_rate', current_lr, epoch)

        # 保存最佳模型
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.save_checkpoint(epoch, is_best=True)
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_name': self.model_name
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')

        torch.save(checkpoint, path)
        print(f"模型已保存: {path}")

    def train(self, num_epochs):
        """完整训练流程"""
        print(f"\n{'=' * 50}")
        print(f"开始训练 {self.model_name} 模型")
        print(f"{'=' * 50}")
        print(f"设备: {self.device}")
        print(f"混合精度训练: {self.use_amp}")
        print(f"初始学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"训练轮数: {num_epochs}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")

        epoch_times = []

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            # 训练
            train_loss, train_time = self.train_epoch(epoch)
            epoch_times.append(train_time)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # 打印结果
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Epoch Time: {train_time:.2f}s")

            # 早停检查
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n早停触发！验证准确率连续{self.patience_counter}轮未提升。")
                break

        # 训练总结
        total_time = sum(epoch_times)
        avg_epoch_time = total_time / len(epoch_times)

        print(f"\n{'=' * 50}")
        print(f"{self.model_name} 训练完成")
        print(f"{'=' * 50}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"总训练时间: {total_time:.2f}s")
        print(f"平均每轮时间: {avg_epoch_time:.2f}s")

        self.writer.close()

        return {
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'gpu_memory_usage': self.gpu_memory_usage
        }


def test_model(model, model_name, test_loader, device):
    """在测试集上评估模型"""
    print(f"\n{'=' * 50}")
    print(f"测试 {model_name} 模型")
    print(f"{'=' * 50}")

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')

        for batch in pbar:
            if model_name == 'bert':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'acc': f'{(correct / total) * 100:.2f}%'})

    accuracy = correct / total
    print(f"\n测试准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return accuracy, all_preds, all_labels


def run_training():
    """运行所有模型的训练"""
    print("\n" + "=" * 60)
    print("LSTM vs Transformer vs BERT 对比实验")
    print("=" * 60)

    Config.print_config()

    data_loaders = get_data_loaders()

    results = {}

    # 1. 训练LSTM
    print("\n" + "=" * 60)
    print("开始训练 LSTM 模型")
    print("=" * 60)
    lstm_model = LSTMSentiment(Config.LSTM_CONFIG)
    lstm_trainer = Trainer(
        lstm_model, 'lstm',
        data_loaders['lstm'][0], data_loaders['lstm'][1],
        Config.DEVICE, Config
    )
    lstm_results = lstm_trainer.train(Config.NUM_EPOCHS)
    lstm_test_acc, _, _ = test_model(lstm_model, 'lstm', data_loaders['lstm'][2], Config.DEVICE)
    results['lstm'] = {**lstm_results, 'test_acc': lstm_test_acc}

    # 2. 训练Transformer
    print("\n" + "=" * 60)
    print("开始训练 Transformer 模型")
    print("=" * 60)
    transformer_model = TransformerSentiment(Config.TRANSFORMER_CONFIG)
    transformer_trainer = Trainer(
        transformer_model, 'transformer',
        data_loaders['transformer'][0], data_loaders['transformer'][1],
        Config.DEVICE, Config
    )
    transformer_results = transformer_trainer.train(Config.NUM_EPOCHS)
    transformer_test_acc, _, _ = test_model(transformer_model, 'transformer', data_loaders['transformer'][2], Config.DEVICE)
    results['transformer'] = {**transformer_results, 'test_acc': transformer_test_acc}

    # 3. 训练BERT
    print("\n" + "=" * 60)
    print("开始训练 BERT 模型")
    print("=" * 60)
    bert_model = get_bert_model()
    bert_trainer = Trainer(
        bert_model, 'bert',
        data_loaders['bert'][0], data_loaders['bert'][1],
        Config.DEVICE, Config
    )
    bert_results = bert_trainer.train(Config.NUM_EPOCHS)
    bert_test_acc, _, _ = test_model(bert_model, 'bert', data_loaders['bert'][2], Config.DEVICE)
    results['bert'] = {**bert_results, 'test_acc': bert_test_acc}

    # 保存结果
    results_path = os.path.join(Config.CHECKPOINT_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n训练结果已保存到: {results_path}")

    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"{model_name.upper()} 测试准确率: {result['test_acc']*100:.2f}%")

    # 找出最佳模型
    best_model = max(results.keys(), key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 最佳模型: {best_model.upper()} (准确率: {results[best_model]['test_acc']*100:.2f}%)")

    return results


if __name__ == '__main__':
    set_seed(Config.SEED)
    results = run_training()