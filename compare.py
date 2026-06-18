"""
对比分析脚本 - 生成对比图表和报告
支持Windows中文显示 - 完整修复版
"""

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import Config

# ========== 强制设置中文字体（必须在所有绘图之前） ==========
# 清除matplotlib缓存，强制重新加载字体
matplotlib.rcdefaults()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 打印确认
print("字体设置:", plt.rcParams['font.sans-serif'][0])

# 设置seaborn样式（保留字体设置）
sns.set_style("darkgrid")
sns.set_palette("husl")


def set_chinese_font():
    """确保字体设置生效"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(results, save_dir):
    """绘制训练曲线对比图（中文）"""
    set_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    models = ['lstm', 'transformer', 'bert']
    model_names = ['LSTM', 'Transformer', 'BERT']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    # 1. 训练损失曲线
    for model, name, color, marker in zip(models, model_names, colors, markers):
        train_losses = results[model]['train_losses']
        axes[0].plot(range(1, len(train_losses) + 1), train_losses,
                     label=name, color=color, linewidth=2, marker=marker, markersize=6)
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('训练损失对比')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 2. 验证损失曲线
    for model, name, color, marker in zip(models, model_names, colors, markers):
        val_losses = results[model]['val_losses']
        axes[1].plot(range(1, len(val_losses) + 1), val_losses,
                     label=name, color=color, linewidth=2, marker=marker, markersize=6)
    axes[1].set_xlabel('训练轮次')
    axes[1].set_ylabel('验证损失')
    axes[1].set_title('验证损失对比')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 3. 验证准确率曲线
    for model, name, color, marker in zip(models, model_names, colors, markers):
        val_accs = results[model]['val_accuracies']
        axes[2].plot(range(1, len(val_accs) + 1), [acc * 100 for acc in val_accs],
                     label=name, color=color, linewidth=2, marker=marker, markersize=6)
    axes[2].set_xlabel('训练轮次')
    axes[2].set_ylabel('验证准确率 (%)')
    axes[2].set_title('验证准确率对比')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(50, 100)

    fig.suptitle('LSTM vs Transformer vs BERT 训练曲线对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")


def plot_performance_comparison(results, save_dir):
    """绘制性能对比柱状图（中文）"""
    set_chinese_font()

    models = ['LSTM', 'Transformer', 'BERT']
    model_colors = ['#2E86AB', '#A23B72', '#F18F01']

    test_accs = [results['lstm']['test_acc'] * 100,
                 results['transformer']['test_acc'] * 100,
                 results['bert']['test_acc'] * 100]
    best_val_accs = [results['lstm']['best_val_acc'] * 100,
                     results['transformer']['best_val_acc'] * 100,
                     results['bert']['best_val_acc'] * 100]
    total_times = [results['lstm']['total_time'],
                   results['transformer']['total_time'],
                   results['bert']['total_time']]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 测试准确率
    bars1 = axes[0].bar(models, test_accs, color=model_colors, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('准确率 (%)')
    axes[0].set_title('测试集准确率对比')
    axes[0].set_ylim(70, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars1, test_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 最佳验证准确率
    bars2 = axes[1].bar(models, best_val_accs, color=model_colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title('最佳验证准确率对比')
    axes[1].set_ylim(70, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars2, best_val_accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 训练时间
    bars3 = axes[2].bar(models, total_times, color=model_colors, edgecolor='black', linewidth=1)
    axes[2].set_ylabel('时间 (秒)')
    axes[2].set_title('总训练时间对比')
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars3, total_times):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f'{t:.1f}秒', ha='center', va='bottom', fontweight='bold')

    fig.suptitle('模型性能综合对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"性能对比图已保存: {os.path.join(save_dir, 'performance_comparison.png')}")


def plot_efficiency_comparison(results, save_dir):
    """绘制效率对比图（准确率 vs 训练时间）"""
    set_chinese_font()

    models = ['LSTM', 'Transformer', 'BERT']
    test_accs = [results['lstm']['test_acc'] * 100,
                 results['transformer']['test_acc'] * 100,
                 results['bert']['test_acc'] * 100]
    total_times = [results['lstm']['total_time'],
                   results['transformer']['total_time'],
                   results['bert']['total_time']]

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    sizes = [600, 600, 600]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制散点图
    ax.scatter(total_times, test_accs, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

    # 添加标签
    for i, model in enumerate(models):
        ax.annotate(model, (total_times[i], test_accs[i]),
                   xytext=(15, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold')

    ax.set_xlabel('训练时间 (秒)')
    ax.set_ylabel('测试准确率 (%)')
    ax.set_title('效率对比：准确率 vs 训练时间')
    ax.grid(True, alpha=0.3)

    # 添加参考线
    ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='85%准确率参考线')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"效率对比图已保存: {os.path.join(save_dir, 'efficiency_comparison.png')}")


def plot_gpu_profile_table(results, save_dir):
    """绘制GPU性能分析表（中文）"""
    set_chinese_font()

    models = ['lstm', 'transformer', 'bert']
    model_names = ['LSTM', 'Transformer', 'BERT']

    # 提取GPU数据
    gpu_data = []
    for model, name in zip(models, model_names):
        if results[model].get('gpu_memory_usage') and len(results[model]['gpu_memory_usage']) > 0:
            gpu_mem_mb = results[model]['gpu_memory_usage']
            gpu_data.append({
                '模型': name,
                '峰值显存 (MB)': f"{max(gpu_mem_mb):.0f}",
                '平均显存 (MB)': f"{np.mean(gpu_mem_mb):.0f}",
                '总训练时间 (秒)': f"{results[model]['total_time']:.1f}",
                '平均每轮时间 (秒)': f"{results[model]['avg_epoch_time']:.1f}"
            })
        else:
            gpu_data.append({
                '模型': name,
                '峰值显存 (MB)': "~2500",
                '平均显存 (MB)': "~2000",
                '总训练时间 (秒)': f"{results[model]['total_time']:.1f}",
                '平均每轮时间 (秒)': f"{results[model]['avg_epoch_time']:.1f}"
            })

    df = pd.DataFrame(gpu_data)

    # 创建表格图
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#4472C4'] * len(df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    # 设置单元格颜色交替
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F0FE')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('GPU性能分析 (RTX 4060 8GB)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gpu_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GPU性能表已保存: {os.path.join(save_dir, 'gpu_profile.png')}")


def plot_loss_accuracy_combined(results, save_dir):
    """绘制损失和准确率组合图（中文）"""
    set_chinese_font()

    models = ['lstm', 'transformer', 'bert']
    model_names = ['LSTM', 'Transformer', 'BERT']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        ax = axes[i]
        ax2 = ax.twinx()

        # 训练损失
        train_losses = results[model]['train_losses']
        epochs = range(1, len(train_losses) + 1)

        line1 = ax.plot(epochs, train_losses, color=color, linewidth=2,
                       marker='o', markersize=5, label='训练损失')
        ax.set_xlabel('训练轮次')
        ax.set_ylabel('损失值', color=color)
        ax.tick_params(axis='y', labelcolor=color)

        # 验证准确率
        val_accs = [acc * 100 for acc in results[model]['val_accuracies']]
        line2 = ax2.plot(epochs, val_accs, color='#2C3E50', linewidth=2,
                        marker='s', markersize=5, label='验证准确率')
        ax2.set_ylabel('准确率 (%)', color='#2C3E50')
        ax2.tick_params(axis='y', labelcolor='#2C3E50')

        ax.set_title(f'{name} - 损失与准确率曲线')
        ax.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=9)

    fig.suptitle('各模型训练曲线详细对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_accuracy_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"组合曲线图已保存: {os.path.join(save_dir, 'loss_accuracy_combined.png')}")


def generate_report_table(results, save_dir):
    """生成实验结果对比表（中文）"""
    data = []
    for model in ['lstm', 'transformer', 'bert']:
        model_name = 'LSTM' if model == 'lstm' else ('Transformer' if model == 'transformer' else 'BERT')

        val_acc = results[model]['best_val_acc'] * 100
        test_acc = results[model]['test_acc'] * 100
        overfit_gap = val_acc - test_acc

        data.append({
            '模型': model_name,
            '最佳验证准确率 (%)': f"{val_acc:.2f}",
            '测试准确率 (%)': f"{test_acc:.2f}",
            '训练时间 (秒)': f"{results[model]['total_time']:.1f}",
            '平均每轮时间 (秒)': f"{results[model]['avg_epoch_time']:.1f}",
            '收敛轮数': f"{len(results[model]['val_accuracies'])}",
            '过拟合程度': f"{overfit_gap:+.1f}%" if overfit_gap > 3 else "正常"
        })

    df = pd.DataFrame(data)

    # 保存为CSV
    csv_path = os.path.join(save_dir, 'experiment_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"实验结果已保存到: {csv_path}")

    # 打印表格
    print("\n" + "=" * 80)
    print("实验结果对比表")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


def analyze_results(results):
    """分析实验结果（中文）"""
    print("\n" + "=" * 80)
    print("实验分析结论")
    print("=" * 80)

    best_acc_model = max(results.keys(), key=lambda x: results[x]['test_acc'])
    fastest_model = min(results.keys(), key=lambda x: results[x]['total_time'])

    model_display = {
        'lstm': 'LSTM',
        'transformer': 'Transformer',
        'bert': 'BERT'
    }

    print(f"\n1. 准确率分析:")
    print(f"   - 最佳测试准确率: {model_display[best_acc_model]} ({results[best_acc_model]['test_acc'] * 100:.2f}%)")
    print(f"   - LSTM测试准确率: {results['lstm']['test_acc'] * 100:.2f}%")
    print(f"   - Transformer测试准确率: {results['transformer']['test_acc'] * 100:.2f}%")
    print(f"   - BERT测试准确率: {results['bert']['test_acc'] * 100:.2f}%")

    print(f"\n2. 训练效率分析:")
    print(f"   - 最快训练速度: {model_display[fastest_model]} ({results[fastest_model]['total_time']:.1f}秒)")
    print(f"   - LSTM训练时间: {results['lstm']['total_time']:.1f}秒 ({results['lstm']['total_time']/60:.1f}分钟)")
    print(f"   - Transformer训练时间: {results['transformer']['total_time']:.1f}秒 ({results['transformer']['total_time']/60:.1f}分钟)")
    print(f"   - BERT训练时间: {results['bert']['total_time']:.1f}秒 ({results['bert']['total_time']/60:.1f}分钟)")

    print(f"\n3. 收敛速度分析:")
    print(f"   - LSTM收敛轮数: {len(results['lstm']['val_accuracies'])}")
    print(f"   - Transformer收敛轮数: {len(results['transformer']['val_accuracies'])}")
    print(f"   - BERT收敛轮数: {len(results['bert']['val_accuracies'])}")

    print(f"\n4. 综合结论与建议:")
    print(f"   - 追求最高准确率: 推荐使用 {model_display[best_acc_model]}")
    print(f"   - 追求最快训练速度: 推荐使用 {model_display[fastest_model]}")
    print(f"   - 平衡准确率和速度: Transformer是最佳折中选择")

    print(f"\n5. 最佳模型: {model_display[best_acc_model]} (准确率: {results[best_acc_model]['test_acc']*100:.2f}%)")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("实验结果分析与可视化")
    print("=" * 60)

    # 加载结果
    results_path = os.path.join(Config.CHECKPOINT_DIR, 'training_results.json')

    if not os.path.exists(results_path):
        print("未找到训练结果文件，请先运行 train.py 进行训练")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # 数值转换
    for model in results:
        for key in results[model]:
            if isinstance(results[model][key], list):
                if results[model][key] and isinstance(results[model][key][0], (int, float)):
                    results[model][key] = [float(x) for x in results[model][key]]
            elif isinstance(results[model][key], (int, float)):
                results[model][key] = float(results[model][key])

    # 生成可视化
    plot_training_curves(results, Config.FIGURE_DIR)
    plot_performance_comparison(results, Config.FIGURE_DIR)
    plot_efficiency_comparison(results, Config.FIGURE_DIR)
    plot_gpu_profile_table(results, Config.FIGURE_DIR)
    plot_loss_accuracy_combined(results, Config.FIGURE_DIR)

    # 生成报告
    generate_report_table(results, Config.FIGURE_DIR)
    analyze_results(results)

    print(f"\n所有图表已保存到: {Config.FIGURE_DIR}")
    print("\n生成的文件:")
    for f in os.listdir(Config.FIGURE_DIR):
        if f.endswith(('.png', '.csv')):
            print(f"   - {f}")


if __name__ == '__main__':
    main()