import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def plot_ablation_chart(labels, means, mins, maxs, output_file, xlim=(0.35, 2000), xticks=None):
    """
    绘制 ablation study 的水平条形图

    参数:
        labels: Y轴标签列表
        means: 每个配置的平均overhead reduction
        mins: 每个配置的最小overhead reduction
        maxs: 每个配置的最大overhead reduction
        output_file: 输出PDF文件路径
        xlim: X轴范围，默认 (0.35, 2000)
        xticks: X轴刻度，默认 [1, 10, 100, 1000]
    """
    if xticks is None:
        xticks = [1, 10, 100, 1000]

    # Matplotlib 的误差棒需要相对值 (mean - min) 和 (max - mean)
    xerr_lower = means - mins
    xerr_upper = maxs - means
    xerr = [xerr_lower, xerr_upper]

    # ==========================================
    # 样式设置 (OSDI 风格)
    # ==========================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'font.size': 33,
        'axes.labelsize': 36,
        'xtick.labelsize': 33,
        'ytick.labelsize': 36,
        'figure.figsize': (10, 6)
    })

    # ==========================================
    # 绘图核心逻辑
    # ==========================================
    fig, ax = plt.subplots()

    bars = ax.barh(labels, means - 1, left=1, color='gray', edgecolor='black', height=0.5,
                   xerr=xerr, capsize=6,
                   error_kw={'ecolor': 'black', 'elinewidth': 1.5})

    # ==========================================
    # 坐标轴与刻度设置
    # ==========================================
    ax.set_xscale('log')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, zorder=0)

    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Speedup (log)")

    ax.set_xlim(xlim)

    ax.tick_params(axis='y', which='both', length=0)

    # 调整y轴范围，让最下面的柱状图往上移动一点
    ax.set_ylim(-0.6, len(labels) - 0.4)

    # 开放式边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ==========================================
    # 添加数值标注
    # ==========================================
    for i in range(len(labels)):
        ax.text(means[i] * 1.05, i, f"{means[i]:.2f}x",
                va='center', fontsize=28, fontweight='normal')

        ax.text(mins[i], i - 0.28, f"{mins[i]:.2f}",
                ha='center', va='top', fontsize=32)

        ax.text(maxs[i], i - 0.28, f"{maxs[i]:.2f}",
                ha='center', va='top', fontsize=32)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_file}")


def print_table(labels, means, mins, maxs):
    """打印表格"""
    print("| Optimization Level                    | Overhead Reduction            |")
    print("|---------------------------------------|-------------------------------|")
    for i, label in enumerate(labels):
        reduction_str = f"{means[i]:.3f}x ({mins[i]:.3f}x–{maxs[i]:.3f}x)"
        print(f"| {label:<37} | {reduction_str:<29} |")


if __name__ == '__main__':
    # ==========================================
    # 数据准备
    # ==========================================
    df = pd.read_csv('results/profiler_timing_results.csv')

    baseline = df['baseline']
    skipping = baseline / df['+ skipping']
    sampling = baseline / df['+ sampling']
    skipping_sampling = baseline / df['+ skipping, sampling']

    configs = {
        'Symbolic': skipping,
        'Sampling': sampling,
        'Both': skipping_sampling
    }

    labels = list(configs.keys())
    means = np.array([configs[label].mean() for label in labels])
    mins = np.array([configs[label].min() for label in labels])
    maxs = np.array([configs[label].max() for label in labels])

    # 打印表格
    print_table(labels, means, mins, maxs)

    # 绘图
    plot_ablation_chart(labels, means, mins, maxs, 'profiler_ablation_study.pdf')
