import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter


def plot_speedup(input_csv: Path, output_pdf: str, ylim: tuple, yticks: np.ndarray, exclude: list = None, case_order: list = None, bar_color: str = '#6A8DC2'):
    """生成 speedup 垂直条形图

    返回: case_order (list) - 用于保持其他图的顺序一致
    """

    # 读取数据
    df = pd.read_csv(input_csv)

    # 清理数据：去除空格
    df.columns = df.columns.str.strip()
    df['Case Name'] = df['Case Name'].str.strip()

    # 排除指定的 case
    if exclude:
        df = df[~df['Case Name'].isin(exclude)]

    # 排序：如果指定了 case_order 则按该顺序，否则按 speedup 降序
    if case_order:
        # 只保留 case_order 中存在于 df 的 case
        valid_order = [c for c in case_order if c in df['Case Name'].values]
        df_sorted = df.set_index('Case Name').loc[valid_order].reset_index()
    else:
        df_sorted = df.sort_values('speedup', ascending=False)

    # 全局绘图风格设置
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams['font.family'] = 'sans-serif'

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # 定义颜色
    edge_color = 'black'
    error_bar_color = 'black'

    # 绘制垂直条形图
    x_positions = np.arange(len(df_sorted)) * 2
    bars = ax.bar(
        x=x_positions,
        height=df_sorted['speedup'],
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.5,
        width=1.4,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['Case Name'], rotation=30, ha='right')

    # 添加基准线
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # 设置X轴范围
    ax.set_xlim(-1, x_positions[-1] + 1)

    # 坐标轴设置
    ax.set_ylabel('Speedup', fontsize=26, labelpad=10)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.tick_params(axis='x', pad=2, labelsize=20, labelcolor='0.2')
    ax.tick_params(axis='y', pad=0, labelsize=22)

    # 网格线
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
    ax.grid(axis='x', visible=False)

    # 边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_pdf}")

    # 返回 case 顺序，供其他图复用
    return df_sorted['Case Name'].tolist()


def main():
    script_dir = Path(__file__).resolve().parent

    # 统一配置
    unroll_ylim = (0.5, 1.7)
    unroll_yticks = [0.5, 1.0, 1.5]
    mask_ylim = (0.5, 4.0)
    mask_yticks = [1.0, 2.0, 3.0, 4.0]

    # 生成 unroll speedup 图，并获取 case 顺序
    unroll_order = plot_speedup(
        input_csv=script_dir / 'unroll_stats.csv',
        output_pdf='unroll_speedup.pdf',
        ylim=unroll_ylim,
        yticks=unroll_yticks,
        exclude=['triton_matmul']
    )

    # 生成 mask speedup 图，并获取 case 顺序
    mask_order = plot_speedup(
        input_csv=script_dir / 'mask_stats.csv',
        output_pdf='mask_speedup.pdf',
        ylim=mask_ylim,
        yticks=mask_yticks
    )

    # 生成 AMD 版本的图（如果对应的 CSV 文件存在），使用 NVIDIA 版本的顺序
    amd_bar_color = (102/255, 166/255, 103/255)  # RGB (102, 166, 103)

    unroll_amd_csv = script_dir / 'unroll_stats_amd.csv'
    if unroll_amd_csv.exists():
        plot_speedup(
            input_csv=unroll_amd_csv,
            output_pdf='unroll_speedup_amd.pdf',
            ylim=unroll_ylim,
            yticks=unroll_yticks,
            exclude=['triton_matmul'],
            case_order=unroll_order,
            bar_color=amd_bar_color
        )

    mask_amd_csv = script_dir / 'mask_stats_amd.csv'
    if mask_amd_csv.exists():
        plot_speedup(
            input_csv=mask_amd_csv,
            output_pdf='mask_speedup_amd.pdf',
            ylim=mask_ylim,
            yticks=mask_yticks,
            case_order=mask_order,
            bar_color=amd_bar_color
        )


if __name__ == "__main__":
    main()
