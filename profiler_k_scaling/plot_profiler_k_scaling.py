import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '../ablation_studies_profiler')
from plot_profiler_ablation import plot_ablation_chart, print_table

# ==========================================
# 数据准备
# ==========================================
df = pd.read_csv('results/kernel_time.csv')

# 计算每个k值相对于no_sampling (k0) 的overhead reduction
baseline = df['k0_mean (ms)']
k1_reduction = baseline / df['k1_mean (ms)']
k5_reduction = baseline / df['k5_mean (ms)']
k10_reduction = baseline / df['k10_mean (ms)']

configs = {
    'k=1': k1_reduction,
    'k=5': k5_reduction,
    'k=10': k10_reduction
}

labels = list(configs.keys())
means = np.array([configs[label].mean() for label in labels])
mins = np.array([configs[label].min() for label in labels])
maxs = np.array([configs[label].max() for label in labels])

# 打印表格
print_table(labels, means, mins, maxs)

# 绘图 (调整x轴范围适应k scaling数据)
plot_ablation_chart(
    labels, means, mins, maxs,
    'profiler_k_scaling.pdf',
    xlim=(0.5, 1400),
    xticks=[1, 10, 100, 1000]
)
