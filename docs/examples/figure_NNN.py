import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图的字体和标题等样式
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 14,
                    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

# 假设CSV文件路径
file_path = '100.csv'
df = pd.read_csv(file_path)
df = df[df["Number of Hops"] != 1]
df['IID Beta'] = df['IID Beta'].astype(str)

# 按 'IID Beta' 和 'Number of Trainers' 进行分组，计算各项的平均值
numeric_columns = df.select_dtypes(include='number').columns
grouped_df = df[['IID Beta'] + list(numeric_columns)].groupby(
    ['IID Beta', 'Number of Trainers']).mean().reset_index()

# 设置要绘制的三个指标
metrics = ['Average Test Accuracy', 'Train Time', 'Train Network Server']
titles = ['Accuracy Comparison', 'Train Time Comparison',
          'Communication Cost Comparison']
y_labels = ['Accuracy', 'Train Time (ms)', 'Total Communication Cost (Bytes)']

# 设置每个柱状图的宽度
bar_width = 0.3
# 获取每个 IID Beta 的唯一值，方便在并排放置时设置偏移
unique_betas = grouped_df['IID Beta'].unique()
# 设置x轴位置
num_trainers = grouped_df['Number of Trainers'].unique()
x_positions = np.arange(len(num_trainers))

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

# 绘制每个指标的柱状图
for i, metric in enumerate(metrics):
    for j, beta in enumerate(unique_betas):
        beta_data = grouped_df[grouped_df['IID Beta'] == beta]
        # 设置位置偏移，使不同的 IID Beta 值并排放置
        offset_positions = x_positions + \
            (j * bar_width) - (bar_width * (len(unique_betas) - 1) / 2)
        axes[i].bar(offset_positions, beta_data[metric],
                    width=bar_width, label=f'IID Beta {beta}')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Number of Trainers")
    axes[i].set_ylabel(y_labels[i])
    axes[i].set_xticks(x_positions)
    axes[i].set_xticklabels(num_trainers)

# 设置图例的位置
axes[2].legend(loc='upper right', title='IID Beta')
plt.subplots_adjust(wspace=0.3)

# 显示图形
plt.show()
