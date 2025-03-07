import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the CSV file

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def load_csv_file(file_path):
    return pd.read_csv(file_path)


file_path = "NC_papers100M.csv"
df = load_csv_file(file_path)

# 2. Filter data for specific IID Beta and Hop values
iid_beta_values = [10000]
hop_values = [0]
batch_sizes = [16, 32, 64]

# Helper function to filter data


def filter_data(df, iid_beta_value, hop_value):
    return df.loc[
        (df["IID Beta"] == iid_beta_value) & (df["Number of Hops"] == hop_value)
    ]


# Function to add values on top of bars


def add_values_on_bars(ax, values, xpos, width):
    for i, v in enumerate(values):
        ax.text(
            i + xpos, v + 0.01 * v, f"{v:.2f}", ha="center", va="bottom", fontsize=10
        )


# Function to calculate and set y-axis limits based on 70% range


def set_scaled_ylim(ax, values):
    min_val, max_val = values.min(), values.max()
    if min_val == max_val:
        ax.set_ylim(0, max_val * 1.1)  # 如果没有差异，直接从 0 到 1.1 倍的最大值
    else:
        range_val = max_val - min_val
        lower_bound = min_val - 0.5 * range_val  # 下限比最小值稍小
        upper_bound = max_val + 0.5 * range_val  # 上限比最大值稍大
        ax.set_ylim(lower_bound, upper_bound)  # 设置 y 轴范围


def set_scaled_ylim_1(ax, values):
    min_val, max_val = values.min(), values.max()
    if min_val == max_val:
        ax.set_ylim(0, max_val * 1.1)  # 如果没有差异，直接从 0 到 1.1 倍的最大值
    else:
        range_val = max_val - min_val
        lower_bound = 0  # 下限比最小值稍小
        upper_bound = 0.8  # 上限比最大值稍大
        ax.set_ylim(lower_bound, upper_bound)  # 设置 y 轴范围


# 3. Plot three separate charts and combine them into one figure


def plot_combined_charts(df, iid_beta_value, hop_value):
    batch_data = df[df["Batch Size"].isin(batch_sizes)]
    memory_columns = [f"Train Network Large{i}" for i in range(1, 11)]
    batch_data.loc[:, "Total Communication Cost"] = batch_data[memory_columns].sum(
        axis=1
    )

    # Get train_time, memory, accuracy
    train_time = batch_data.groupby("Batch Size")["Train Time"].mean()
    memory = batch_data.groupby("Batch Size")["Total Communication Cost"].mean()
    accuracy = batch_data.groupby("Batch Size")["Average Test Accuracy"].mean()

    # Create a figure with 3 subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    # X-axis positions for the bars
    x = np.arange(len(batch_sizes))
    width = 0.4

    # Plot the bars for Train Time
    ax1.bar(x, train_time, width, color="orange")
    ax1.set_title("Train Time")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Train Time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes])
    # add_values_on_bars(ax1, train_time, 0, width)
    set_scaled_ylim(ax1, train_time)  # 设置 y 轴范围使得差异占 70%

    # Plot the bars for Memory
    ax2.bar(x, memory, width, color="skyblue")
    ax2.set_title("Communication Cost")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Bytes")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(bs) for bs in batch_sizes])
    add_values_on_bars(ax2, memory, 0, width)
    set_scaled_ylim(ax2, memory)  # 设置 y 轴范围使得差异占 70%

    # Plot the bars for Accuracy
    ax3.bar(x, accuracy, width, color="green")
    ax3.set_title("Test Accuracy")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Accuracy")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(bs) for bs in batch_sizes])
    add_values_on_bars(ax3, accuracy, 0, width)
    set_scaled_ylim_1(ax3, accuracy)  # 设置 y 轴范围使得差异占 70%

    # Set a main title for the figure
    plt.suptitle(f"Combined Plot (IID Beta {iid_beta_value}, Hop {hop_value})")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    # Show the plot
    plt.show()


# 4. Loop through the IID Beta values and plot the charts
for iid_beta_value in iid_beta_values:
    for hop_value in hop_values:
        filtered_df = filter_data(df, iid_beta_value, hop_value)
        plot_combined_charts(filtered_df, iid_beta_value, hop_value)
