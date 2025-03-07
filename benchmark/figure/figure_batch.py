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
    df = pd.read_csv(file_path)
    return df


file_path = "NC_arxiv_batchsize.csv"
df = load_csv_file(file_path)

# 2. Define specific IID Beta, Hop values, and Batch Sizes
iid_beta_values = [10000]
hop_values = [1]
batch_sizes = [16, 32, 64]

# Function to filter data based on IID Beta and Hop


def filter_data(df, iid_beta_value, hop_value):
    return df[(df["IID Beta"] == iid_beta_value) & (df["Number of Hops"] == hop_value)]


# 3. Plot combined charts for Time, Memory, and Accuracy comparison


def plot_combined_charts(df, iid_beta_value, hop_value):
    batch_data = df[df["Batch Size"].isin(batch_sizes)]
    width = 0.25  # Width of the bars
    batch_size_range = np.arange(len(batch_sizes))  # X positions for bars

    # Calculate values for each metric
    pretrain_values = (
        batch_data.groupby("Batch Size")["Pretrain Time"].mean()
        if hop_value == 1
        else None
    )
    train_values = batch_data.groupby("Batch Size")["Train Time"].mean()
    pre_columns = [f"Pretrain Network Large{i}" for i in range(1, 11)]
    pre_values = (
        batch_data[pre_columns].sum(axis=1).groupby(batch_data["Batch Size"]).mean()
    )
    tre_columns = [f"Train Network Large{i}" for i in range(1, 11)]
    tre_values = (
        batch_data[tre_columns].sum(axis=1).groupby(batch_data["Batch Size"]).mean()
    )
    accuracy_values = batch_data.groupby("Batch Size")["Average Test Accuracy"].mean()

    # Create a figure with 3 subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Train Time and Pretrain Time
    if hop_value == 1:
        ax1.bar(
            batch_size_range - width / 2,
            pretrain_values.values,
            width=width,
            label="Pretrain Time",
            color="skyblue",
        )
    ax1.bar(
        batch_size_range + (width / 2 if hop_value == 1 else 0),
        train_values.values,
        width=width,
        label="Train Time",
        color="orange",
    )
    ax1.set_title(
        f"Pretrain vs Train Time (IID Beta {iid_beta_value}, Hop {hop_value})"
    )
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xticks(batch_size_range)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(loc="lower right")

    # Plot Total Train Memory
    if hop_value == 1:
        ax2.bar(
            batch_size_range - width / 2,
            pre_values.values,
            width=width,
            label="Pretrain Communication Cost",
            color="skyblue",
        )
    ax2.bar(
        batch_size_range + (width / 2 if hop_value == 1 else 0),
        tre_values.values,
        width=width,
        label="Train Communication Cost",
        color="orange",
    )
    ax2.set_title(
        f"Total Communication Cost (IID Beta {iid_beta_value}, Hop {hop_value})"
    )
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Communication Cost (Bytes)")
    ax2.set_xticks(batch_size_range)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(loc="lower right")
    # Plot Accuracy
    ax3.bar(batch_size_range, accuracy_values.values, color="green", width=width)
    ax3.set_title(f"Test Accuracy (IID Beta {iid_beta_value}, Hop {hop_value})")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Accuracy")
    ax3.set_xticks(batch_size_range)
    ax3.set_xticklabels(batch_sizes)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[1, 0, 0, 0.96])
    plt.show()


# 4. Loop through the IID Beta values and Hops, and plot the combined charts
for iid_beta_value in iid_beta_values:
    for hop_value in hop_values:
        filtered_df = filter_data(df, iid_beta_value, hop_value)
        plt.subplots_adjust(left=0.2, wspace=0.3)
        plot_combined_charts(filtered_df, iid_beta_value, hop_value)
