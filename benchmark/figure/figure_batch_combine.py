import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the CSV file


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


file_path = "NC_arxiv_batchsize.csv"
df = load_csv_file(file_path)

# 2. Filter data for specific Batch Sizes and Hop values
iid_beta_values = [10000, 100, 10]
batch_sizes = [-1, 16, 32, 64]
hop_value = 1  # Set a specific hop value, or change as needed

# Function to filter data based on IID Beta, Hop and Batch Size


def filter_data(df, iid_beta_value, hop_value):
    return df[
        (df["IID Beta"] == iid_beta_value)
        & (df["Number of Hops"] == hop_value)
        & (df["Batch Size"].isin(batch_sizes))
    ]


# 3. Plot chart for Accuracy comparison across Batch Sizes for different IID Beta values


def plot_accuracy_comparison(df, hop_value):
    width = 0.2  # Width of each bar
    batch_size_range = np.arange(len(batch_sizes))  # X positions for bars

    min_accuracy = float("inf")  # Track the minimum accuracy
    max_accuracy = float("-inf")  # Track the maximum accuracy

    for i, iid_beta_value in enumerate(iid_beta_values):
        filtered_df = filter_data(df, iid_beta_value, hop_value)
        accuracy_values = filtered_df.groupby("Batch Size")[
            "Average Test Accuracy"
        ].mean()

        # Update min and max accuracy
        min_accuracy = min(min_accuracy, accuracy_values.min())
        max_accuracy = max(max_accuracy, accuracy_values.max())

        # Plot the bars for each IID Beta, with slight shifts in x positions to avoid overlap
        plt.bar(
            batch_size_range + i * width,
            accuracy_values.values,
            width=width,
            label=f"IID Beta {iid_beta_value}",
        )

    # Calculate diff and adjust the y-axis to make diff occupy 70% of the plot's height
    diff = max_accuracy - min_accuracy
    # Calculate lower bound to make diff occupy 70% of the plot
    lower_bound = max_accuracy - diff / 0.7

    # Set y-axis limit to make the difference more visible
    plt.ylim(lower_bound, max_accuracy * 1.01)

    # Title and labels
    plt.title(f"Test Accuracy Comparison (Hop {hop_value})")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")

    # Set x-axis labels to batch sizes
    plt.xticks(batch_size_range + width, labels=batch_sizes)

    plt.legend()
    plt.show()


# 4. Call the plotting function for the given hop value
plot_accuracy_comparison(df, hop_value)
