import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the CSV file


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


file_path = "LP2.csv"
df = load_csv_file(file_path)

# 2. Define algorithms, datasets, and trainers
algorithms = ["4D-FED-GNN+", "STFL", "StaticGNN", "FedLink"]
datasets = ["US", "US+BR", "US+BR+ID+TR+JP"]
trainers = [10]  # Specify the number of trainers

# Function to filter data based on Algorithm, Dataset, and Number of Trainers


def filter_data(df, algorithm, dataset, trainers):
    return df[
        (df["Algorithm"] == algorithm)
        & (df["Dataset"] == dataset)
        & (df["Number of Trainers"].isin(trainers))
    ]


# 3. Plot chart for comparing Accuracy, Train Time, and Communication Cost


def plot_combined_comparison(df, algorithms, datasets, trainers):
    width = 0.15  # Width of each bar
    algorithm_range = np.arange(len(algorithms))  # X positions for bars

    # Track min and max values for scaling y-axis for each metric
    min_values = {
        "accuracy": float("inf"),
        "train_time": float("inf"),
        "communication_cost": float("inf"),
    }
    max_values = {
        "accuracy": float("-inf"),
        "train_time": float("-inf"),
        "communication_cost": float("-inf"),
    }

    # Create a figure with 3 subplots in one row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for j, dataset in enumerate(datasets):
        accuracy_values = []
        train_time_values = []
        communication_cost_values = []

        for i, algorithm in enumerate(algorithms):
            filtered_df = filter_data(df, algorithm, dataset, trainers)
            avg_accuracy = filtered_df["Average Test AUC"].mean()
            avg_train_time = filtered_df["Train Time"].mean()
            avg_communication_cost = (
                filtered_df[[f"Train Network Large{k}" for k in range(1, 11)]]
                .sum(axis=1)
                .mean()
            )  # Summing `Train Network Large` columns

            accuracy_values.append(avg_accuracy)
            train_time_values.append(avg_train_time)
            communication_cost_values.append(avg_communication_cost)

            # Update min and max values for each metric
            min_values["accuracy"] = min(min_values["accuracy"], avg_accuracy)
            max_values["accuracy"] = max(max_values["accuracy"], avg_accuracy)
            min_values["train_time"] = min(min_values["train_time"], avg_train_time)
            max_values["train_time"] = max(max_values["train_time"], avg_train_time)
            min_values["communication_cost"] = min(
                min_values["communication_cost"], avg_communication_cost
            )
            max_values["communication_cost"] = max(
                max_values["communication_cost"], avg_communication_cost
            )

        # Plot the bars for each metric and dataset
        ax1.bar(
            algorithm_range + j * width,
            accuracy_values,
            width=width,
            label=f"{dataset}",
        )
        ax2.bar(
            algorithm_range + j * width,
            train_time_values,
            width=width,
            label=f"{dataset}",
        )
        ax3.bar(
            algorithm_range + j * width,
            communication_cost_values,
            width=width,
            label=f"{dataset}",
        )

    # Set titles and labels for each subplot
    ax1.set_title("AUC Comparison")
    ax1.set_xlabel("Algorithms")
    ax1.set_ylabel("AUC")
    ax1.set_xticks(algorithm_range + width * (len(datasets) - 1) / 2)
    ax1.set_xticklabels(algorithms)

    ax2.set_title("Train Time Comparison")
    ax2.set_xlabel("Algorithms")
    ax2.set_ylabel("Train Time (ms)")
    ax2.set_xticks(algorithm_range + width * (len(datasets) - 1) / 2)
    ax2.set_xticklabels(algorithms)

    ax3.set_title("Communication Cost Comparison")
    ax3.set_xlabel("Algorithms")
    ax3.set_ylabel("Total Communication Cost (Bytes)")
    ax3.set_xticks(algorithm_range + width * (len(datasets) - 1) / 2)
    ax3.set_xticklabels(algorithms)

    # Adjust y-axis for each subplot to occupy 70% of the plot's height
    for ax, metric in zip(
        [ax1, ax2, ax3], ["accuracy", "train_time", "communication_cost"]
    ):
        diff = max_values[metric] - min_values[metric]
        lower_bound = min_values[metric] - diff / 5
        if metric == "communication_cost":
            lower_bound = 0
        ax.set_ylim(lower_bound, max_values[metric] * 1.01)

    # Display the legend
    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Datasets")

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.3)
    plt.show()


# 4. Call the plotting function
plot_combined_comparison(df, algorithms, datasets, trainers)
