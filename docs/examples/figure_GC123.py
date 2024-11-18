import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the CSV file
def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df

file_path = "11.csv"
df = load_csv_file(file_path)

# 2. Define algorithms, datasets, and trainers
algorithms = ["SelfTrain", "FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]
datasets = ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]
trainers = [10]  # Specify the number of trainers

# Function to filter data based on Algorithm, Dataset, and Number of Trainers
def filter_data(df, algorithm, dataset, trainers):
    return df[(df["Algorithm"] == algorithm) &
              (df["Dataset"] == dataset) &
              (df["Number of Trainers"].isin(trainers))]

# 3. Plot chart for Accuracy comparison across Algorithms and Datasets
def plot_accuracy_comparison(df, algorithms, datasets, trainers):
    width = 0.15  # Width of each bar
    algorithm_range = np.arange(len(algorithms))  # X positions for bars

    # Track min and max accuracy for y-axis adjustment
    min_accuracy = float('inf')
    max_accuracy = float('-inf')

    # Loop over each dataset and plot accuracy for all algorithms
    for j, dataset in enumerate(datasets):
        accuracy_values = []

        # Gather accuracy data for each algorithm within the current dataset
        for i, algorithm in enumerate(algorithms):
            filtered_df = filter_data(df, algorithm, dataset, trainers)
            avg_accuracy = filtered_df["Average Test Accuracy"].mean()
            accuracy_values.append(avg_accuracy)

            # Update min and max accuracy for y-axis scaling
            if not np.isnan(avg_accuracy):  # Check for non-empty values
                min_accuracy = min(min_accuracy, avg_accuracy)
                max_accuracy = max(max_accuracy, avg_accuracy)

        # Plot the bars for each dataset with a different color
        plt.bar(algorithm_range + j * width, accuracy_values, 
                width=width, label=f"{dataset}")

    # Calculate diff and adjust the y-axis to make diff occupy 70% of the plot's height
    diff = max_accuracy - min_accuracy
    lower_bound = max_accuracy - diff / 0.7

    # Set y-axis limit to make the difference more visible
    plt.ylim(lower_bound, max_accuracy * 1.01)

    # Title and labels
    plt.title("Test Accuracy Comparison across Datasets and Algorithms")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy")

    # Set x-axis labels to algorithms
    plt.xticks(algorithm_range + width * (len(datasets) - 1) / 2, labels=algorithms)

    # Display the legend
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Datasets")
    plt.tight_layout()
    plt.show()

# 4. Call the plotting function
plot_accuracy_comparison(df, algorithms, datasets, trainers)
