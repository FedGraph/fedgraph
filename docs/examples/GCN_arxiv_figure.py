import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a DataFrame
file_path = "GCN_arxiv.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Group by 'Distribution Type' and 'IID Beta', then calculate the mean for each group
numeric_cols = df.select_dtypes(include="number").columns
df = df.drop(columns=["IID Beta", "Distribution Type"])
grouped_df = (
    df.groupby(["Distribution Type", "IID Beta"])[numeric_cols].mean().reset_index()
)


# Plot settings


def plot_metric_vs_setting(metric, setting, ylabel, title):
    plt.figure(figsize=(10, 6))
    for dist_type in grouped_df["Distribution Type"].unique():
        subset = grouped_df[grouped_df["Distribution Type"] == dist_type]
        plt.plot(subset[setting], subset[metric], label=dist_type)
    plt.xlabel(setting)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Distribution Type")
    plt.grid(True)
    plt.show()


# Plot for Training Time vs IID Beta
plot_metric_vs_setting(
    "Train Time", "IID Beta", "Training Time", "Training Time vs IID Beta"
)

# Plot for Test Accuracy vs IID Beta
plot_metric_vs_setting(
    "Average Test Accuracy", "IID Beta", "Test Accuracy", "Test Accuracy vs IID Beta"
)

# Plot for Communication Cost vs IID Beta
plot_metric_vs_setting(
    "Pretrain Network Large1",
    "IID Beta",
    "Communication Cost",
    "Communication Cost vs IID Beta",
)

# Plot for Training Time vs Distribution Type
plot_metric_vs_setting(
    "Train Time",
    "Distribution Type",
    "Training Time",
    "Training Time vs Distribution Type",
)

# Plot for Test Accuracy vs Distribution Type
plot_metric_vs_setting(
    "Average Test Accuracy",
    "Distribution Type",
    "Test Accuracy",
    "Test Accuracy vs Distribution Type",
)

# Plot for Communication Cost vs Distribution Type
plot_metric_vs_setting(
    "Pretrain Network Large1",
    "Distribution Type",
    "Communication Cost",
    "Communication Cost vs Distribution Type",
)
