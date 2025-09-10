#!/usr/bin/env python3
"""
Federated Graph Classification Visualization Tool

This script analyzes log files from federated graph classification experiments
and generates visualizations for accuracy, training time, and communication costs.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_gc_data(logfile):
    """Extract data from Graph Classification log files"""
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Extract both standard and informal experiment sections
    formal_experiments = re.split(r"-{80}\nRunning experiment \d+/\d+:", log_content)
    informal_runs = re.findall(
        r"Running ([A-Za-z0-9+_]+) \.\.\..*?(?=Running|\Z)", log_content, re.DOTALL
    )

    results = []

    # Process formal experiment sections
    for exp in formal_experiments[1:]:  # Skip first empty section
        # Extract basic experiment info
        algo_match = re.search(r"Algorithm: ([A-Za-z0-9+_]+)", exp)
        dataset_match = re.search(r"Dataset: ([A-Z0-9-]+)", exp)
        trainers_match = re.search(r"Trainers: (\d+)", exp)

        if not (algo_match and dataset_match):
            continue

        algorithm = algo_match.group(1).strip()
        dataset = dataset_match.group(1).strip()
        trainers = int(trainers_match.group(1)) if trainers_match else 10

        # Filter datasets and algorithms
        if dataset not in ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]:
            continue

        if algorithm not in ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]:
            continue

        # Extract metrics
        result = extract_metrics(exp, algorithm, dataset, trainers)
        if result:
            results.append(result)

    # Process informal runs
    for run in informal_runs:
        # Extract algorithm from the "Running X ..." line
        algo_line = re.search(r"Running ([A-Za-z0-9+_]+) \.\.\.", run)
        if not algo_line:
            continue

        algorithm = algo_line.group(1).strip()

        # Skip if not in target algorithms
        if algorithm not in ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]:
            continue

        # Try to extract dataset from dataset-related lines
        dataset_match = re.search(r"Dataset: ([A-Z0-9-]+)", run)
        if not dataset_match:
            # Look for trainer dataset name patterns
            dataset_trainer_matches = re.findall(
                r"dataset_trainer_name: \d+-([A-Z0-9-]+)", run
            )
            if dataset_trainer_matches:
                dataset = dataset_trainer_matches[0]
            else:
                continue
        else:
            dataset = dataset_match.group(1).strip()

        # Filter datasets
        if dataset not in ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]:
            continue

        # Extract trainers count
        trainers_match = re.search(r"Trainers: (\d+)", run)
        trainers = int(trainers_match.group(1)) if trainers_match else 10

        # Extract metrics
        result = extract_metrics(run, algorithm, dataset, trainers)
        if result:
            results.append(result)

    return pd.DataFrame(results)


def extract_metrics(exp_text, algorithm, dataset, trainers):
    """Extract metrics from experiment text"""
    # Extract accuracy
    accuracy_match = re.search(r"Average test accuracy: ([\d.]+)", exp_text)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    # Extract train time
    train_time_match = re.search(r"//train_time: ([\d.]+) ms//end", exp_text)
    train_time = float(train_time_match.group(1)) if train_time_match else None

    # Extract theoretical comm costs
    theoretical_pretrain = re.findall(
        r"//Log Theoretical Pretrain Comm Cost: ([\d.]+) MB //end", exp_text
    )
    theoretical_train = re.findall(
        r"//Log Theoretical Train Comm Cost: ([\d.]+) MB //end", exp_text
    )

    # Extract actual comm costs
    actual_pretrain_match = re.search(
        r"//Log Total Actual Pretrain Comm Cost: ([\d.]+) MB //end", exp_text
    )
    actual_train_match = re.search(
        r"//Log Total Actual Train Comm Cost: ([\d.]+) MB //end", exp_text
    )

    # Check if we have at least some valid data
    if not (
        accuracy
        or train_time
        or theoretical_pretrain
        or theoretical_train
        or actual_pretrain_match
        or actual_train_match
    ):
        return None

    # Create result record
    result = {
        "Algorithm": algorithm,
        "Dataset": dataset,
        "Trainers": trainers,
        "Accuracy": accuracy,
        "Train_Time_ms": train_time,
        "Theoretical_Pretrain_MB": float(theoretical_pretrain[-1])
        if theoretical_pretrain
        else 0,
        "Theoretical_Train_MB": float(theoretical_train[-1])
        if theoretical_train
        else 0,
        "Actual_Pretrain_MB": float(actual_pretrain_match.group(1))
        if actual_pretrain_match
        else None,
        "Actual_Train_MB": float(actual_train_match.group(1))
        if actual_train_match
        else None,
    }

    # Calculate totals
    result["Theoretical_Total_MB"] = (
        result["Theoretical_Pretrain_MB"] + result["Theoretical_Train_MB"]
    )

    if (
        result["Actual_Pretrain_MB"] is not None
        and result["Actual_Train_MB"] is not None
    ):
        result["Actual_Total_MB"] = (
            result["Actual_Pretrain_MB"] + result["Actual_Train_MB"]
        )

    return result


def generate_accuracy_comparison(df, output_file="gc_accuracy_comparison.pdf"):
    if df.empty or df["Accuracy"].isna().all():
        print("No accuracy data available to plot")
        return None
    df_filtered = df.dropna(subset=["Accuracy"])
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"])
        .agg({"Accuracy": "mean"})
        .reset_index()
    )
    print(f"Plotting accuracy comparison with {len(comparison_data)} data points")
    plt.figure(figsize=(14, 8))
    datasets = sorted(
        comparison_data["Dataset"].unique(),
        key=lambda x: ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"].index(x)
        if x in ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]
        else 999,
    )
    algorithms = sorted(
        comparison_data["Algorithm"].unique(),
        key=lambda x: ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"].index(x)
        if x in ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]
        else 999,
    )
    x_positions = np.arange(len(datasets))
    width = 0.8 / len(algorithms)
    actual_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]
        accuracy_values = []
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(dataset_row["Accuracy"].values[0]):
                accuracy_values.append(dataset_row["Accuracy"].values[0])
            else:
                accuracy_values.append(0)
        plt.bar(
            x_positions + (i - len(algorithms) / 2 + 0.5) * width,
            accuracy_values,
            width=width,
            label=algo,
            color=actual_colors[i % len(actual_colors)],
        )
    # plt.title("Accuracy Comparison", fontsize=30)
    plt.xlabel("Dataset", fontsize=30)
    plt.ylabel("Accuracy", fontsize=30)
    plt.xticks(x_positions, datasets, rotation=30, fontsize=20)
    plt.yticks(fontsize=30)
    plt.ylim(0, 1.0)
    plt.legend(
        # title="Algorithms",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=22,
        # title_fontsize=25,
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Accuracy comparison plot saved to: {output_file}")
    return output_file


def generate_train_time_comparison(df, output_file="gc_train_time_comparison.pdf"):
    if df.empty or df["Train_Time_ms"].isna().all():
        print("No training time data available to plot")
        return None
    df_filtered = df.dropna(subset=["Train_Time_ms"])
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"])
        .agg({"Train_Time_ms": "mean"})
        .reset_index()
    )
    print(f"Plotting training time comparison with {len(comparison_data)} data points")
    plt.figure(figsize=(14, 8))
    datasets = sorted(
        comparison_data["Dataset"].unique(),
        key=lambda x: ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"].index(x)
        if x in ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]
        else 999,
    )
    algorithms = sorted(
        comparison_data["Algorithm"].unique(),
        key=lambda x: ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"].index(x)
        if x in ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]
        else 999,
    )
    x_positions = np.arange(len(datasets))
    width = 0.8 / len(algorithms)
    actual_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]
        time_values = []
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(
                dataset_row["Train_Time_ms"].values[0]
            ):
                time_values.append(dataset_row["Train_Time_ms"].values[0] / 1000.0)
            else:
                time_values.append(0)
        plt.bar(
            x_positions + (i - len(algorithms) / 2 + 0.5) * width,
            time_values,
            width=width,
            label=algo,
            color=actual_colors[i % len(actual_colors)],
        )
    # plt.title("Training Time Comparison", fontsize=30)
    plt.xlabel("Dataset", fontsize=30)
    plt.ylabel("Training Time (s)", fontsize=28)
    plt.xticks(x_positions, datasets, rotation=30, fontsize=20)
    plt.yticks(fontsize=28)
    plt.legend(
        # title="Algorithms",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=22,
        # title_fontsize=25,
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Training time comparison plot saved to: {output_file}")
    return output_file


def generate_comm_cost_comparison(df, output_file="gc_comm_cost_comparison.pdf"):
    """Generate communication cost plot with datasets on x-axis and algorithms paired with theoretical values, styled like LP visualization."""
    if df.empty or (
        df["Actual_Train_MB"].isna().all() and df["Theoretical_Train_MB"].isna().all()
    ):
        print("No communication cost data available to plot")
        return None

    # Filter valid data
    df_filtered = df.dropna(
        subset=["Actual_Train_MB", "Theoretical_Train_MB"], how="all"
    )

    # Group data
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"])
        .agg({"Theoretical_Train_MB": "mean", "Actual_Train_MB": "mean"})
        .reset_index()
    )

    print(
        f"Plotting communication cost comparison with {len(comparison_data)} data points"
    )

    # Create plot
    plt.figure(figsize=(14, 8))

    # Datasets and algorithms
    datasets = sorted(
        comparison_data["Dataset"].unique(),
        key=lambda x: ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"].index(x)
        if x in ["IMDB-BINARY", "IMDB-MULTI", "MUTAG", "BZR", "COX2"]
        else 999,
    )

    algorithms = sorted(
        comparison_data["Algorithm"].unique(),
        key=lambda x: ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"].index(x)
        if x in ["FedAvg", "GCFL", "GCFL+", "GCFL+dWs"]
        else 999,
    )

    # X-axis setup
    x_positions = np.arange(len(datasets))

    # Bar setup
    total_bars = len(algorithms) * 2  # each algorithm has 2 bars: actual + theoretical
    width = 0.8 / total_bars

    # Colors
    actual_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    theoretical_color = "#aec7e8"

    current_pos = 0

    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]

        # Actual values
        actual_values = []
        for dataset in datasets:
            row = algo_data[(algo_data["Dataset"] == dataset)]
            if not row.empty and not pd.isna(row["Actual_Train_MB"].values[0]):
                actual_values.append(row["Actual_Train_MB"].values[0])
            else:
                actual_values.append(0)

        bar_pos_actual = x_positions + (current_pos - total_bars / 2 + 0.5) * width
        plt.bar(
            bar_pos_actual,
            actual_values,
            width=width,
            label=f"{algo} Actual",
            color=actual_colors[i % len(actual_colors)],
        )
        current_pos += 1

        # Theoretical values
        theoretical_values = []
        for dataset in datasets:
            row = algo_data[(algo_data["Dataset"] == dataset)]
            if not row.empty and not pd.isna(row["Theoretical_Train_MB"].values[0]):
                theoretical_values.append(row["Theoretical_Train_MB"].values[0])
            else:
                theoretical_values.append(0)

        bar_pos_theo = x_positions + (current_pos - total_bars / 2 + 0.5) * width
        plt.bar(
            bar_pos_theo,
            theoretical_values,
            width=width,
            label=f"{algo} Theoretical",
            color=theoretical_color,
        )
        current_pos += 1

    # Plot settings
    # plt.title("Communication Cost Comparison", fontsize=30)
    plt.xlabel("Dataset", fontsize=30)
    plt.ylabel("Communication Cost (MB)", fontsize=28)
    plt.xticks(x_positions, datasets, rotation=30, fontsize=20)
    plt.yticks(fontsize=28)
    plt.legend(
        # title="Legend",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=18,
        # title_fontsize=25,
    )
    plt.grid(False)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Communication cost plot saved to: {output_file}")
    return output_file


def process_all_log_files(log_folder):
    """Process all log files in a folder"""
    # Find all log files
    log_files = glob.glob(os.path.join(log_folder, "*.log"))

    if not log_files:
        print(f"No log files found in {log_folder}")
        return pd.DataFrame()

    print(f"Found {len(log_files)} log files to process")

    # Process each log file
    all_results = []

    for log_file in log_files:
        print(f"Processing log file: {log_file}")
        df = extract_gc_data(log_file)
        if not df.empty:
            all_results.append(df)

    # Combine results
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    import sys

    # Process command line arguments or default to current directory
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

        if os.path.isfile(log_path):
            print(f"Processing single log file: {log_path}")
            df = extract_gc_data(log_path)
            print(f"Extracted {len(df)} data points from log file")
        elif os.path.isdir(log_path):
            print(f"Processing log files in folder: {log_path}")
            df = process_all_log_files(log_path)
            print(f"Extracted {len(df)} total data points from log files")
        else:
            print(f"Error: {log_path} is neither a file nor a directory")
            sys.exit(1)
    else:
        # Look for GC.log in current directory
        default_log = "GC.log"
        if os.path.exists(default_log):
            print(f"Processing default log file: {default_log}")
            df = extract_gc_data(default_log)
            print(f"Extracted {len(df)} data points from log file")
        else:
            print(
                f"Default log file {default_log} not found. Looking for log files in current directory"
            )
            df = process_all_log_files(os.getcwd())
            print(f"Extracted {len(df)} total data points from log files")

    # Save and visualize data
    if not df.empty:
        df.to_csv("gc_data_raw.csv", index=False)
        print("Raw data saved to gc_data_raw.csv")

        # Print summary
        print("\nSummary of extracted data:")
        print(f"Algorithms: {df['Algorithm'].unique().tolist()}")
        print(f"Datasets: {df['Dataset'].unique().tolist()}")
        print(f"Total data points: {len(df)}")

        # Generate plots
        generate_accuracy_comparison(df, "gc_accuracy_comparison.pdf")
        generate_train_time_comparison(df, "gc_train_time_comparison.pdf")
        generate_comm_cost_comparison(df, "gc_comm_cost_comparison.pdf")
    else:
        print("No data was extracted from log files")
