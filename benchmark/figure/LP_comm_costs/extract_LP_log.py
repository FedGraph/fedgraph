#!/usr/bin/env python3
"""
Federated Link Prediction Visualization Tool

This script analyzes log files from federated link prediction experiments
and generates visualizations for AUC, training time, and communication costs.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_lp_data(logfile):
    """
    Extract communication cost data from log files.

    Parameters
    ----------
    logfile : str
        Path to the log file

    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted communication cost metrics
    """
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Extract experiment sections
    experiments = []

    # Try extracting by "Running experiment" format
    exp_sections = re.split(r"-{80}\nRunning experiment \d+/\d+:", log_content)
    if len(exp_sections) > 1:
        experiments = exp_sections[1:]  # Skip the first empty section
    else:
        # Alternative: Try extracting by "The whole process has ended"
        process_sections = re.split(r"The whole process has ended", log_content)
        if len(process_sections) > 1:
            experiments = process_sections[:-1]  # Skip the last empty section

    results = []

    # Process each experiment section
    for i, exp in enumerate(experiments, 1):
        # Extract method/algorithm
        method_match = re.search(r"Method: ([^,\n]+)", exp)
        if not method_match:
            method_match = re.search(r"Running method: ([^,\n]+)", exp)

        method = method_match.group(1).strip() if method_match else f"Method_{i}"

        # Extract countries/datasets
        countries_match = re.search(r"Countries: ([^,\n]+(?:, [^,\n]+)*)", exp)
        if not countries_match:
            countries_match = re.search(r"country_codes: \[(.*?)\]", exp)

        countries = (
            countries_match.group(1).replace("'", "").replace('"', "").strip()
            if countries_match
            else ""
        )

        # For single country experiments, try to extract from file paths
        if not countries:
            country_file_match = re.search(r"data_([A-Z]{2})\.txt", exp)
            if country_file_match:
                countries = country_file_match.group(1)
            else:
                countries = "US"  # Default to US for unknown

        # Extract train time data
        train_time_matches = re.findall(r"train time ([\d.]+)", exp)
        train_time_ms = None
        if train_time_matches:
            # Convert to milliseconds and take average
            train_times = [float(t) * 1000 for t in train_time_matches]
            train_time_ms = np.mean(train_times)

        total_train_time_match = re.search(r"//train_time: ([\d.]+) ms//end", exp)
        if total_train_time_match:
            train_time_ms = float(total_train_time_match.group(1))

        # Extract theoretical and actual comm costs
        theoretical_pretrain_match = re.search(
            r"//Log Theoretical Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        theoretical_train_match = re.search(
            r"//Log Theoretical Train Comm Cost: ([\d.]+) MB //end", exp
        )
        actual_pretrain_match = re.search(
            r"//Log Total Actual Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        actual_train_match = re.search(
            r"//Log Total Actual Train Comm Cost: ([\d.]+) MB //end", exp
        )

        # Extract performance metrics (last occurrence)
        auc_matches = re.findall(r"Test AUC: ([\d.]+)", exp)
        if not auc_matches:
            auc_matches = re.findall(
                r"Predict Day \d+ average auc score: ([\d.]+)", exp
            )

        hit_rate_matches = re.findall(r"Test Hit Rate at \d+: ([\d.]+)", exp)
        if not hit_rate_matches:
            hit_rate_matches = re.findall(r"hit rate: ([\d.]+)", exp)

        auc = float(auc_matches[-1]) if auc_matches else None
        hit_rate = float(hit_rate_matches[-1]) if hit_rate_matches else None

        # Create result record
        result = {
            "Algorithm": method,
            "Dataset": countries,
            "AUC": auc,
            "TrainTime": train_time_ms,
            "Theoretical_Pretrain_MB": float(theoretical_pretrain_match.group(1))
            if theoretical_pretrain_match
            else 0,
            "Theoretical_Train_MB": float(theoretical_train_match.group(1))
            if theoretical_train_match
            else 0,
            "Actual_Pretrain_MB": float(actual_pretrain_match.group(1))
            if actual_pretrain_match
            else 0,
            "Actual_Train_MB": float(actual_train_match.group(1))
            if actual_train_match
            else 0,
            "Hit_Rate": hit_rate,
        }

        # Add embedding communication cost for FedLink, STFL, 4D-FED-GNN+
        algorithms_with_embedding = ["4D-FED-GNN+", "STFL", "FedLink"]

        if method in algorithms_with_embedding:
            # Split countries into list
            country_list = [c.strip() for c in countries.split(",")]

            # Determine number of clients
            num_clients = len(country_list)

            # Use user/item numbers according to your experiments
            if num_clients == 1:
                number_of_users = 114362
                number_of_items = 459912
            elif num_clients == 2:
                number_of_users = 160392
                number_of_items = 620385
            else:
                number_of_users = 160392
                number_of_items = 620385

            hidden_channels = 64  # From config
            float_size = 4  # bytes

            embedding_param_size_MB = (
                (number_of_users + number_of_items)
                * hidden_channels
                * float_size
                / (1024 * 1024)
            )

            global_rounds = 8  # From config

            embedding_comm_MB = (
                embedding_param_size_MB * (1 + num_clients) * global_rounds
            )

            print(
                f"[Info] Adding {embedding_comm_MB:.2f} MB embedding cost for {method} ({countries}) with {global_rounds} rounds."
            )

            # Update theoretical communication cost
            result["Theoretical_Train_MB"] += embedding_comm_MB

        # Calculate totals
        result["Theoretical_Total_MB"] = (
            result["Theoretical_Pretrain_MB"] + result["Theoretical_Train_MB"]
        )
        result["Actual_Total_MB"] = (
            result["Actual_Pretrain_MB"] + result["Actual_Train_MB"]
        )

        results.append(result)

    return pd.DataFrame(results)


def generate_auc_comparison(df, output_file="lp_auc_comparison.pdf"):
    """Generate AUC comparison plot using real data from logs"""
    if df.empty or df["AUC"].isna().all():
        print("No AUC data available to plot")
        return None

    # Filter out rows with missing AUC
    df_filtered = df.dropna(subset=["AUC"])

    # Create a grouped DataFrame
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"]).agg({"AUC": "mean"}).reset_index()
    )

    print(f"Plotting AUC comparison with {len(comparison_data)} data points")

    # Create a large figure
    plt.figure(figsize=(14, 8))

    # Get unique datasets and algorithms
    datasets = comparison_data["Dataset"].unique()
    algorithms = comparison_data["Algorithm"].unique()

    # Set x positions for datasets
    x_positions = np.arange(len(datasets)) * 0.7

    # Calculate width based on number of algorithms
    width = 0.3 / len(algorithms)

    # Define colors for algorithms
    algorithm_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]  # Blue, Orange, Green, Red, Purple

    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]

        # Prepare data in dataset order
        auc_values = []

        # Ensure consistent dataset ordering
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(dataset_row["AUC"].values[0]):
                auc_values.append(dataset_row["AUC"].values[0])
            else:
                auc_values.append(0)

        # Plot AUC values
        plt.bar(
            x_positions + (i - len(algorithms) / 2 + 0.5) * width,  # Position bars
            auc_values,
            width=width,
            label=algo,
            color=algorithm_colors[
                i % len(algorithm_colors)
            ],  # Use color from specified palette
        )

    # Removed plot title
    plt.xlabel("Dataset (Countries)", fontsize=30)
    plt.ylabel("AUC", fontsize=30)
    plt.xticks(x_positions, datasets, rotation=0, fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim(0, 1.0)
    plt.legend(
        # title="Algorithms",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=25,
        #title_fontsize=25,
    )

    # Remove grid lines
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"AUC comparison plot saved to: {output_file}")
    return output_file


def generate_train_time_comparison(df, output_file="lp_train_time_comparison.pdf"):
    """Generate train time comparison plot using real data from logs"""
    if df.empty or df["TrainTime"].isna().all():
        print("No training time data available to plot")
        return None

    # Filter out rows with missing train time
    df_filtered = df.dropna(subset=["TrainTime"])

    # Create a grouped DataFrame
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"])
        .agg({"TrainTime": "mean"})
        .reset_index()
    )

    print(f"Plotting training time comparison with {len(comparison_data)} data points")

    # Create a large figure
    plt.figure(figsize=(14, 8))

    # Get unique datasets and algorithms
    datasets = comparison_data["Dataset"].unique()
    algorithms = comparison_data["Algorithm"].unique()

    # Set x positions for datasets
    x_positions = np.arange(len(datasets)) * 0.7

    # Calculate width based on number of algorithms
    width = 0.3 / len(algorithms)

    # Define colors for algorithms
    algorithm_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]  # Blue, Orange, Green, Red, Purple

    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]

        # Prepare data in dataset order
        train_time_values = []

        # Ensure consistent dataset ordering
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(
                dataset_row["TrainTime"].values[0]
            ):
                # Convert ms to s
                train_time_values.append(dataset_row["TrainTime"].values[0] / 1000)
            else:
                train_time_values.append(0)

        # Plot train time values
        plt.bar(
            x_positions + (i - len(algorithms) / 2 + 0.5) * width,  # Position bars
            train_time_values,
            width=width,
            label=algo,
            color=algorithm_colors[
                i % len(algorithm_colors)
            ],  # Use color from specified palette
        )

    # Removed plot title
    plt.xlabel("Dataset (Countries)", fontsize=30)
    plt.ylabel("Train Time (s)", fontsize=28)
    plt.xticks(x_positions, datasets, rotation=0, fontsize=30)
    plt.yticks(fontsize=28)
    plt.legend(
        # title="Algorithms",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=25,
        #title_fontsize=25,
    )

    # Remove grid lines
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Train time comparison plot saved to: {output_file}")
    return output_file


def generate_comm_cost_comparison(df, output_file="lp_comm_cost_comparison.pdf"):
    """Generate communication cost comparison plot with each algorithm paired with its theoretical value."""
    if df.empty or (
        df["Actual_Total_MB"].isna().all() and df["Theoretical_Total_MB"].isna().all()
    ):
        print("No communication cost data available to plot")
        return None

    # Filter out rows with missing comm cost
    df_filtered = df.dropna(
        subset=["Actual_Total_MB", "Theoretical_Total_MB"], how="all"
    )

    # Convert MB to GB for plotting
    df_filtered = df_filtered.copy()
    df_filtered["Theoretical_Total_GB"] = df_filtered["Theoretical_Total_MB"] / 1024
    df_filtered["Actual_Total_GB"] = df_filtered["Actual_Total_MB"] / 1024
    # Create a grouped DataFrame
    comparison_data = (
        df_filtered.groupby(["Dataset", "Algorithm"])
        .agg({"Theoretical_Total_GB": "mean", "Actual_Total_GB": "mean"})
        .reset_index()
    )

    print(
        f"Plotting communication cost comparison with {len(comparison_data)} data points"
    )

    # Create a large figure
    plt.figure(figsize=(14, 8))

    # Get unique datasets and algorithms
    datasets = comparison_data["Dataset"].unique()
    algorithms = comparison_data["Algorithm"].unique()

    # Set x positions for datasets
    x_positions = np.arange(len(datasets))

    # Total number of bars: for each algorithm 2 bars (Actual + Theoretical)
    total_bars = len(algorithms) * 2
    width = 0.8 / total_bars

    # Define colors
    actual_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    theoretical_color = "#aec7e8"  # Light blue for all theoretical

    current_pos = 0

    for i, algo in enumerate(algorithms):
        algo_data = comparison_data[comparison_data["Algorithm"] == algo]

        # Actual values (in GB)
        actual_values = []
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(
                dataset_row["Actual_Total_GB"].values[0]
            ):
                actual_values.append(dataset_row["Actual_Total_GB"].values[0])
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

        # Theoretical values (in GB)
        theoretical_values = []
        for dataset in datasets:
            dataset_row = algo_data[algo_data["Dataset"] == dataset]
            if not dataset_row.empty and not pd.isna(
                dataset_row["Theoretical_Total_GB"].values[0]
            ):
                theoretical_values.append(dataset_row["Theoretical_Total_GB"].values[0])
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

    # Removed plot title
    plt.xlabel("Dataset (Countries)", fontsize=30)
    plt.ylabel("Communication Cost (GB)", fontsize=28)
    plt.xticks(x_positions, datasets, rotation=0, fontsize=30)
    plt.yticks(fontsize=28)
    plt.legend(
        # title="Algorithms",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=18,
        #title_fontsize=25,
    )

    # Remove grid lines
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Communication cost plot saved to: {output_file}")
    return output_file


def process_all_log_files(log_folder):
    """Process all log files in the given folder"""
    # Find all log files in the folder
    log_files = glob.glob(os.path.join(log_folder, "*.log"))

    if not log_files:
        print(f"No log files found in {log_folder}")
        return pd.DataFrame()

    print(f"Found {len(log_files)} log files to process")

    # Process each log file and combine results
    all_results = []

    for log_file in log_files:
        print(f"Processing log file: {log_file}")
        df = extract_lp_data(log_file)
        if not df.empty:
            all_results.append(df)

    # Combine all results
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    import sys

    # Check if a log file or folder was provided as a command line argument
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

        if os.path.isfile(log_path):
            # Process a single log file
            print(f"Processing single log file: {log_path}")
            df = extract_lp_data(log_path)
            print(f"Extracted {len(df)} data points from log file")
        elif os.path.isdir(log_path):
            # Process all log files in the given folder
            print(f"Processing log files in folder: {log_path}")
            df = process_all_log_files(log_path)
            print(f"Extracted {len(df)} total data points from log files")
        else:
            print(f"Error: {log_path} is neither a file nor a directory")
            sys.exit(1)
    else:
        # No command line argument, look for log files in the current directory
        print("No log file specified, looking for log files in current directory")
        df = process_all_log_files(os.getcwd())
        print(f"Extracted {len(df)} total data points from log files")

    # Save the raw data
    if not df.empty:
        df.to_csv("lp_data_raw.csv", index=False)
        print("Raw data saved to lp_data_raw.csv")

        # Print summary of extracted data
        print("\nSummary of extracted data:")
        print(f"Algorithms: {df['Algorithm'].unique().tolist()}")
        print(f"Datasets: {df['Dataset'].unique().tolist()}")
        print(f"Total data points: {len(df)}")

        # Generate plots
        generate_auc_comparison(df, "lp_auc_comparison.pdf")
        generate_train_time_comparison(df, "lp_train_time_comparison.pdf")
        generate_comm_cost_comparison(df, "lp_comm_cost_comparison.pdf")
    else:
        print("No data was extracted from log files")


def print_theoretical_comm_cost(df):
    print("\n== Current Theoretical Communication Costs ==")
    for idx, row in df.iterrows():
        print(
            f"Algorithm: {row['Algorithm']}, Dataset: {row['Dataset']}, "
            f"Theoretical_Pretrain_MB: {row['Theoretical_Pretrain_MB']:.2f} MB, "
            f"Theoretical_Train_MB: {row['Theoretical_Train_MB']:.2f} MB, "
            f"Theoretical_Total_MB: {row['Theoretical_Total_MB']:.2f} MB"
        )


print_theoretical_comm_cost(df)
