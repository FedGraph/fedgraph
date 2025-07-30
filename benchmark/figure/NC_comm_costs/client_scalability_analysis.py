#!/usr/bin/env python3

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
sns.set_context("talk")


def extract_nc_scalability_data(logfile):
    """Extract training and communication time data from NC log files"""
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    results = []

    # Find CSV FORMAT RESULT sections
    csv_sections = re.findall(
        r"CSV FORMAT RESULT:.*?DS,IID,BS,Time\[s\],FinalAcc\[%\],CompTime\[s\],CommCost\[MB\],PeakMem\[MB\],AvgRoundTime\[s\],ModelSize\[MB\],TotalParams\n(.*?)\n",
        log_content,
        re.DOTALL,
    )

    # Extract number of trainers from experiment configuration
    trainer_matches = re.findall(r"Trainers: (\d+)", log_content)

    for csv_idx, csv_line in enumerate(csv_sections):
        parts = csv_line.strip().split(",")
        if len(parts) >= 11:
            try:
                # Get number of trainers for this experiment
                num_trainers = (
                    int(trainer_matches[csv_idx])
                    if csv_idx < len(trainer_matches)
                    else 10
                )

                result = {
                    "Dataset": parts[0],
                    "IID_Beta": float(parts[1]),
                    "Batch_Size": int(parts[2]),
                    "Total_Time": float(parts[3]),
                    "Final_Accuracy": float(parts[4]),
                    "Training_Time": float(parts[5]),  # CompTime[s]
                    "Communication_Cost": float(
                        parts[6]
                    ),  # CommCost[MB] - will convert to time
                    "Peak_Memory": float(parts[7]),
                    "Avg_Round_Time": float(parts[8]),
                    "Model_Size": float(parts[9]),
                    "Total_Params": int(float(parts[10])),
                    "Num_Trainers": num_trainers,
                }
                results.append(result)
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(results)


def estimate_communication_time(comm_cost_mb, num_trainers):
    """Estimate communication time based on communication cost and network assumptions"""
    # Assume network bandwidth: 100 Mbps = 12.5 MB/s
    # This is a reasonable assumption for federated learning scenarios
    network_bandwidth_mbps = 100 / 8  # Convert to MB/s

    # Communication time = Total communication cost / bandwidth
    comm_time = comm_cost_mb / network_bandwidth_mbps

    return comm_time


def load_all_nc_logs():
    """Load data from all NC log files"""
    log_files = ["NC.log", "NC5.log", "NC10.log", "NC20.log", "NC40.log"]
    trainer_counts = [10, 5, 10, 20, 40]  # Default mapping

    all_data = []

    for log_file, default_trainers in zip(log_files, trainer_counts):
        if os.path.exists(log_file):
            df = extract_nc_scalability_data(log_file)
            if not df.empty:
                # If trainer count not detected, use default
                if "Num_Trainers" not in df.columns or df["Num_Trainers"].isna().all():
                    df["Num_Trainers"] = default_trainers
                all_data.append(df)
                print(
                    f"Loaded {len(df)} records from {log_file} (Trainers: {default_trainers})"
                )

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No NC log files found")
        return pd.DataFrame()


def create_scalability_plot(df):
    """Create scalability plot showing training time and communication time vs number of clients"""

    if df.empty:
        print("No data available for plotting")
        return

    # Filter for IID_Beta = 10.0 (as specified in your benchmark)
    df_filtered = df[df["IID_Beta"] == 10.0].copy()

    if df_filtered.empty:
        print("No data found for IID_Beta = 10.0")
        return

    # Add estimated communication time
    df_filtered["Communication_Time"] = df_filtered.apply(
        lambda row: estimate_communication_time(
            row["Communication_Cost"], row["Num_Trainers"]
        ),
        axis=1,
    )

    # Group by number of trainers and calculate average times
    scalability_data = (
        df_filtered.groupby("Num_Trainers")
        .agg(
            {
                "Training_Time": "mean",
                "Communication_Time": "mean",
                "Total_Time": "mean",
            }
        )
        .reset_index()
    )

    # Sort by number of trainers
    scalability_data = scalability_data.sort_values("Num_Trainers")

    print("Scalability Data Summary:")
    print(scalability_data)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot training time
    plt.plot(
        scalability_data["Num_Trainers"],
        scalability_data["Training_Time"],
        "o-",
        linewidth=3,
        markersize=8,
        color="#1f77b4",
        label="Training Time",
    )

    # Plot communication time
    plt.plot(
        scalability_data["Num_Trainers"],
        scalability_data["Communication_Time"],
        "s-",
        linewidth=3,
        markersize=8,
        color="#ff7f0e",
        label="Communication Time",
    )

    # Add value labels on points
    for _, row in scalability_data.iterrows():
        plt.annotate(
            f'{row["Training_Time"]:.1f}s',
            (row["Num_Trainers"], row["Training_Time"]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=10,
            color="#1f77b4",
        )

        plt.annotate(
            f'{row["Communication_Time"]:.1f}s',
            (row["Num_Trainers"], row["Communication_Time"]),
            textcoords="offset points",
            xytext=(0, -25),
            ha="center",
            fontsize=10,
            color="#ff7f0e",
        )

    # Customize plot
    plt.xlabel("Number of Clients", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    plt.title("Federated Learning Scalability Analysis", fontsize=18, fontweight="bold")
    plt.legend(fontsize=14, loc="upper left")
    plt.grid(True, alpha=0.3)

    # Set x-axis to show all client numbers
    client_numbers = sorted(scalability_data["Num_Trainers"].unique())
    plt.xticks(client_numbers, fontsize=14)
    plt.yticks(fontsize=14)

    # Add some padding to y-axis
    y_max = max(
        scalability_data["Training_Time"].max(),
        scalability_data["Communication_Time"].max(),
    )
    plt.ylim(0, y_max * 1.2)

    plt.tight_layout()
    plt.savefig("federated_learning_scalability.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generated: federated_learning_scalability.pdf")

    # Create additional analysis table
    scalability_data["Training_Growth"] = (
        scalability_data["Training_Time"] / scalability_data["Training_Time"].iloc[0]
    )
    scalability_data["Communication_Growth"] = (
        scalability_data["Communication_Time"]
        / scalability_data["Communication_Time"].iloc[0]
    )

    print(f"\n{'='*60}")
    print("SCALABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(
        f"{'Clients':<8} {'Train Time':<12} {'Comm Time':<12} {'Train Growth':<13} {'Comm Growth':<12}"
    )
    print("-" * 60)

    for _, row in scalability_data.iterrows():
        print(
            f"{row['Num_Trainers']:<8.0f} "
            f"{row['Training_Time']:<12.1f} "
            f"{row['Communication_Time']:<12.1f} "
            f"{row['Training_Growth']:<13.2f}x "
            f"{row['Communication_Growth']:<12.2f}x"
        )

    # Save detailed results
    scalability_data.to_csv("scalability_analysis.csv", index=False)
    print(f"\nDetailed results saved to: scalability_analysis.csv")


def main():
    """Main function to analyze federated learning scalability"""
    print("Loading federated learning scalability data...")

    # Load all NC log data
    df = load_all_nc_logs()

    if df.empty:
        print("No data found. Please check if NC log files exist:")
        print("- NC.log, NC5.log, NC10.log, NC20.log, NC40.log")
        return

    print(f"\nLoaded data summary:")
    print(f"Total records: {len(df)}")
    print(f"Client counts: {sorted(df['Num_Trainers'].unique())}")
    print(f"Datasets: {list(df['Dataset'].unique())}")
    print(f"IID Betas: {sorted(df['IID_Beta'].unique())}")

    # Create scalability analysis
    print("\nGenerating scalability analysis...")
    create_scalability_plot(df)

    print(f"\nScalability analysis completed!")
    print("Generated files:")
    print("- federated_learning_scalability.pdf")
    print("- scalability_analysis.csv")


if __name__ == "__main__":
    main()
