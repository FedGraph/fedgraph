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


def extract_nc_scalability_data(logfile, expected_trainers=None):
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    results = []

    csv_sections = re.findall(
        r"CSV FORMAT RESULT:.*?DS,IID,BS,TotalTime\[s\],PureTrainingTime\[s\],CommTime\[s\],FinalAcc\[%\],CommCost\[MB\],PeakMem\[MB\],AvgRoundTime\[s\],ModelSize\[MB\],TotalParams\n(.*?)\n",
        log_content,
        re.DOTALL,
    )

    trainer_matches = re.findall(r"Trainers: (\d+)", log_content)

    for csv_idx, csv_line in enumerate(csv_sections):
        parts = csv_line.strip().split(",")
        if len(parts) >= 12:
            try:
                num_trainers = expected_trainers if expected_trainers else (
                    int(trainer_matches[csv_idx])
                    if csv_idx < len(trainer_matches)
                    else 10
                )

                result = {
                    "Dataset": parts[0],
                    "IID_Beta": float(parts[1]),
                    "Batch_Size": int(parts[2]) if parts[2] != '-1' else -1,
                    "Total_Time": float(parts[3]),
                    "Training_Time": float(parts[4]),
                    "Communication_Time": float(parts[5]),
                    "Final_Accuracy": float(parts[6]),
                    "Communication_Cost": float(parts[7]),
                    "Peak_Memory": float(parts[8]),
                    "Avg_Round_Time": float(parts[9]),
                    "Model_Size": float(parts[10]),
                    "Total_Params": int(float(parts[11])),
                    "Num_Trainers": num_trainers,
                }
                results.append(result)
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(results)


def load_all_nc_logs():
    log_files = ["NC5.log", "NC10.log", "NC15.log", "NC20.log"]
    trainer_counts = [5, 10, 15, 20]

    all_data = []

    for log_file, expected_trainers in zip(log_files, trainer_counts):
        if os.path.exists(log_file):
            df = extract_nc_scalability_data(log_file, expected_trainers)
            if not df.empty:
                df["Num_Trainers"] = expected_trainers
                all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def create_scalability_plot(df):
    if df.empty:
        return

    df_filtered = df[df["IID_Beta"] == 10.0].copy()

    if df_filtered.empty:
        return

    scalability_data = (
        df_filtered.groupby("Num_Trainers")
        .agg(
            {
                "Training_Time": "mean",
                "Communication_Time": "mean",
                "Total_Time": "mean",
                "Final_Accuracy": "mean",
                "Communication_Cost": "mean",
                "Peak_Memory": "mean",
            }
        )
        .reset_index()
    )

    scalability_data = scalability_data.sort_values("Num_Trainers")

    plt.figure(figsize=(12, 8))

    plt.plot(
        scalability_data["Num_Trainers"],
        scalability_data["Training_Time"],
        "o-",
        linewidth=3,
        markersize=8,
        color="#1f77b4",
        label="Training Time",
    )

    plt.plot(
        scalability_data["Num_Trainers"],
        scalability_data["Communication_Time"],
        "s-",
        linewidth=3,
        markersize=8,
        color="#ff7f0e",
        label="Communication Time",
    )

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

    plt.xlabel("Number of Clients", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    plt.title("Federated Learning Scalability Analysis", fontsize=18, fontweight="bold")
    plt.legend(fontsize=14, loc="upper left")
    plt.grid(True, alpha=0.3)

    client_numbers = sorted(scalability_data["Num_Trainers"].unique())
    plt.xticks(client_numbers, fontsize=14)
    plt.yticks(fontsize=14)

    y_max = max(
        scalability_data["Training_Time"].max(),
        scalability_data["Communication_Time"].max(),
    )
    plt.ylim(0, y_max * 1.2)

    plt.tight_layout()
    plt.savefig("federated_learning_scalability.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    scalability_data.to_csv("scalability_analysis.csv", index=False)


def main():
    df = load_all_nc_logs()
    
    if not df.empty:
        create_scalability_plot(df)


if __name__ == "__main__":
    main()