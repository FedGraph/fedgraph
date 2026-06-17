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


def extract_nc_data(logfile):
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()
    exp_sections = re.findall(
        r"Running experiment \d+/\d+:.*?(?=Running experiment|\Z)",
        log_content,
        re.DOTALL,
    )
    if not exp_sections:
        exp_sections = re.findall(
            r"-{80}\nRunning experiment \d+/\d+:.*?(?=-{80}|\Z)", log_content, re.DOTALL
        )
    if not exp_sections:
        exp_sections = re.findall(
            r"Dataset: [a-zA-Z0-9-]+, Trainers: \d+, Distribution: [a-zA-Z0-9-]+, IID Beta: [\d.]+.*?(?=Dataset:|\Z)",
            log_content,
            re.DOTALL,
        )
    results = []
    for exp in exp_sections:
        dataset_match = re.search(r"Dataset: ([a-zA-Z0-9-]+)", exp)
        trainers_match = re.search(r"Trainers: (\d+)", exp)
        iid_beta_match = re.search(r"IID Beta: ([\d.]+)", exp)
        if not (dataset_match and iid_beta_match):
            continue
        dataset = dataset_match.group(1).strip()
        trainers = int(trainers_match.group(1)) if trainers_match else 10
        iid_beta = float(iid_beta_match.group(1))
        algo_match = re.search(r"method': '([A-Za-z0-9+_]+)'", exp)
        if not algo_match:
            algo_match = re.search(r"Changing method to ([A-Za-z0-9+_]+)", exp)
        algorithm = algo_match.group(1).strip() if algo_match else "FedAvg"
        if dataset not in ["cora", "citeseer", "pubmed"]:  # , "ogbn-arxiv"
            continue
        result = extract_metrics(exp, algorithm, dataset, trainers, iid_beta)
        if result:
            results.append(result)
    return pd.DataFrame(results)


def extract_metrics(exp_text, algorithm, dataset, trainers, iid_beta):
    final_accuracy_match = re.search(r"Average test accuracy, ([\d.]+)", exp_text)
    if not final_accuracy_match:
        round_accuracies = re.findall(
            r"Round \d+: Global Test Accuracy = ([\d.]+)", exp_text
        )
        accuracy = float(round_accuracies[-1]) if round_accuracies else None
    else:
        accuracy = float(final_accuracy_match.group(1))
    train_time_match = re.search(r"//train_time: ([\d.]+) ms//end", exp_text)
    train_time_ms = float(train_time_match.group(1)) if train_time_match else None
    train_time_s = train_time_ms / 1000.0 if train_time_ms is not None else None
    theoretical_pretrain = re.findall(
        r"//Log Theoretical Pretrain Comm Cost: ([\d.]+) MB //end", exp_text
    )
    theoretical_train = re.findall(
        r"//Log Theoretical Train Comm Cost: ([\d.]+) MB //end", exp_text
    )
    actual_pretrain_match = re.search(
        r"//Log Total Actual Pretrain Comm Cost: ([\d.]+) MB //end", exp_text
    )
    actual_train_match = re.search(
        r"//Log Total Actual Train Comm Cost: ([\d.]+) MB //end", exp_text
    )
    if not (
        accuracy
        or train_time
        or theoretical_pretrain
        or theoretical_train
        or actual_pretrain_match
        or actual_train_match
    ):
        return None
    result = {
        "Algorithm": algorithm,
        "Dataset": dataset,
        "Trainers": trainers,
        "IID_Beta": iid_beta,
        "Accuracy": accuracy,
        "Train_Time_ms": train_time_ms,
        "Train_Time_s": train_time_s,
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


def plot_metric(df, metric, ylabel, filename_prefix):
    datasets = ["cora", "citeseer", "pubmed"]  # , "ogbn-arxiv"
    algorithms = ["FedAvg", "FedGCN"]
    colors = {"FedAvg": "#1f77b4", "FedGCN": "#ff7f0e"}
    target_betas = [10000.0, 100.0, 10.0]
    for beta in target_betas:
        plt.figure(figsize=(12, 6))
        df_beta = df[df["IID_Beta"] == beta]
        x_positions = np.arange(len(datasets))
        width = 0.35
        for idx, algo in enumerate(algorithms):
            df_algo = df_beta[df_beta["Algorithm"].str.lower() == algo.lower()]
            values = []
            for dataset in datasets:
                temp = df_algo[df_algo["Dataset"] == dataset]
                if not temp.empty and not pd.isna(temp[metric].values[0]):
                    val = temp[metric].values[0]
                    values.append(val)
                else:
                    values.append(0)
            plt.bar(
                x_positions + idx * width,
                values,
                width=width,
                label=algo,
                color=colors[algo],
            )
        # plt.title(f"{ylabel} (IID Beta={beta})", fontsize=26)
        # plt.xlabel("Dataset", fontsize=26)
        plt.ylabel(ylabel, fontsize=24)
        pretty_names = ["Cora", "Citeseer", "Pubmed"]
        plt.xticks(x_positions + width / 2, pretty_names, rotation=0, fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=24,
        )
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_beta{int(beta)}.pdf", dpi=300)
        plt.close()


def plot_comm_cost(df):
    datasets = ["cora", "citeseer", "pubmed"]  # , "ogbn-arxiv"
    algorithms = ["FedAvg", "FedGCN"]
    actual_colors = {"FedAvg": "#1f77b4", "FedGCN": "#ff7f0e"}
    theoretical_colors = {
        "FedAvg": "#aec7e8",
        "FedGCN_Pretrain": "#c5b0d5",
        "FedGCN_Train": "#98df8a",
    }
    pretrain_colors_actual = "#2ca02c"
    target_betas = [10000.0, 100.0, 10.0]

    for beta in target_betas:
        plt.figure(figsize=(12, 6))
        df_beta = df[df["IID_Beta"] == beta]
        x_positions = np.arange(len(datasets))
        width = 0.18

        for d_idx, dataset in enumerate(datasets):
            xpos_base = x_positions[d_idx]
            for a_idx, algo in enumerate(algorithms):
                df_algo = df_beta[
                    (df_beta["Algorithm"].str.lower() == algo.lower())
                    & (df_beta["Dataset"] == dataset)
                ]
                if not df_algo.empty:
                    pretrain_actual = (
                        df_algo["Actual_Pretrain_MB"].values[0]
                        if not pd.isna(df_algo["Actual_Pretrain_MB"].values[0])
                        else 0
                    )
                    train_actual = (
                        df_algo["Actual_Train_MB"].values[0]
                        if not pd.isna(df_algo["Actual_Train_MB"].values[0])
                        else 0
                    )
                    pretrain_theo = (
                        df_algo["Theoretical_Pretrain_MB"].values[0]
                        if not pd.isna(df_algo["Theoretical_Pretrain_MB"].values[0])
                        else 0
                    )
                    train_theo = (
                        df_algo["Theoretical_Train_MB"].values[0]
                        if not pd.isna(df_algo["Theoretical_Train_MB"].values[0])
                        else 0
                    )
                else:
                    pretrain_actual, train_actual, pretrain_theo, train_theo = (
                        0,
                        0,
                        0,
                        0,
                    )

                if algo == "FedAvg":
                    xpos_actual = xpos_base - 1.5 * width
                    xpos_theo = xpos_base - 0.5 * width
                    plt.bar(
                        xpos_actual,
                        train_actual,
                        width=width,
                        color=actual_colors[algo],
                    )
                    plt.bar(
                        xpos_theo,
                        train_theo,
                        width=width,
                        color=theoretical_colors["FedAvg"],
                    )
                else:
                    xpos_actual = xpos_base + 0.5 * width
                    xpos_theo = xpos_base + 1.5 * width
                    plt.bar(
                        xpos_actual,
                        pretrain_actual,
                        width=width,
                        color=pretrain_colors_actual,
                    )
                    plt.bar(
                        xpos_actual,
                        train_actual,
                        width=width,
                        bottom=pretrain_actual,
                        color=actual_colors[algo],
                    )
                    plt.bar(
                        xpos_theo,
                        pretrain_theo,
                        width=width,
                        color=theoretical_colors["FedGCN_Pretrain"],
                    )
                    plt.bar(
                        xpos_theo,
                        train_theo,
                        width=width,
                        bottom=pretrain_theo,
                        color=theoretical_colors["FedGCN_Train"],
                    )

        # plt.title(f"Communication Cost (IID Beta={beta})", fontsize=22)
        # plt.xlabel("Dataset", fontsize=22)
        plt.ylabel("Communication Cost (MB)", fontsize=22)
        pretty_names = ["Cora", "Citeseer", "Pubmed"]
        plt.xticks(x_positions, pretty_names, rotation=0, fontsize=22)
        plt.yticks(fontsize=24)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        custom_lines = [
            plt.Line2D([0], [0], color="#1f77b4", lw=8),
            plt.Line2D([0], [0], color="#aec7e8", lw=8),
            plt.Line2D([0], [0], color="#2ca02c", lw=8),
            plt.Line2D([0], [0], color="#ff7f0e", lw=8),
            plt.Line2D([0], [0], color="#c5b0d5", lw=8),
            plt.Line2D([0], [0], color="#98df8a", lw=8),
        ]
        plt.legend(
            custom_lines,
            [
                "FedAvg Train Actual",
                "FedAvg Train Theoretical",
                "FedGCN Pretrain Actual",
                "FedGCN Train Actual",
                "FedGCN Pretrain Theoretical",
                "FedGCN Train Theoretical",
            ],
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(f"nc_comm_cost_comparison_beta{int(beta)}.pdf", dpi=300)
        plt.close()


def process_all_log_files(log_folder):
    log_files = glob.glob(os.path.join(log_folder, "*.log"))
    if not log_files:
        print(f"No log files found in {log_folder}")
        return pd.DataFrame()
    all_results = []
    for log_file in log_files:
        df = extract_nc_data(log_file)
        if not df.empty:
            all_results.append(df)
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        if os.path.isfile(log_path):
            df = extract_nc_data(log_path)
        elif os.path.isdir(log_path):
            df = process_all_log_files(log_path)
        else:
            sys.exit(1)
    else:
        default_log = "NC.log"
        if os.path.exists(default_log):
            df = extract_nc_data(default_log)
        else:
            df = process_all_log_files(os.getcwd())
    if not df.empty:
        # Only save ms to CSV
        df_csv = df.copy()
        if "Train_Time_s" in df_csv.columns:
            df_csv = df_csv.drop(columns=["Train_Time_s"])
        df_csv.to_csv("nc_data_raw.csv", index=False)
        plot_metric(df, "Accuracy", "Accuracy", "nc_accuracy_comparison")
        plot_metric(df, "Train_Time_s", "Training Time (s)", "nc_train_time_comparison")
        plot_comm_cost(df)
