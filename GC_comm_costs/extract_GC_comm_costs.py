import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_comm_costs(logfile):
    with open(logfile, "r") as f:
        log_content = f.read()

    experiments = re.split(r"-{80}\nRunning experiment \d+/\d+:", log_content)
    results = []

    for exp in experiments[1:]:
        algo_match = re.search(r"Algorithm: (\w+)", exp)
        dataset_match = re.search(r"Dataset: ([A-Z0-9-]+)", exp)
        trainers_match = re.search(r"Trainers: (\d+)", exp)
        accuracy_match = re.search(r"Average test accuracy: ([\d.]+)", exp)

        if not (algo_match and dataset_match and trainers_match):
            continue

        algo = algo_match.group(1)
        dataset = dataset_match.group(1)
        trainers = trainers_match.group(1)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None

        theoretical_pretrain = re.findall(
            r"//Log Theoretical Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        theoretical_train = re.findall(
            r"//Log Theoretical Train Comm Cost: ([\d.]+) MB //end", exp
        )

        actual_pretrain = re.search(
            r"//Log Total Actual Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        actual_train = re.search(
            r"//Log Total Actual Train Comm Cost: ([\d.]+) MB //end", exp
        )

        if not (theoretical_pretrain and theoretical_train):
            continue

        result = {
            "Algorithm": algo,
            "Dataset": dataset,
            "Trainers": int(trainers),
            "Theoretical_Pretrain_MB": float(theoretical_pretrain[-1])
            if theoretical_pretrain
            else 0,
            "Theoretical_Train_MB": float(theoretical_train[-1])
            if theoretical_train
            else 0,
            "Actual_Pretrain_MB": float(actual_pretrain.group(1))
            if actual_pretrain
            else None,
            "Actual_Train_MB": float(actual_train.group(1)) if actual_train else None,
            "Accuracy": accuracy,
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

            if (
                result["Theoretical_Pretrain_MB"] > 0
                and result["Actual_Pretrain_MB"] > 0
            ):
                result["Pretrain_Ratio"] = (
                    result["Actual_Pretrain_MB"] / result["Theoretical_Pretrain_MB"]
                )
            else:
                result["Pretrain_Ratio"] = (
                    float("inf")
                    if result["Actual_Pretrain_MB"] and result["Actual_Pretrain_MB"] > 0
                    else None
                )

            if result["Theoretical_Train_MB"] > 0:
                result["Train_Ratio"] = (
                    result["Actual_Train_MB"] / result["Theoretical_Train_MB"]
                )
            else:
                result["Train_Ratio"] = (
                    float("inf")
                    if result["Actual_Train_MB"] and result["Actual_Train_MB"] > 0
                    else None
                )

            if result["Theoretical_Total_MB"] > 0:
                result["Total_Ratio"] = (
                    result["Actual_Total_MB"] / result["Theoretical_Total_MB"]
                )
            else:
                result["Total_Ratio"] = (
                    float("inf")
                    if result["Actual_Total_MB"] and result["Actual_Total_MB"] > 0
                    else None
                )

        results.append(result)

    return pd.DataFrame(results)


def generate_dataset_comparisons(df, output_prefix="comm_cost"):
    comparison_data = (
        df.groupby(["Dataset", "Algorithm"])
        .agg(
            {
                "Theoretical_Pretrain_MB": "mean",
                "Theoretical_Train_MB": "mean",
                "Theoretical_Total_MB": "mean",
                "Actual_Pretrain_MB": "mean",
                "Actual_Train_MB": "mean",
                "Actual_Total_MB": "mean",
                "Train_Ratio": "mean",
                "Accuracy": "mean",
            }
        )
        .reset_index()
    )

    comparison_data.to_csv(
        f"{output_prefix}_dataset_algorithm_comparison.csv", index=False
    )

    datasets = df["Dataset"].unique()
    report_tables = []

    for dataset in datasets:
        dataset_data = comparison_data[comparison_data["Dataset"] == dataset]

        table_rows = []
        for _, row in dataset_data.iterrows():
            table_row = {
                "Algorithm": row["Algorithm"],
                "Theoretical Train (MB)": f"{row['Theoretical_Train_MB']:.2f}",
                "Actual Train (MB)": f"{row['Actual_Train_MB']:.2f}"
                if pd.notna(row["Actual_Train_MB"])
                else "N/A",
                "Train Overhead (MB)": f"{row['Actual_Train_MB'] - row['Theoretical_Train_MB']:.2f}"
                if pd.notna(row["Actual_Train_MB"])
                else "N/A",
                "Accuracy": f"{row['Accuracy']:.4f}"
                if pd.notna(row["Accuracy"])
                else "N/A",
            }
            table_rows.append(table_row)

        dataset_table = pd.DataFrame(table_rows)
        dataset_table.to_csv(f"{output_prefix}_{dataset}_comparison.csv", index=False)
        report_tables.append((dataset, dataset_table))

        # Create visualization for theoretical vs actual training communication costs
        plt.figure(figsize=(12, 8))
        plot_data = pd.melt(
            dataset_data,
            id_vars=["Algorithm"],
            value_vars=["Theoretical_Train_MB", "Actual_Train_MB"],
            var_name="Type",
            value_name="Communication Cost (MB)",
        )
        ax = sns.barplot(
            x="Algorithm", y="Communication Cost (MB)", hue="Type", data=plot_data
        )
        plt.title(f"{dataset} - Theoretical vs Actual Training Communication Costs")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{dataset}_train_comparison.png", dpi=300)
        plt.close()

    return report_tables


def generate_report(logfile, output_prefix="comm_cost"):
    df = extract_comm_costs(logfile)
    if df.empty:
        print("No communication cost data found in log file.")
        return None

    df.to_csv(f"{output_prefix}_raw.csv", index=False)

    report_tables = generate_dataset_comparisons(df, output_prefix)

    consolidated_report = pd.DataFrame()

    for dataset, dataset_table in report_tables:
        dataset_table["Dataset"] = dataset
        consolidated_report = pd.concat([consolidated_report, dataset_table])

    consolidated_report.to_csv(f"{output_prefix}_consolidated_report.csv", index=False)

    algorithm_summary = (
        df.groupby("Algorithm")
        .agg(
            {
                "Theoretical_Train_MB": "mean",
                "Actual_Train_MB": "mean",
                "Accuracy": "mean",
            }
        )
        .reset_index()
    )

    algorithm_summary["Average Overhead (MB)"] = (
        algorithm_summary["Actual_Train_MB"] - algorithm_summary["Theoretical_Train_MB"]
    )

    algorithm_summary.to_csv(f"{output_prefix}_algorithm_summary.csv", index=False)

    return consolidated_report


if __name__ == "__main__":
    import sys

    logfile = "GC.log"
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    output_prefix = "comm_cost"
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]

    consolidated_report = generate_report(logfile, output_prefix)

    if consolidated_report is not None:
        print("\nComparison by Dataset and Algorithm:")
        for dataset in consolidated_report["Dataset"].unique():
            print(f"\n=== Dataset: {dataset} ===")
            dataset_data = consolidated_report[
                consolidated_report["Dataset"] == dataset
            ]
            print(
                dataset_data[
                    [
                        "Algorithm",
                        "Theoretical Train (MB)",
                        "Actual Train (MB)",
                        "Accuracy",
                    ]
                ]
            )
