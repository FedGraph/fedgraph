import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_nc_comm_costs(logfile):
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    experiments = re.split(r"-{80}\nRunning experiment \d+/\d+:", log_content)
    results = []

    for i, exp in enumerate(experiments[1:], 1):
        dataset_match = re.search(r"Dataset: ([^,]+)", exp)
        trainers_match = re.search(r"Trainers: (\d+)", exp)
        distribution_match = re.search(r"Distribution: ([^,]+)", exp)
        iid_beta_match = re.search(r"IID Beta: ([\d.]+)", exp)
        hops_match = re.search(r"Hops: (\d+)", exp)
        method_match = re.search(r"'method': '([^']+)'", exp)

        dataset = dataset_match.group(1) if dataset_match else f"Dataset_{i}"
        trainers = int(trainers_match.group(1)) if trainers_match else 0
        distribution = distribution_match.group(1) if distribution_match else "Unknown"
        iid_beta = float(iid_beta_match.group(1)) if iid_beta_match else 0
        hops = int(hops_match.group(1)) if hops_match else 0
        method = method_match.group(1) if method_match else "FedAvg"

        theoretical_pretrain = re.findall(
            r"//Log Theoretical Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        theoretical_train = re.findall(
            r"//Log Theoretical Train Comm Cost: ([\d.]+) MB //end", exp
        )
        actual_pretrain = re.findall(
            r"//Log Total Actual Pretrain Comm Cost: ([\d.]+) MB //end", exp
        )
        actual_train = re.findall(
            r"//Log Total Actual Train Comm Cost: ([\d.]+) MB //end", exp
        )

        test_loss_match = re.search(r"average_final_test_loss, ([\d.]+)", exp)
        test_acc_match = re.search(r"Average test accuracy, ([\d.]+)", exp)

        test_loss = float(test_loss_match.group(1)) if test_loss_match else None
        test_acc = float(test_acc_match.group(1)) if test_acc_match else None

        if not any(
            [theoretical_pretrain, theoretical_train, actual_pretrain, actual_train]
        ):
            continue

        experiment_id = f"{method}_{dataset}_{distribution}_{iid_beta}_{hops}"

        result = {
            "Experiment": experiment_id,
            "Method": method,
            "Dataset": dataset,
            "Trainers": trainers,
            "Distribution": distribution,
            "IID_Beta": iid_beta,
            "Hops": hops,
            "Theoretical_Pretrain_MB": float(theoretical_pretrain[-1])
            if theoretical_pretrain
            else 0,
            "Theoretical_Train_MB": float(theoretical_train[-1])
            if theoretical_train
            else 0,
            "Actual_Pretrain_MB": float(actual_pretrain[-1])
            if actual_pretrain
            else None,
            "Actual_Train_MB": float(actual_train[-1]) if actual_train else None,
            "Test_Loss": test_loss,
            "Test_Accuracy": test_acc,
            "Actual_Total_MB": None,  # Initialize with None to avoid KeyError
            "Pretrain_Ratio": None,
            "Train_Ratio": None,
            "Total_Ratio": None,
            "Pretrain_Overhead_MB": None,
            "Train_Overhead_MB": None,
            "Total_Overhead_MB": None,
            "Pretrain_Percentage": None,
            "Accuracy_per_MB": None,
        }

        result["Theoretical_Total_MB"] = (
            result["Theoretical_Pretrain_MB"] + result["Theoretical_Train_MB"]
        )

        if result["Actual_Pretrain_MB"] is not None:
            if result["Theoretical_Pretrain_MB"] > 0:
                result["Pretrain_Ratio"] = (
                    result["Actual_Pretrain_MB"] / result["Theoretical_Pretrain_MB"]
                )
                result["Pretrain_Overhead_MB"] = (
                    result["Actual_Pretrain_MB"] - result["Theoretical_Pretrain_MB"]
                )
            else:
                result["Pretrain_Ratio"] = (
                    float("inf") if result["Actual_Pretrain_MB"] > 0 else None
                )
                result["Pretrain_Overhead_MB"] = result["Actual_Pretrain_MB"]

        if result["Actual_Train_MB"] is not None:
            if result["Theoretical_Train_MB"] > 0:
                result["Train_Ratio"] = (
                    result["Actual_Train_MB"] / result["Theoretical_Train_MB"]
                )
                result["Train_Overhead_MB"] = (
                    result["Actual_Train_MB"] - result["Theoretical_Train_MB"]
                )
            else:
                result["Train_Ratio"] = (
                    float("inf") if result["Actual_Train_MB"] > 0 else None
                )
                result["Train_Overhead_MB"] = result["Actual_Train_MB"]

        if (
            result["Actual_Pretrain_MB"] is not None
            and result["Actual_Train_MB"] is not None
        ):
            result["Actual_Total_MB"] = (
                result["Actual_Pretrain_MB"] + result["Actual_Train_MB"]
            )

            if result["Theoretical_Total_MB"] > 0:
                result["Total_Ratio"] = (
                    result["Actual_Total_MB"] / result["Theoretical_Total_MB"]
                )
                result["Total_Overhead_MB"] = (
                    result["Actual_Total_MB"] - result["Theoretical_Total_MB"]
                )
            else:
                result["Total_Ratio"] = (
                    float("inf") if result["Actual_Total_MB"] > 0 else None
                )
                result["Total_Overhead_MB"] = result["Actual_Total_MB"]

        if result["Actual_Total_MB"] is not None and result["Actual_Total_MB"] > 0:
            result["Pretrain_Percentage"] = (
                result["Actual_Pretrain_MB"] / result["Actual_Total_MB"]
            ) * 100

        if (
            result["Test_Accuracy"] is not None
            and result["Actual_Total_MB"] is not None
        ):
            result["Accuracy_per_MB"] = (
                result["Test_Accuracy"] / result["Actual_Total_MB"]
            )

        results.append(result)

    return pd.DataFrame(results)


def generate_comparison_charts(df, output_prefix="nc_comm_cost"):
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    colors = {"Theoretical": "#1f77b4", "Actual": "#ff7f0e"}  # Blue  # Orange

    grouped_data = (
        df.groupby(["Method", "Dataset", "Hops", "IID_Beta"])
        .agg(
            {
                "Theoretical_Pretrain_MB": "mean",
                "Actual_Pretrain_MB": "mean",
                "Theoretical_Train_MB": "mean",
                "Actual_Train_MB": "mean",
            }
        )
        .reset_index()
    )

    grouped_data["Config"] = grouped_data.apply(
        lambda row: f"{row['Method']}\n{row['Dataset']}\nHops={row['Hops']}, Beta={row['IID_Beta']}",
        axis=1,
    )

    # 1. Pretrain Phase - Theoretical vs Actual
    plt.figure(figsize=(14, 8))

    num_configs = len(grouped_data)
    x = np.arange(num_configs)
    width = 0.35

    sorted_data = grouped_data.sort_values(["Method", "Dataset", "Hops", "IID_Beta"])

    theo_bars = plt.bar(
        x - width / 2,
        sorted_data["Theoretical_Pretrain_MB"],
        width,
        label="Theoretical",
        color=colors["Theoretical"],
    )

    actual_bars = plt.bar(
        x + width / 2,
        sorted_data["Actual_Pretrain_MB"],
        width,
        label="Actual",
        color=colors["Actual"],
    )

    for i, (theo, actual) in enumerate(
        zip(sorted_data["Theoretical_Pretrain_MB"], sorted_data["Actual_Pretrain_MB"])
    ):
        if theo > 0:
            ratio = actual / theo
            plt.text(
                i,
                max(theo, actual) + 50,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.title("Pretrain Phase - Theoretical vs Actual Communication Cost", fontsize=16)
    plt.xlabel("Method / Dataset / Configuration", fontsize=14)
    plt.ylabel("Communication Cost (MB)", fontsize=14)
    plt.xticks(x, sorted_data["Config"], rotation=45, ha="right")
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{output_prefix}_pretrain_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Train Phase - Theoretical vs Actual
    plt.figure(figsize=(14, 8))

    theo_bars = plt.bar(
        x - width / 2,
        sorted_data["Theoretical_Train_MB"],
        width,
        label="Theoretical",
        color=colors["Theoretical"],
    )

    actual_bars = plt.bar(
        x + width / 2,
        sorted_data["Actual_Train_MB"],
        width,
        label="Actual",
        color=colors["Actual"],
    )

    for i, (theo, actual) in enumerate(
        zip(sorted_data["Theoretical_Train_MB"], sorted_data["Actual_Train_MB"])
    ):
        if theo > 0:
            ratio = actual / theo
            plt.text(
                i,
                max(theo, actual) + 50,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.title("Train Phase - Theoretical vs Actual Communication Cost", fontsize=16)
    plt.xlabel("Method / Dataset / Configuration", fontsize=14)
    plt.ylabel("Communication Cost (MB)", fontsize=14)
    plt.xticks(x, sorted_data["Config"], rotation=45, ha="right")
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{output_prefix}_train_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Comparison charts saved to {output_dir}/ directory")


def generate_nc_report(logfile, output_prefix="nc_comm_cost"):
    # Extract data
    df = extract_nc_comm_costs(logfile)

    if df.empty:
        print("No communication cost data found in log file.")
        return None

    # Save raw data
    df.to_csv(f"{output_prefix}_raw.csv", index=False)
    print(f"Raw data saved to {output_prefix}_raw.csv")

    # Generate only the comparison charts
    generate_comparison_charts(df, output_prefix)

    # Print summary
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nCommunication Cost Summary:\n")

    for method in df["Method"].unique():
        print(f"\n=== Method: {method} ===")
        for dataset in sorted(df[df["Method"] == method]["Dataset"].unique()):
            for hops in sorted(
                df[(df["Method"] == method) & (df["Dataset"] == dataset)][
                    "Hops"
                ].unique()
            ):
                for beta in sorted(
                    df[
                        (df["Method"] == method)
                        & (df["Dataset"] == dataset)
                        & (df["Hops"] == hops)
                    ]["IID_Beta"].unique()
                ):
                    data = df[
                        (df["Method"] == method)
                        & (df["Dataset"] == dataset)
                        & (df["Hops"] == hops)
                        & (df["IID_Beta"] == beta)
                    ]

                    if not data.empty:
                        row = data.iloc[0]
                        print(f"\nDataset: {dataset}, Hops: {hops}, IID Beta: {beta}")
                        print(
                            f"  Pretrain: Theoretical={row['Theoretical_Pretrain_MB']:.2f} MB, "
                            + f"Actual={row['Actual_Pretrain_MB']:.2f} MB, "
                            + f"Ratio={row['Pretrain_Ratio']:.2f}"
                            if row["Pretrain_Ratio"] is not None
                            else "Ratio=N/A"
                        )
                        print(
                            f"  Train: Theoretical={row['Theoretical_Train_MB']:.2f} MB, "
                            + f"Actual={row['Actual_Train_MB']:.2f} MB, "
                            + f"Ratio={row['Train_Ratio']:.2f}"
                            if row["Train_Ratio"] is not None
                            else "Ratio=N/A"
                        )

    return df


if __name__ == "__main__":
    import sys

    logfile = "NC.log"
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    output_prefix = "nc_comm_cost"
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]

    df = generate_nc_report(logfile, output_prefix)

    if df is not None:
        print("\nAnalysis completed successfully.")
