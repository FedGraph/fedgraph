import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def extract_accuracy_by_dataset_algo(logfile):
    """
    Extract round-wise Global Test Accuracy per dataset and algorithm from a log file.

    Returns:
        dict: {(dataset, algorithm): pd.DataFrame with columns ['Round', 'Accuracy']}
    """
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Split log into experiment blocks
    experiments = re.findall(
        r"Running experiment \d+/\d+:.*?(?=Running experiment|\Z)",
        log_content,
        re.DOTALL,
    )

    results = {}

    for exp in experiments:
        # Extract dataset
        dataset_match = re.search(r"Dataset: ([a-zA-Z0-9_-]+)", exp)
        if not dataset_match:
            continue
        dataset = dataset_match.group(1)

        # Extract algorithm
        algo_match = re.search(r"method': '([A-Za-z0-9+_]+)'", exp)
        if not algo_match:
            algo_match = re.search(r"Changing method to ([A-Za-z0-9+_]+)", exp)
        algorithm = algo_match.group(1).strip() if algo_match else "FedAvg"

        # Extract all round accuracies
        round_accs = re.findall(r"Round (\d+): Global Test Accuracy = ([\d.]+)", exp)
        if not round_accs:
            continue

        rounds = [int(r[0]) for r in round_accs]
        accs = [float(r[1]) for r in round_accs]
        df = pd.DataFrame({"Round": rounds, "Accuracy": accs})
        results[(dataset, algorithm)] = df

    return results


def plot_accuracy_curves_grouped(results):
    """
    Plot accuracy curves with both FedAvg and FedGCN in the same chart per dataset.

    Saves 4 figures, one per dataset.
    """
    datasets = {
        "cora": "Cora",
        "citeseer": "Citeseer",
        "pubmed": "Pubmed",
        "ogbn-arxiv": "Ogbn-Arxiv",
    }
    algos = ["FedAvg", "fedgcn"]
    display_names = {"FedAvg": "FedAvg", "fedgcn": "FedGCN"}
    colors = {"FedAvg": "#1f77b4", "fedgcn": "#ff7f0e"}

    for dataset_key, dataset_title in datasets.items():
        plt.figure(figsize=(10, 5))  # Shorter figure for compact display
        for algo in algos:
            df = results.get((dataset_key, algo))
            if df is not None and not df.empty:
                plt.plot(
                    df["Round"],
                    df["Accuracy"],
                    label=display_names[algo],
                    linewidth=4,
                    color=colors[algo],
                )
        plt.title(dataset_title, fontsize=38)
        plt.xlabel("Training Round", fontsize=34)
        plt.ylabel("Test Accuracy", fontsize=34)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=20, loc="lower right")
        plt.tight_layout()
        plt.savefig(f"nc_accuracy_curve_{dataset_key}.pdf", dpi=300)
        plt.close()


if __name__ == "__main__":
    log_path = "NC.log"
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        exit(1)

    results = extract_accuracy_by_dataset_algo(log_path)
    plot_accuracy_curves_grouped(results)
