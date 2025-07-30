#!/usr/bin/env python3

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# Set matplotlib style
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def extract_batch_size_data(log_file):
    """
    Extract training time, test accuracy, and memory usage data for different batch sizes.

    Args:
        log_file (str): Path to the log file

    Returns:
        tuple: (batch_size_results, memory_data)
    """
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Split log into sections by batch size experiments
    batch_sections = re.split(r"Running experiment \d+/\d+:", log_content)

    batch_size_results = []
    memory_data = []

    for section in batch_sections[1:]:  # Skip first empty section
        # Extract batch size
        batch_size_match = re.search(r"Batch Size: (-?\d+)", section)
        if not batch_size_match:
            continue
        batch_size = int(batch_size_match.group(1))

        # Extract final test accuracy (last round)
        accuracy_matches = re.findall(
            r"Round \d+: Global Test Accuracy = ([\d.]+)", section
        )
        if accuracy_matches:
            final_accuracy = float(accuracy_matches[-1])
        else:
            final_accuracy = None

        # Extract training time
        train_time_match = re.search(r"Training Time = ([\d.]+) seconds", section)
        if train_time_match:
            train_time = float(train_time_match.group(1))
        else:
            train_time = None

        if final_accuracy is not None and train_time is not None:
            batch_size_results.append(
                {
                    "batch_size": batch_size,
                    "final_accuracy": final_accuracy,
                    "train_time": train_time,
                }
            )

        # Extract memory usage data
        memory_section = re.search(
            r"TRAINER MEMORY vs LOCAL GRAPH SIZE.*?Total Memory Usage: ([\d.]+) MB",
            section,
            re.DOTALL,
        )

        if memory_section:
            # Extract individual trainer memory data
            trainer_lines = re.findall(
                r"(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
                memory_section.group(0),
            )

            batch_memory_data = []
            for trainer_data in trainer_lines:
                (
                    trainer_id,
                    memory_mb,
                    nodes,
                    edges,
                    memory_per_node,
                    memory_per_edge,
                ) = trainer_data
                batch_memory_data.append(
                    {
                        "batch_size": batch_size,
                        "trainer_id": int(trainer_id),
                        "memory_mb": float(memory_mb),
                        "nodes": int(nodes),
                        "edges": int(edges),
                        "memory_per_node": float(memory_per_node),
                        "memory_per_edge": float(memory_per_edge)
                        if int(edges) > 0
                        else 0,
                    }
                )

            if batch_memory_data:
                memory_data.extend(batch_memory_data)

    return batch_size_results, memory_data


def remove_outliers_by_residuals(x, y, threshold_std=2.0):
    """
    Remove outliers based on residuals from initial fit.

    Args:
        x: input data
        y: target data
        threshold_std: number of standard deviations for outlier threshold

    Returns:
        tuple: (clean_x, clean_y, outlier_mask)
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 5:
        return x, y, np.ones(len(x), dtype=bool)

    # Initial fit with polynomial (degree 2)
    try:
        coeffs = np.polyfit(x, y, 2)
        y_pred = np.polyval(coeffs, x)
    except:
        # Fallback to linear fit
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)

    # Calculate residuals
    residuals = np.abs(y - y_pred)

    # Define outliers as points with residuals > threshold_std * std
    residual_threshold = threshold_std * np.std(residuals)
    mask = residuals <= residual_threshold

    return x[mask], y[mask], mask


def fit_clean_data(x, y, method="polynomial"):
    """
    Fit cleaned data with specified method.

    Args:
        x: cleaned input data
        y: cleaned target data
        method: fitting method

    Returns:
        tuple: (x_trend, y_trend, r2_score, equation_str)
    """
    if len(x) < 3:
        return None, None, 0, "Insufficient data"

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    if method == "linear":
        # Simple linear regression
        coeffs = np.polyfit(x.flatten(), y, 1)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = np.polyval(coeffs, x_trend)
        r2 = r2_score(y, np.polyval(coeffs, x.flatten()))
        equation = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.1f}"

    elif method == "polynomial":
        # Polynomial regression (degree 2)
        coeffs = np.polyfit(x.flatten(), y, 2)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = np.polyval(coeffs, x_trend)
        r2 = r2_score(y, np.polyval(coeffs, x.flatten()))
        equation = f"y = {coeffs[0]:.2e}x² + {coeffs[1]:.3f}x + {coeffs[2]:.1f}"

    elif method == "log":
        # Logarithmic fitting
        x_log = np.log(x.flatten() + 1)
        coeffs = np.polyfit(x_log, y, 1)
        x_trend = np.linspace(x.min(), x.max(), 100)
        x_trend_log = np.log(x_trend + 1)
        y_trend = np.polyval(coeffs, x_trend_log)
        r2 = r2_score(y, np.polyval(coeffs, x_log))
        equation = f"y = {coeffs[0]:.1f}log(x) + {coeffs[1]:.1f}"

    return x_trend, y_trend, r2, equation


def plot_batch_size_performance(batch_results):
    """
    Plot training time and test accuracy vs batch size.

    Args:
        batch_results (list): List of dictionaries with batch size results
    """
    if not batch_results:
        print("No batch size results found")
        return

    df = pd.DataFrame(batch_results)
    df = df.sort_values("batch_size")

    # Replace -1 with "Full" for better visualization
    df["batch_size_label"] = df["batch_size"].apply(
        lambda x: "Full" if x == -1 else str(x)
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training time
    color_time = "#5B9BD5"
    ax1.set_xlabel("Batch Size", fontsize=20)
    ax1.set_ylabel("Train Time (s)", color=color_time, fontsize=20)
    bars = ax1.bar(
        df["batch_size_label"], df["train_time"], color=color_time, alpha=0.7, width=0.6
    )
    ax1.tick_params(axis="y", labelcolor=color_time, labelsize=18)
    ax1.tick_params(axis="x", labelsize=18)

    # Set y-axis limits to make the chart more compact
    min_time = df["train_time"].min()
    max_time = df["train_time"].max()

    y_min = min_time * 0.9
    y_max = max_time * 1.05
    ax1.set_ylim(y_min, y_max)

    # Plot test accuracy on secondary y-axis
    ax2 = ax1.twinx()
    color_acc = "#FF7F0E"
    ax2.set_ylabel("Test Accuracy", color=color_acc, fontsize=20)
    line = ax2.plot(
        df["batch_size_label"],
        df["final_accuracy"],
        color=color_acc,
        marker="o",
        linewidth=3,
        markersize=8,
    )
    ax2.tick_params(axis="y", labelcolor=color_acc, labelsize=18)

    # Set accuracy y-axis limits
    acc_y_min = 0.4000
    acc_y_max = 0.4200
    ax2.set_ylim(acc_y_min, acc_y_max)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        [bars[0]] + line, ["Train Time", "Test Accuracy"], loc="upper left", fontsize=16
    )

    plt.tight_layout()
    plt.savefig("batch_size_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_memory_analysis(memory_data):
    """
    Plot memory usage analysis with outlier removal and clean fitting.

    Args:
        memory_data (list): List of dictionaries with memory usage data
    """
    if not memory_data:
        print("No memory data found")
        return

    df_memory = pd.DataFrame(memory_data)

    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Get unique batch sizes and colors
    batch_sizes = sorted(df_memory["batch_size"].unique())
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]  # Red, Orange, Green

    # Plot 1: Memory vs Nodes with outlier removal
    print("Memory vs Nodes Analysis (After Outlier Removal):")
    print("-" * 50)

    for i, (batch_size, color) in enumerate(zip(batch_sizes, colors)):
        batch_data = df_memory[df_memory["batch_size"] == batch_size]
        batch_data = batch_data[
            batch_data["nodes"] > 0
        ]  # Filter out trainers with 0 nodes

        label = "Batch Full" if batch_size == -1 else f"Batch {batch_size}"

        # Remove outliers based on residuals from fit
        clean_x, clean_y, mask = remove_outliers_by_residuals(
            batch_data["nodes"], batch_data["memory_mb"], threshold_std=1.5
        )

        # Plot outliers (removed points) in light gray
        outlier_x = batch_data["nodes"][~mask]
        outlier_y = batch_data["memory_mb"][~mask]
        if len(outlier_x) > 0:
            ax1.scatter(
                outlier_x,
                outlier_y,
                alpha=0.3,
                color="lightgray",
                s=30,
                marker="x",
                label="Outliers" if i == 0 else "",
            )

        # Plot cleaned data (bright)
        ax1.scatter(
            clean_x,
            clean_y,
            alpha=0.8,
            color=color,
            label=label,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        # Fit cleaned data - for nodes vs memory, linear/polynomial is usually good
        if len(clean_x) > 5:
            # For nodes, try linear and polynomial
            methods = ["linear", "polynomial"]
            best_r2 = -np.inf
            best_fit = None
            best_method = None

            for method in methods:
                try:
                    x_trend, y_trend, r2, equation = fit_clean_data(
                        clean_x, clean_y, method
                    )
                    if x_trend is not None and r2 > best_r2:
                        # For polynomial fits, avoid if the curve is too extreme
                        if method == "polynomial":
                            # Check if the curve has reasonable shape (not too curved)
                            if max(y_trend) - min(y_trend) > 2 * (
                                max(clean_y) - min(clean_y)
                            ):
                                continue  # Skip overly curved fits
                        best_r2 = r2
                        best_fit = (x_trend, y_trend, equation)
                        best_method = method
                except:
                    continue

            if best_fit is not None:
                x_trend, y_trend, equation = best_fit
                ax1.plot(
                    x_trend,
                    y_trend,
                    "--",
                    color=color,
                    alpha=0.9,
                    linewidth=3,
                    label=f"{label} Fit (R²={best_r2:.3f})",
                )

                outliers_removed = len(batch_data) - len(clean_x)
                print(f"{label}:")
                print(f"  Equation: {equation}")
                print(f"  R²: {best_r2:.3f}")
                print(f"  Method: {best_method.upper()}")
                print(
                    f"  Outliers removed: {outliers_removed}/{len(batch_data)} ({outliers_removed/len(batch_data)*100:.1f}%)"
                )
                print()

    ax1.set_xlabel("Number of Local Nodes", fontsize=16)
    ax1.set_ylabel("Memory Usage (MB)", fontsize=16)
    ax1.set_title(
        "Memory Usage vs Number of Local Nodes\n(195 Trainers - Outliers Removed)",
        fontsize=18,
        pad=15,
    )
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    # Plot 2: Memory vs Edges with outlier removal
    print("Memory vs Edges Analysis (After Outlier Removal):")
    print("-" * 50)

    for i, (batch_size, color) in enumerate(zip(batch_sizes, colors)):
        batch_data = df_memory[df_memory["batch_size"] == batch_size]
        batch_data = batch_data[
            batch_data["edges"] > 0
        ]  # Filter out trainers with 0 edges

        label = "Batch Full" if batch_size == -1 else f"Batch {batch_size}"

        # Remove outliers based on residuals from fit
        clean_x, clean_y, mask = remove_outliers_by_residuals(
            batch_data["edges"], batch_data["memory_mb"], threshold_std=1.5
        )

        # Plot outliers (removed points) in light gray
        outlier_x = batch_data["edges"][~mask]
        outlier_y = batch_data["memory_mb"][~mask]
        if len(outlier_x) > 0:
            ax2.scatter(
                outlier_x, outlier_y, alpha=0.3, color="lightgray", s=30, marker="x"
            )

        # Plot cleaned data (bright)
        ax2.scatter(
            clean_x,
            clean_y,
            alpha=0.8,
            color=color,
            label=label,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        # Fit cleaned data - for edges vs memory, prefer linear or log fits
        if len(clean_x) > 5:
            # For edges, try linear and log (avoid polynomial which can be too curved)
            methods = ["linear", "log"]
            best_r2 = -np.inf
            best_fit = None
            best_method = None

            for method in methods:
                try:
                    x_trend, y_trend, r2, equation = fit_clean_data(
                        clean_x, clean_y, method
                    )
                    if x_trend is not None and r2 > best_r2:
                        best_r2 = r2
                        best_fit = (x_trend, y_trend, equation)
                        best_method = method
                except:
                    continue

            # If both linear and log have poor fits, try polynomial but be cautious
            if best_fit is None or best_r2 < 0.7:
                try:
                    x_trend, y_trend, r2, equation = fit_clean_data(
                        clean_x, clean_y, "polynomial"
                    )
                    if x_trend is not None and r2 > best_r2:
                        # Check if polynomial curve is reasonable
                        if max(y_trend) - min(y_trend) <= 1.5 * (
                            max(clean_y) - min(clean_y)
                        ):
                            best_r2 = r2
                            best_fit = (x_trend, y_trend, equation)
                            best_method = "polynomial"
                except:
                    pass

            if best_fit is not None:
                x_trend, y_trend, equation = best_fit
                ax2.plot(
                    x_trend,
                    y_trend,
                    "--",
                    color=color,
                    alpha=0.9,
                    linewidth=3,
                    label=f"{label} Fit (R²={best_r2:.3f})",
                )

                outliers_removed = len(batch_data) - len(clean_x)
                print(f"{label}:")
                print(f"  Equation: {equation}")
                print(f"  R²: {best_r2:.3f}")
                print(f"  Method: {best_method.upper()}")
                print(
                    f"  Outliers removed: {outliers_removed}/{len(batch_data)} ({outliers_removed/len(batch_data)*100:.1f}%)"
                )
                print()

    ax2.set_xlabel("Number of Local Edges", fontsize=16)
    ax2.set_ylabel("Memory Usage (MB)", fontsize=16)
    ax2.set_title(
        "Memory Usage vs Number of Local Edges\n(195 Trainers - Outliers Removed)",
        fontsize=18,
        pad=15,
    )
    ax2.legend(fontsize=11, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)

    # Print overall statistics
    print("\n" + "=" * 50)
    print("MEMORY USAGE STATISTICS")
    print("=" * 50)
    for batch_size in batch_sizes:
        batch_data = df_memory[df_memory["batch_size"] == batch_size]
        batch_label = "Full" if batch_size == -1 else f"{batch_size}"
        print(f"Batch {batch_label}:")
        print(f"  Total samples: {len(batch_data)}")
        print(f"  Avg Memory: {batch_data['memory_mb'].mean():.1f} MB")
        print(
            f"  Memory Range: {batch_data['memory_mb'].min():.1f} - {batch_data['memory_mb'].max():.1f} MB"
        )
        print(f"  Avg Nodes: {batch_data['nodes'].mean():.0f}")
        print(f"  Avg Edges: {batch_data['edges'].mean():.0f}")
        print()

    plt.tight_layout()
    plt.savefig("memory_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def main():
    """
    Main function to extract data and generate plots.
    """
    log_file = "NC_100M.log"  # Change this to your log file path

    print("Extracting data from log file...")
    batch_results, memory_data = extract_batch_size_data(log_file)

    if batch_results:
        print(f"Found {len(batch_results)} batch size experiments")
        print("Batch size results:")
        for result in batch_results:
            print(
                f"  Batch {result['batch_size']}: Accuracy={result['final_accuracy']:.4f}, Time={result['train_time']:.1f}s"
            )

        print("\nGenerating batch size performance plot...")
        plot_batch_size_performance(batch_results)

    if memory_data:
        print(f"Found memory data for {len(memory_data)} trainer instances")
        print("\nGenerating clean memory analysis plots...")
        plot_memory_analysis(memory_data)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
