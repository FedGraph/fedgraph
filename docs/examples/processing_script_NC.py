import re
import time

import pandas as pd


def process_log(log_content):
    experiments = []
    current_experiment = {}

    for line in log_content.splitlines():
        experiment_match = re.match(
            r"Running experiment with: Dataset=([^,]+),\s*Number of Trainers=(\d+),\s*Distribution Type=([^,]+),\s*IID Beta=([0-9.]+),\s*Number of Hops=(\d+),\s*Batch Size=([^,]+)",
            line,
        )

        if experiment_match:
            if current_experiment:
                experiments.append(current_experiment)
            current_experiment = {
                "Dataset": experiment_match.group(1),
                "Number of Trainers": int(experiment_match.group(2)),
                "Distribution Type": experiment_match.group(3),
                "IID Beta": float(experiment_match.group(4)),
                "Number of Hops": int(experiment_match.group(5)),
                "Batch Size": int(experiment_match.group(6)),
            }
            pretrain_mode = True
            train_mode = False

        pretrain_time_match = re.search(r"pretrain_time: (\d+\.\d+)", line)
        if pretrain_time_match:
            pretrain_mode = True
            train_mode = False
            current_experiment["Pretrain Time"] = float(pretrain_time_match.group(1))

        pretrain_max_trainer_memory_match = re.search(
            r"Log Max memory for Large(\d+): (\d+\.\d+)", line
        )
        if pretrain_max_trainer_memory_match and pretrain_mode:
            current_experiment[
                f"Pretrain Max Trainer Memory{pretrain_max_trainer_memory_match.group(1)}"
            ] = float(pretrain_max_trainer_memory_match.group(2))

        pretrain_max_server_memory_match = re.search(
            r"Log Max memory for Server: (\d+\.\d+)", line
        )
        if pretrain_max_server_memory_match and pretrain_mode:
            current_experiment["Pretrain Max Server Memory"] = float(
                pretrain_max_server_memory_match.group(1)
            )

        pretrain_network_match = re.search(r"Log ([^,]+) network: (\d+\.\d+)", line)
        if pretrain_network_match and pretrain_mode:
            current_experiment[
                f"Pretrain Network {pretrain_network_match.group(1)}"
            ] = float(pretrain_network_match.group(2))
        if re.search("Pretrain end time recorded and duration set to gauge.", line):
            pretrain_mode = False
            train_mode = True

        train_time_match = re.search(r"train_time: (\d+\.\d+)", line)
        if train_time_match:
            current_experiment["Train Time"] = float(train_time_match.group(1))

        train_max_trainer_memory_match = re.search(
            r"Log Max memory for Large(\d+): (\d+\.\d+)", line
        )
        if train_max_trainer_memory_match and train_mode:
            current_experiment[
                f"Train Max Trainer Memory{train_max_trainer_memory_match.group(1)}"
            ] = float(train_max_trainer_memory_match.group(2))

        train_max_server_memory_match = re.search(
            r"Log Max memory for Server: (\d+\.\d+)", line
        )
        if train_max_server_memory_match and train_mode:
            current_experiment["Train Max Server Memory"] = float(
                train_max_server_memory_match.group(1)
            )

        train_network_match = re.search(r"Log ([^,]+) network: (\d+\.\d+)", line)
        if train_network_match and train_mode:
            current_experiment[
                f"Train Network {(train_network_match.group(1))}"
            ] = float(train_network_match.group(2))
        average_accuracy_match = re.search(r"Average test accuracy: (\d+\.\d+)", line)
        if average_accuracy_match:
            current_experiment["Average Test Accuracy"] = float(
                average_accuracy_match.group(1)
            )

    if current_experiment:
        experiments.append(current_experiment)

    return pd.DataFrame(experiments)


def load_log_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        log_content = file.read()
    return log_content


file_path = "NC_arxiv_batch_gpu.log"
log_content = load_log_file(file_path)
df = process_log(log_content)


def reorder_dataframe_columns(df):
    desired_columns = [
        "Dataset",
        "Number of Trainers",
        "Distribution Type",
        "IID Beta",
        "Number of Hops",
        "Batch Size",
        "Average Test Accuracy",
    ]

    new_column_order = desired_columns + [
        col for col in df.columns if col not in desired_columns
    ]

    df = df[new_column_order]

    return df


df = reorder_dataframe_columns(df)
csv_file_path = "NC_arxiv_batch_gpu.csv"
df.to_csv(csv_file_path)
print(df.iloc[0, :])
