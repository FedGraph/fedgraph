import pandas as pd
import re



def process_log(log_content):
    experiments = []
    current_experiment = {}


    for line in log_content.splitlines():
        experiment_match = re.match(
            r"Running experiment with: Algorithm=(\w+), Dataset=(\w+),Number of Trainers=(\d+)", line)
        if experiment_match:
            if current_experiment:
                experiments.append(current_experiment)
            current_experiment = {
                "Algorithm": experiment_match.group(1),
                "Dataset": experiment_match.group(2),
                "Number of Trainers": int(experiment_match.group(3)),
                "Pretrain Time": None,
                "Pretrain Max Trainer Memory": None,
                "Pretrain Max Server Memory": None,
                "Pretrain Network": None,
                "Train Time": None,
                "Train Max Trainer Memory": None,
                "Train Max Server Memory": None,
                "Train Network": None,
            }
            pretrain_mode = True
            train_mode = False



        pretrain_time_match = re.search(r"pretrain_time: (\d+\.\d+)", line)
        if pretrain_time_match:
            pretrain_mode = True
            train_mode = False
            current_experiment["Pretrain Time"] = float(
                pretrain_time_match.group(1))

        pretrain_max_trainer_memory_match = re.search(
            r"Log Max trainer memory value: (\d+\.\d+)", line)
        if pretrain_max_trainer_memory_match and pretrain_mode:
            current_experiment["Pretrain Max Trainer Memory"] = float(
                pretrain_max_trainer_memory_match.group(1))

        pretrain_max_server_memory_match = re.search(
            r"Log Max server memory value: (\d+\.\d+)", line)
        if pretrain_max_server_memory_match and pretrain_mode:
            current_experiment["Pretrain Max Server Memory"] = float(
                pretrain_max_server_memory_match.group(1))

        pretrain_network_match = re.search(r"network: (\d+\.\d+)", line)
        if pretrain_network_match and pretrain_mode:
            current_experiment["Pretrain Network"] = float(
                pretrain_network_match.group(1))
            pretrain_mode = False
            train_mode = True


        train_time_match = re.search(r"train_time: (\d+\.\d+)", line)
        if train_time_match:
            
            current_experiment["Train Time"] = float(
                train_time_match.group(1))

        train_max_trainer_memory_match = re.search(
            r"Log Max trainer memory value: (\d+\.\d+)", line)
        if train_max_trainer_memory_match and train_mode:
            current_experiment["Train Max Trainer Memory"] = float(
                train_max_trainer_memory_match.group(1))

        train_max_server_memory_match = re.search(
            r"Log Max server memory value: (\d+\.\d+)", line)
        if train_max_server_memory_match and train_mode:
            current_experiment["Train Max Server Memory"] = float(
                train_max_server_memory_match.group(1))

        train_network_match = re.search(r"network: (\d+\.\d+)", line)
        if train_network_match and train_mode:
            current_experiment["Train Network"] = float(
                train_network_match.group(1))
        average_accuracy_match = re.search(r"Average test accuracy: (\d+\.\d+)", line)
        if average_accuracy_match:
            current_experiment["Average Test Accuracy"] = float(average_accuracy_match.group(1))

    if current_experiment:
        experiments.append(current_experiment)

    return pd.DataFrame(experiments)



def load_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()
    return log_content


file_path = "testGC.log"  
log_content = load_log_file(file_path)
df = process_log(log_content)

print(df.iloc[1, :])
