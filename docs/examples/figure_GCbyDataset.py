import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV file
file_path = '9.csv'  # Adjust this to the actual file path
df = pd.read_csv(file_path)

# Filter for GCFL+dWs algorithm
gcfl_dws_data = df[df['Algorithm'] == 'GCFL+dWs']

# Group by dataset and take the mean of communication cost and memory
grouped_by_dataset = gcfl_dws_data.groupby('Dataset').agg({
    'Pretrain Network Large1': 'mean',
    'Pretrain Network Large2': 'mean',
    'Pretrain Network Large3': 'mean',
    'Pretrain Network Large4': 'mean',
    'Pretrain Network Large5': 'mean',
    'Pretrain Network Large6': 'mean',
    'Pretrain Network Large7': 'mean',
    'Pretrain Network Large8': 'mean',
    'Pretrain Network Large9': 'mean',
    'Pretrain Network Large10': 'mean',
    'Train Network Large1': 'mean',
    'Train Network Large2': 'mean',
    'Train Network Large3': 'mean',
    'Train Network Large4': 'mean',
    'Train Network Large5': 'mean',
    'Train Network Large6': 'mean',
    'Train Network Large7': 'mean',
    'Train Network Large8': 'mean',
    'Train Network Large9': 'mean',
    'Train Network Large10': 'mean',
    'Pretrain Max Trainer Memory1': 'mean',
    'Pretrain Max Trainer Memory2': 'mean',
    'Pretrain Max Trainer Memory3': 'mean',
    'Pretrain Max Trainer Memory4': 'mean',
    'Pretrain Max Trainer Memory5': 'mean',
    'Pretrain Max Trainer Memory6': 'mean',
    'Pretrain Max Trainer Memory7': 'mean',
    'Pretrain Max Trainer Memory8': 'mean',
    'Pretrain Max Trainer Memory9': 'mean',
    'Pretrain Max Trainer Memory10': 'mean',
    'Train Max Trainer Memory1': 'mean',
    'Train Max Trainer Memory2': 'mean',
    'Train Max Trainer Memory3': 'mean',
    'Train Max Trainer Memory4': 'mean',
    'Train Max Trainer Memory5': 'mean',
    'Train Max Trainer Memory6': 'mean',
    'Train Max Trainer Memory7': 'mean',
    'Train Max Trainer Memory8': 'mean',
    'Train Max Trainer Memory9': 'mean',
    'Train Max Trainer Memory10': 'mean',
}).reset_index()

# Plot Pretrain Network for each large across datasets
plt.figure()
for i in range(1, 11):
    plt.plot(grouped_by_dataset['Dataset'], grouped_by_dataset[f'Pretrain Network Large{i}'],
             label=f'Pretrain Network Large{i}', marker='o')

plt.xlabel('Dataset')
plt.ylabel('Communication Cost (Pretrain Network)')
plt.title(
    'Pretrain Network Communication Cost for GCFL+dWs Across Datasets (Large Network)')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot Train Network for each large across datasets
plt.figure()
for i in range(1, 11):
    plt.plot(grouped_by_dataset['Dataset'], grouped_by_dataset[f'Train Network Large{i}'],
             label=f'Train Network Large{i}', marker='x')

plt.xlabel('Dataset')
plt.ylabel('Communication Cost (Train Network)')
plt.title(
    'Train Network Communication Cost for GCFL+dWs Across Datasets (Large Network)')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot Pretrain Max Trainer Memory for each trainer across datasets
plt.figure()
for i in range(1, 11):
    plt.plot(grouped_by_dataset['Dataset'], grouped_by_dataset[f'Pretrain Max Trainer Memory{i}'],
             label=f'Pretrain Max Trainer Memory {i}', marker='o')

plt.xlabel('Dataset')
plt.ylabel('Memory (Pretrain Max Trainer Memory)')
plt.title('Pretrain Max Trainer Memory Usage for GCFL+dWs Across Datasets')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot Train Max Trainer Memory for each trainer across datasets
plt.figure()
for i in range(1, 11):
    plt.plot(grouped_by_dataset['Dataset'], grouped_by_dataset[f'Train Max Trainer Memory{i}'],
             label=f'Train Max Trainer Memory {i}', marker='x')

plt.xlabel('Dataset')
plt.ylabel('Memory (Train Max Trainer Memory)')
plt.title('Train Max Trainer Memory Usage for GCFL+dWs Across Datasets')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
