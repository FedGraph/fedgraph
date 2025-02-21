import matplotlib.pyplot as plt
import pandas as pd

# Let's assume the CSV content is stored in a file called 'data.csv'
# Now I will read the data from the CSV and process it

# Load the data from CSV file
file_path = "4.csv"  # Adjust this to the actual file path
df = pd.read_csv(file_path)

# Filter for GCFL algorithm
gcfl_data = df[df["Algorithm"].str.contains("GCFL")]

# Group by number of trainers and take the mean of communication cost and memory
gcfl_grouped = (
    gcfl_data.groupby("Number of Trainers")
    .agg(
        {
            "Pretrain Network": "mean",
            "Train Network": "mean",
            "Pretrain Max Trainer Memory": "mean",
            "Train Max Trainer Memory": "mean",
        }
    )
    .reset_index()
)

# Plot the communication cost and memory for different trainers
plt.figure()
plt.plot(
    gcfl_grouped["Number of Trainers"],
    gcfl_grouped["Pretrain Network"],
    label="Pretrain Network",
    color="tab:blue",
    marker="o",
)
plt.plot(
    gcfl_grouped["Number of Trainers"],
    gcfl_grouped["Train Network"],
    label="Train Network",
    color="tab:orange",
    marker="o",
)
plt.xlabel("Number of Trainers")
plt.ylabel("Communication Cost (Network)")
plt.title("GCFL Communication Cost with Different Trainers")
plt.legend()
plt.show()

# Plot memory (Max Trainer Memory)
plt.figure()
plt.plot(
    gcfl_grouped["Number of Trainers"],
    gcfl_grouped["Pretrain Max Trainer Memory"],
    label="Pretrain Max Trainer Memory",
    color="tab:green",
    marker="x",
)
plt.plot(
    gcfl_grouped["Number of Trainers"],
    gcfl_grouped["Train Max Trainer Memory"],
    label="Train Max Trainer Memory",
    color="tab:red",
    marker="x",
)
plt.xlabel("Number of Trainers")
plt.ylabel("Memory (Max Trainer Memory)")
plt.title("GCFL Memory Usage with Different Trainers")
plt.legend()
plt.show()
