import matplotlib.pyplot as plt
import pandas as pd

file_path = "NC_papers100M.csv"
data = pd.read_csv(file_path)
# data = {
#     'Batch Size': [16, 32, 64, -1],
#     'Train Time': [620510.904, 625067.836, 646383.4789999999, 625576.189],
#     'Average Test Accuracy': [0.4148867676286986, 0.4148867676286986, 0.41487743657214304, 0.37154400992824416]
# }

# Create the DataFrame
df = pd.DataFrame(data)

# Plot Train Time vs Batch Size
plt.figure()
plt.plot(
    df["Batch Size"], df["Train Time"], marker="o", color="skyblue", label="Train Time"
)
plt.xlabel("Batch Size")
plt.ylabel("Train Time")
plt.title("Train Time vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Accuracy vs Batch Size
plt.figure()
plt.plot(
    df["Batch Size"],
    df["Average Test Accuracy"],
    marker="x",
    color="orange",
    label="Accuracy",
)
plt.xlabel("Batch Size")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.show()
