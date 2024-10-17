import pandas as pd

file1 = "trainer10_newarc.csv"
file2 = "10.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1_filtered = df1[df1["Algorithm"] != "SelfTrain"]

df_combined = pd.concat([df1_filtered, df2])
df_combined_sorted = df_combined.sort_values(by=["Dataset", "Algorithm"])
output_file = "trainer10_updated_selftrain.csv"
df_combined_sorted.to_csv(output_file, index=False)
