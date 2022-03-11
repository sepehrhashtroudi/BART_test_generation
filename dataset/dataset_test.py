import pandas as pd
path = "train"
df = pd.read_csv("train_final.csv")

for index, row in df.iterrows():
  if "shouldCreateAddress" in row["0.1"]:
    print(row["0.1"])
    print(index)

