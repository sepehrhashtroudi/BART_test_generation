import pandas as pd
methods = pd.read_csv( "train_combined.methods", sep="\n", header=None)
tests = pd.read_csv("train_combined.methods", sep="\n", header=None)
df = pd.concat([methods, tests], axis=1)
df.to_csv("train_combined.csv", encoding='utf-8', index=False)
methods = pd.read_csv( "eval_combined.methods", sep="\n", header=None)
tests = pd.read_csv("eval_combined.methods", sep="\n", header=None)
df = pd.concat([methods, tests], axis=1)
df.to_csv("eval_combined.csv", encoding='utf-8', index=False)
