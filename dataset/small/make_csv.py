import pandas as pd
methods = pd.read_csv( "train_final_500.methods", sep="\n",lineterminator="\n", header=None)
tests = pd.read_csv("train_final_500.tests", sep="\n",lineterminator="\n", header=None)
df = pd.concat([methods, tests], axis=1)
df.to_csv("train_final_500.csv", encoding='utf-8', index=False)
methods = pd.read_csv( "eval_final_500.methods", sep="\n",lineterminator="\n", header=None)
tests = pd.read_csv("eval_final_500.tests", sep="\n",lineterminator="\n", header=None)
df = pd.concat([methods, tests], axis=1)
df.to_csv("eval_final_500.csv", encoding='utf-8', index=False)
