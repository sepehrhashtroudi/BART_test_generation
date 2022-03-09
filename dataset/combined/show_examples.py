import pandas as pd
example = pd.read_csv( "train_combined.csv", header=None)
print(example.loc[0,0])
print(example.loc[0,1])
