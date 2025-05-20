import os 
import pandas as pd
from model import path 

train_df = pd.read_csv(os.path.join(path, "train.csv"), low_memory = "False")
store_df = pd.read_csv(os.path.join(path, "store.csv"))
df = pd.merge(train_df, store_df, on = "Store", how = "left")

print(df)

