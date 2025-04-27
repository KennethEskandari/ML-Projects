import pandas as pd 

data = pd.read_csv('titanic/train.csv')
data.head()

print(data.info())
print(data.describe())
print(data.isnull().sum())
