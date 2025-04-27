from Data import data

data['Age'].fillna(data['Age'].median(),inplace=True)
data['Sex']=data['Sex'].map({'male':0,'female':1})
data.drop(columns=['Cabin','Ticket','Name'],inplace=True)

print(data.info())
print(data.describe())
print(data.isnull().sum())
