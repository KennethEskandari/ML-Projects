from CleanData import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Defining Features
features = ['Sex','Age']
X = data[features]
y = data['Survived']

#Training a Simple Model 
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


#Testing 
score = model.score(X_test,y_test)
print (f"Model Score: {score}")


