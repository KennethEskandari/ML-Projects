from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 

#Loading Data For Machine Learning 
Iris = load_iris()
X = Iris.data
Y = Iris.target

#Splitting Data Into Train and Test Sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Loading The Model 
model = RandomForestClassifier()
model.fit(X_train, Y_train)


