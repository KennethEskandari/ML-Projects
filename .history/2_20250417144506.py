import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re 

#Loading Data 
df = pd.read_csv('/Users/kennetheskandari/ML-Projects/dummy_movie_reviews.csv')
print(df.head())

#Simply cleaning data 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W','',text)
    text = re.sub(r'\s+','',text)
    return text

df['cleaned'] = df['review'].apply(clean_text)

#This is the cool shit, we are going to make it into numbers now
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transformation(df['cleaned'])
y = df['sentiment']

#Setting Up Data for Training and Testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training 
model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predit(X_test)
print(classification_report(y_test,y_pred))

#Using The Model 
def predict_sentiment():
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'

print(predict_sentiment())