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
Y = df['sentiment']