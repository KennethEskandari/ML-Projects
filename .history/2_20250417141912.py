import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re 

#Loading Data 
df = pd.read_csv('/Users/kennetheskandari/ML-Projects/dummy_movie_reviews.csv')
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W','',text)
    text = re.sub(r'\s+','',text)