import pandas as pd
import gzip
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def parse_gzip_to_df(path):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)
    
    df = {}
    for i, data in enumerate(parse(path)):
        df[i] = data
    return pd.DataFrame.from_dict(df, orient='index')

# Function to label reviews based on their ratings
def label_review(overall):
    if overall >= 4:
        return 'positive'
    elif overall == 3:
        return 'neutral'
    else:
        return 'negative'

file_path = r"C:\Users\ASUS\Desktop\polynomial\Cell_Phones_and_Accessories_5.json.gz"


reviews = parse_gzip_to_df(file_path)
reviews.dropna(subset=['summary'], inplace=True)  
reviews['review_label'] = reviews['overall'].apply(label_review)

X_train, X_test, y_train, y_test = train_test_split(reviews['summary'], reviews['review_label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Training the model
pipeline.fit(X_train, y_train)

# Predicting on test data
y_pred = pipeline.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


model_file_path = r'C:\Users\ASUS\Desktop\polynomial\Cell_Phones_and_Accessories_5.json\sentiment_analysis_model.pkl'
joblib.dump(pipeline, model_file_path)
print(f"Model saved to {model_file_path}")

def predict_sentiment(text):
    prediction = pipeline.predict([text])
    return prediction[0]



