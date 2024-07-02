import pandas as pd
import gzip
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  # Correct import
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Function to parse the gzip file and create DataFrame
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

# Path to the gzip JSON file
file_path = r"C:\Users\ASUS\Desktop\FUN SIMPLY\polynomial\Cell_Phones_and_Accessories_5.json.gz"

# Create DataFrame
reviews = parse_gzip_to_df(file_path)
reviews.dropna(subset=['reviewText'], inplace=True)  # Handle missing values in 'reviewText' column
reviews['review_label'] = reviews['overall'].apply(label_review)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(reviews['reviewText'], reviews['review_label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF Vectorizer and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf',  RandomForestClassifier(n_estimators=100, random_state=42))
  # You can replace LogisticRegression with other classifiers
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Predict the test set
y_pred = pipeline.predict(X_test)

# Evaluate the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(pipeline, 'sentiment_classifier_rf.pkl')
