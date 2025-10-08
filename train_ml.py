# train_ml.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from utils import clean_text

# Load dataset
df = pd.read_csv("C:/Users/Admin/Desktop/Jansi/Project/Multi model article classification/Dataset/train.csv", header=None, names=["label", "title", "desc"])
df["text"] = (df["title"].fillna("") + " " + df["desc"].fillna("")).apply(clean_text)

X = df["text"]
y = df["label"]

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# Save model + vectorizer
joblib.dump(model, "models/ml_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("✅ ML model and TF-IDF saved in models/ folder")
