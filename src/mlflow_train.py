import re
import string
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

mlflow.set_tracking_uri(f"sqlite:///{Path.cwd().joinpath('mlflow.db').as_posix()}")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


# Load dataset
df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "email"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["email"] = df["email"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["email"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Spam_Email_Detection")

with mlflow.start_run():

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "spam_model")

    print("Accuracy:", accuracy)

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")