import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


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

# Clean emails
df["email"] = df["email"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["email"])

print("Feature Matrix Shape:", X.shape)
print("Vocabulary Size:", len(vectorizer.vocabulary_))