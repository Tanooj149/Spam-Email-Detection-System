import re
import string
import pandas as pd

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = " ".join(text.split())

    return text


if __name__ == "__main__":
    df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "email"]

    df["clean_email"] = df["email"].apply(clean_text)

    print(df[["label", "clean_email"]].head())