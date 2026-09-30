from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

app = FastAPI(title="Spam Email Detection API")

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


class EmailRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Spam Email Detection API Running Successfully"}


@app.post("/predict")
def predict(data: EmailRequest):
    text = clean_text(data.message)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Ham"

    return {
        "email": data.message,
        "prediction": result
    }