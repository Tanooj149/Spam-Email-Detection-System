import re
import string
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

email = input("Enter Email: ")

email = clean_text(email)

vector = vectorizer.transform([email])

prediction = model.predict(vector)[0]

if prediction == 1:
    print("🚨 Prediction : SPAM")
else:
    print("✅ Prediction : HAM")