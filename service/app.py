from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import sys
import os

sys.path.append("assets")
from model import SentimentClassifier

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "assets/best_model.pt"
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentimentClassifier(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# FastAPI app
app = FastAPI(title="Sentiment Classifier", version="1.0")

class InText(BaseModel):
    text: str

class OutScore(BaseModel):
    probability: float
    label: str

@app.post("/predict", response_model=OutScore)
def predict(input_data: InText):
    encoded = tokenizer(
        input_data.text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    ids = encoded["input_ids"].to(DEVICE)
    mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logit = model(ids, mask).item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()

    return {
        "probability": prob,
        "label": "positive" if prob >= 0.5 else "negative"
    }
