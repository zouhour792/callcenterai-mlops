from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Transformer API", version="2.0")

MODEL_PATH = "models/transformer"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class Ticket(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(ticket: Ticket):
    inputs = tokenizer(ticket.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    label = model.config.id2label[label_id]
    resp = {"label": label, "confidence": round(confidence, 3)}

    # Avertir si la confiance est faible
    if confidence < 0.7:
        resp["needs_review"] = True
    return resp
