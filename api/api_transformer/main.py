from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Transformer API", version="2.0")

# =========================================================
# === CONFIGURATION DU MODÈLE =============================
# =========================================================

# Nom EXACT de ton modèle Hugging Face
HF_MODEL_NAME = "zouhour792/callcenterai_mopls"

print("☁️ Téléchargement du modèle depuis HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

# GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================================================
# === SCHÉMA D'ENTRÉE =====================================
# =========================================================


class Ticket(BaseModel):
    text: str


# =========================================================
# === ROUTES ==============================================
# =========================================================


@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device)}


@app.post("/predict")
def predict(ticket: Ticket):
    if not ticket.text.strip():
        return {"error": "Texte vide."}

    inputs = tokenizer(
        ticket.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    return {
        "label": model.config.id2label[label_id],
        "confidence": round(confidence, 3),
        "source": "huggingface"
    }
