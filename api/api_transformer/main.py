from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(title="Transformer API", version="2.0")

# =========================================================
# === CONFIGURATION DU MODÈLE =============================
# =========================================================

# Nom du modèle sur Hugging Face Hub
HF_MODEL_NAME = "callcenterai_mopls"
# Chemin local de secours
LOCAL_MODEL_PATH = "models/transformer"

# Choisir le mode de chargement
if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
    print("📁 Chargement du modèle local...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
else:
    print(f"☁️ Chargement du modèle depuis Hugging Face Hub : {HF_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

# Gestion du périphérique (GPU si dispo)
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
    """Vérifie que le service est opérationnel."""
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
def predict(ticket: Ticket):
    """Prédit la catégorie d'un ticket de support (texte)."""
    if not ticket.text.strip():
        return {"error": "Texte vide."}

    # Préparation de l'entrée
    inputs = tokenizer(
        ticket.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Prédiction
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    label = model.config.id2label[label_id]

    response = {
        "label": label,
        "confidence": round(confidence, 3),
        "source": "local" if os.path.exists(LOCAL_MODEL_PATH) else "huggingface"
    }

    # Optionnel : avertissement si la confiance est faible
    if confidence < 0.7:
        response["needs_review"] = True

    return response
