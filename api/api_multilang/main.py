from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Multilingual Ticket Classifier", version="1.0")

# === Charger le classifieur fine-tuné ===
HF_MODEL_NAME = "zouhour792/callcenterai_mopls"
clf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
clf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

# === Charger FLAN-T5 pour la traduction ===
FLAN = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(FLAN)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model.to(device)
flan_model.to(device)

class Ticket(BaseModel):
    text: str

def translate(text: str):
    """Détection simple de l'arabe + traduction vers anglais."""
    if any("\u0600" <= c <= "\u06FF" for c in text):
        prompt = f"Translate this to English: {text}"
        inputs = flan_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = flan_model.generate(**inputs, max_length=128)
        translated = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated, "translated_arabic"
    return text, "original"

@app.post("/predict")
def predict(ticket: Ticket):

    english_text, status = translate(ticket.text)

    inputs = clf_tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = clf_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs).item()
        confidence = probs[0][label_id].item()

    return {
        "input": ticket.text,
        "english_processed": english_text,
        "label": clf_model.config.id2label[label_id],
        "confidence": round(confidence, 3),
        "processing": status
    }
