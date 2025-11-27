from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="FLAN Translation API")

# Charger le modèle
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

# GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextInput(BaseModel):
    text: str


# ===============================
#       HEALTH CHECK
# ===============================
@app.get("/health")
def health():
    return {"status": "running", "device": str(device)}


# ===============================
#   EN → AR (traduction anglais → arabe)
# ===============================
@app.post("/translate/en-ar")
def translate_en_ar(input: TextInput):

    prompt = f"""
    Translate the following sentence from English to Arabic:
    "{input.text}"
    """

    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **tokens,
        max_length=100,
        num_beams=5,
        early_stopping=True
    )

    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translated_text": translation}


# ===============================
#   AR → FR (traduction arabe → français)
# ===============================
@app.post("/translate/ar-fr")
def translate_ar_fr(input: TextInput):

    prompt = f"""
    Translate the following sentence from Arabic to French:
    "{input.text}"
    """

    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **tokens,
        max_length=100,
        num_beams=5,
        early_stopping=True
    )

    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translated_text": translation}
