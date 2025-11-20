from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="FLAN-T5 API", version="1.0")

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

class Input(BaseModel):
    text: str

@app.post("/translate/ar")
def translate_to_ar(input: Input):
    prompt = f"Translate this to Arabic: {input.text}"
    tokens = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**tokens, max_length=100)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translated_text": translated}
