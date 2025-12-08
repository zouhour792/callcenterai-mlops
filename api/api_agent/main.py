from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Ajoutez cette ligne
from pydantic import BaseModel
import requests, re
from langdetect import detect, DetectorFactory
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DetectorFactory.seed = 0

app = FastAPI(title="Agent IA API", version="3.0")

# Ajoutez cette configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En développement, en production spécifiez les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# URLs (pour Docker)
TFIDF_URL = "http://tfidf-svc:8001/predict"
TRANSFORMER_URL = "http://transformer-svc:8002/predict"


# Session avec retry
session = requests.Session()
retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)


class Ticket(BaseModel):
    text: str


def scrub_pii(text: str) -> str:
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{8,}\b", "[PHONE]", text)
    return text.strip()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def route_ticket(ticket: Ticket):
    clean = scrub_pii(ticket.text)
    if not clean:
        raise HTTPException(status_code=400, detail="Texte vide après nettoyage")

    try:
        lang = detect(clean)

        # Routage intelligent
        if lang == "fr":
            resp = session.post(
                TRANSFORMER_URL, json={"text": clean}, timeout=15
            ).json()
            resp["model"] = "Transformer"
        elif lang == "en" and len(clean) < 80:
            resp = session.post(TFIDF_URL, json={"text": clean}, timeout=10).json()
            resp["model"] = "TF-IDF"
        else:
            resp = session.post(
                TRANSFORMER_URL, json={"text": clean}, timeout=15
            ).json()
            resp["model"] = "Transformer"

        # Correction heuristique FR
        if lang == "fr" and any(
            word in clean.lower() for word in ["accès", "compte", "connexion"]
        ):
            resp["label"] = "Access"

        resp["lang"] = lang
        return resp

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Services indisponibles")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
