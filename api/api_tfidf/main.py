from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CollectorRegistry
import joblib
import os

app = FastAPI(title="TF-IDF API", version="1.1")

# Chemin correct pour le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis: {MODEL_PATH}")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    model = None

registry = CollectorRegistry()
PRED_REQUESTS = Counter(
    "tfidf_requests_total", "Nombre total de requêtes TF-IDF", registry=registry
)

class Ticket(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(ticket: Ticket):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé")
        
        PRED_REQUESTS.inc()
        
        # Prédiction
        pred = model.predict([ticket.text])[0]
        
        # Calcul du score de confiance (probabilité si disponible)
        try:
            # Si le modèle a predict_proba
            proba = model.predict_proba([ticket.text])[0]
            confidence = max(proba)  # Probabilité maximale
        except:
            # Sinon, valeur par défaut
            confidence = 0.95
        
        return {
            "label": pred,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return generate_latest(registry)