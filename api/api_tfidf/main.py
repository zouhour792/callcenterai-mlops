from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CollectorRegistry
import joblib

app = FastAPI(title="TF-IDF API", version="1.1")

model = joblib.load("../models/tfidf.joblib")

registry = CollectorRegistry()
PRED_REQUESTS = Counter(
    "tfidf_requests_total", "Nombre total de requêtes TF-IDF", registry=registry
)


class Ticket(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(ticket: Ticket):
    try:
        PRED_REQUESTS.inc()
        pred = model.predict([ticket.text])[0]
        return {"label": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return generate_latest(registry)
