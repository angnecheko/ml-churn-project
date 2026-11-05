import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        from src.train import train_model

        train_model()

    return joblib.load(MODEL_PATH)


app = FastAPI(title="Churn Prediction API", version="1.0.0")
model = load_model()


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/model/info")
def model_info():
    return {"model": "Random Forest", "version": "1.0.0"}


@app.post("/predict")
def predict(data: ClientData):
    start = time.time()

    feature_names = [
        "age",
        "anciennete",
        "revenu",
        "nb_reclamations",
        "montant_mensuel",
        "retards_paiement",
    ]
    values = pd.DataFrame(
        [[
            data.age,
            data.anciennete,
            data.revenu,
            data.nb_reclamations,
            data.montant_mensuel,
            data.retards_paiement,
        ]],
        columns=feature_names,
    )

    prediction = model.predict(values)[0]
    probability = model.predict_proba(values)[0][1]
    duration = time.time() - start

    print(
        f"prediction={prediction}, "
        f"probability={probability:.4f}, "
        f"duration={duration:.4f}s"
    )

    return {
        "prediction": "CHURN" if prediction == 1 else "NO_CHURN",
        "probability": round(float(probability), 4),
        "model_version": "1.0.0",
        "duration_ms": round(duration * 1000, 2),
    }
# Ajout de prediction pour le modèle de régression logistique: dummy change