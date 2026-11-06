import logging
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

# add-on start here

# Fichier partagé avec Logstash
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "api.log"

logger.setLevel(logging.INFO)

# Évite d'ajouter plusieurs fois le même gestionnaire
if not any(
    isinstance(handler, logging.FileHandler)
    and handler.baseFilename == str(LOG_FILE)
    for handler in logger.handlers
):
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    )
    logger.addHandler(file_handler)

# end of add-on

MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Churn Prediction API",
    version=MODEL_VERSION,
)

# Chemin indépendant du dossier depuis lequel on lance l'API
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info(
        "Model loaded: model_version=%s",
        MODEL_VERSION,
    )
except Exception:
    logger.exception(
        "Model loading error: model_version=%s",
        MODEL_VERSION,
    )
    raise


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
        "version": MODEL_VERSION,
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/model/info")
def model_info():
    return {
        "model": "Random Forest",
        "version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(data: ClientData, request: Request):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "Prediction request: client_ip=%s model_version=%s",
        client_ip,
        MODEL_VERSION,
    )

    try:
        values = [[
            data.age,
            data.anciennete,
            data.revenu,
            data.nb_reclamations,
            data.montant_mensuel,
            data.retards_paiement,
        ]]

        prediction = int(model.predict(values)[0])
        probability = float(model.predict_proba(values)[0][1])

        duration_ms = (time.perf_counter() - start) * 1000
        result = "CHURN" if prediction == 1 else "NO_CHURN"

        logger.info(
            "Prediction success: client_ip=%s model_version=%s "
            "prediction=%s probability=%.4f duration_ms=%.2f",
            client_ip,
            MODEL_VERSION,
            result,
            probability,
            duration_ms,
        )

        return {
            "prediction": result,
            "probability": round(probability, 4),
            "model_version": MODEL_VERSION,
            "duration_ms": round(duration_ms, 2),
        }

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "Prediction error: client_ip=%s model_version=%s "
            "duration_ms=%.2f",
            client_ip,
            MODEL_VERSION,
            duration_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la prédiction.",
        ) from None


Instrumentator().instrument(app).expose(app)