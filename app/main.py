"""FastAPI service for breast-cancer triage predictions (Meridian Diagnostics Lab)."""
import logging
import sys
from contextlib import asynccontextmanager

import sklearn
from fastapi import FastAPI

from app.config import load_settings
from app.model import ModelService
from app.schemas import FEATURE_NAMES, PatientFeatures, PredictionResponse

settings = load_settings()

logging.basicConfig(
    stream=sys.stdout,
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    uvicorn_logger = logging.getLogger(name)
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True
logger = logging.getLogger("triage")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = ModelService.load(settings.model_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Cannot start the service: %s", exc)
        raise
    state["model"] = service
    logger.info(
        "Loaded %s model (version %s) from %s, decision threshold %.2f",
        service.card.get("model_family"), current_model_version(service),
        settings.model_path, settings.decision_threshold,
    )
    yield
    state.clear()


def current_model_version(service: ModelService) -> str:
    return settings.model_version or service.card.get("model_version", "unknown")


app = FastAPI(title="Breast Cancer Triage API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    service = state["model"]
    probability = service.malignant_probability(features.model_dump())
    label = "malignant" if probability >= settings.decision_threshold else "benign"
    logger.info("prediction label=%s malignant_probability=%.4f", label, probability)
    return PredictionResponse(
        label=label,
        malignant_probability=round(probability, 4),
        threshold=settings.decision_threshold,
        model_version=current_model_version(service),
    )


@app.get("/model/info")
def model_info():
    service = state["model"]
    card = service.card
    return {
        "model_version": current_model_version(service),
        "model_family": card.get("model_family"),
        "sklearn_version": sklearn.__version__,
        "trained_with_sklearn_version": card.get("sklearn_version"),
        "git_commit": settings.git_sha,
        "decision_threshold": settings.decision_threshold,
        "feature_count": len(FEATURE_NAMES),
        "test_metrics": card.get("metrics", {}).get("test"),
    }


@app.get("/stats")
def stats():
    return {"predictions_served": None, "counter_enabled": False,
            "note": "Redis request counter is added in Milestone 4"}
