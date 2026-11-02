# ---------- Stage 1: train the model ----------
FROM python:3.10.15-slim-bookworm AS trainer

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first: this layer is reused unless requirements change
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt

# Training inputs: the DVC-restored dataset, params and training code
COPY params.yaml .
COPY training/ training/
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv

ARG MODEL_VERSION=1.0.0
RUN python -m training.train \
      --data-path data/raw/breast_cancer.csv \
      --output-dir /build/artifacts \
      --model-version "${MODEL_VERSION}" \
      --metrics-path /build/artifacts/metrics.json

# ---------- Stage 2: serving image ----------
FROM python:3.10.15-slim-bookworm AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unprivileged user that will run the service
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# Serving dependencies only (no PyYAML, no DVC, no training code, no dataset)
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only the model artifacts are taken from the training stage
COPY --from=trainer /build/artifacts/model.joblib /build/artifacts/model_card.json artifacts/

# Application code last: editing it does not reinstall dependencies
COPY app/ app/

# Build arguments: recorded as labels and exposed as environment variables
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="breast-cancer-triage-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="FastAPI breast cancer triage screening service (educational)"

ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    MODEL_PATH=/app/artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_PORT=6379

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
