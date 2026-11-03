# Breast Cancer Triage Service

FastAPI service that classifies breast-cancer fine-needle-aspirate cell-nucleus
measurements as **likely malignant** or **likely benign** (malignant = positive class),
packaged as a Docker image and run next to Redis with Docker Compose.
Educational MLOps project, **not a medical device**.

- Docker Hub: https://hub.docker.com/r/azka370/breast-cancer-triage-api (tags `1.0.0`, `latest`; `linux/amd64` + `linux/arm64`)
- Dataset: `sklearn.datasets.load_breast_cancer`, exported to `data/raw/breast_cancer.csv` and versioned with DVC (v2 = 539 rows)
- Model: scikit-learn `Pipeline` (StandardScaler + LogisticRegression), selected by 5-fold CV malignant recall, seed 42

## Repository layout

| Path | Purpose |
|---|---|
| `training/prepare_data.py` | Exports the dataset to CSV and validates it (no nulls, 30 features) |
| `training/train.py` | Trains and cross-validates two model families, selects the best, writes `model.joblib` and `model_card.json` |
| `params.yaml`, `dvc.yaml`, `dvc.lock` | Seed/hyperparameters and the DVC training pipeline |
| `data/raw/breast_cancer.csv.dvc`, `dvc-remote/` | DVC pointer file and the local DVC remote |
| `app/` | FastAPI service (`/health`, `/predict`, `/model/info`, `/stats`) |
| `Dockerfile`, `Dockerfile.naive`, `.dockerignore` | Production multi-stage image and the naive baseline |
| `docker-compose.yml`, `docker-compose.hub.yml`, `.env.example` | Local build stack and the published-image stack |

## 1. Set up and restore the data

Requires Python 3.10, Docker Desktop (or Docker Engine) with Compose v2, and Git.

```bash
git clone https://github.com/azka09000/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
dvc pull
```

`dvc pull` restores `data/raw/breast_cancer.csv` from the committed local remote `dvc-remote/`.

## 2. Run the training pipeline

```bash
dvc repro
dvc metrics show
```

Produces `artifacts/model.joblib`, `artifacts/model_card.json` and `metrics.json`.
The script exits with a non-zero status if the selected model's test malignant recall is below 0.95.

## 3. Run the API locally (optional)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

Open http://localhost:8090/docs, or send the sample record:

```bash
curl -X POST http://localhost:8090/predict -H "Content-Type: application/json" -d @sample_payload.json
```

## 4. Build the Docker image

```bash
docker build \
  --build-arg MODEL_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t breast-cancer-triage-api:1.0.0 .

docker run -d --name triage-api -p 8090:8000 breast-cancer-triage-api:1.0.0
```

The image trains the model in a separate stage, runs as a non-root user, exposes port 8000
and has a `HEALTHCHECK` on `/health`.

## 5. Start the Compose stack (API + Redis)

```bash
cp .env.example .env
# edit .env: set DOCKERHUB_USERNAME (and GIT_SHA=$(git rev-parse --short HEAD))
docker compose up -d --build
docker compose ps
curl http://localhost:8090/stats
```

Stop it with `docker compose down` (keeps the Redis volume) or `docker compose down -v` (removes it).

## 6. Push to Docker Hub

```bash
docker login -u <your-dockerhub-username>
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg MODEL_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t <your-dockerhub-username>/breast-cancer-triage-api:1.0.0 \
  -t <your-dockerhub-username>/breast-cancer-triage-api:latest \
  --push .
```

## 7. Run from Docker Hub only

In an empty directory containing only `docker-compose.hub.yml` and a `.env` with
`DOCKERHUB_USERNAME=azka370`:

```bash
docker compose -f docker-compose.hub.yml up -d
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/app/artifacts/model.joblib` | Model artifact location |
| `DECISION_THRESHOLD` | `0.5` | Malignant probability cutoff |
| `LOG_LEVEL` | `INFO` | Logging verbosity (stdout only) |
| `REDIS_HOST` / `REDIS_PORT` | unset / `6379` | Request counting is enabled only when `REDIS_HOST` is set |
| `MODEL_VERSION`, `GIT_SHA` | build arguments | Reported by `/model/info` |
