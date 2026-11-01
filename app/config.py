"""Service configuration, read only from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    decision_threshold: float
    log_level: str
    redis_host: str | None
    redis_port: int
    model_version: str | None
    git_sha: str


def load_settings() -> Settings:
    threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"DECISION_THRESHOLD must be between 0 and 1, got {threshold}")
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "artifacts/model.joblib")),
        decision_threshold=threshold,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        redis_host=os.getenv("REDIS_HOST") or None,
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        model_version=os.getenv("MODEL_VERSION") or None,
        git_sha=os.getenv("GIT_SHA", "unknown"),
    )
