"""Request and response schemas for the triage API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

_BASE = [
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURE_NAMES = (
    [f"mean_{name}" for name in _BASE]
    + [f"{name}_error" for name in _BASE]
    + [f"worst_{name}" for name in _BASE]
)

PatientFeatures = create_model(
    "PatientFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{name: (float, Field(..., ge=0, allow_inf_nan=False)) for name in FEATURE_NAMES},
)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: Literal["malignant", "benign"]
    malignant_probability: float
    threshold: float
    model_version: str
