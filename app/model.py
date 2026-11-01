"""Loads the trained pipeline and its model card, and computes malignant probabilities."""
import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import FEATURE_NAMES

MALIGNANT = 0


class ModelService:
    def __init__(self, model, card):
        self.model = model
        self.card = card
        self.malignant_index = list(model.classes_).index(MALIGNANT)

    @classmethod
    def load(cls, model_path: Path) -> "ModelService":
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        card_path = model_path.with_name("model_card.json")
        if not card_path.is_file():
            raise FileNotFoundError(f"Model card not found at {card_path}")
        card = json.loads(card_path.read_text())
        if card.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Model card feature names do not match the API schema")
        return cls(joblib.load(model_path), card)

    def malignant_probability(self, features: dict) -> float:
        row = pd.DataFrame([features], columns=FEATURE_NAMES)
        return float(self.model.predict_proba(row)[0, self.malignant_index])
