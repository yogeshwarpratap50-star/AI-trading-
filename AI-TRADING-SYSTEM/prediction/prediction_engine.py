from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from models.base_model import TradingModel
from models.model_registry import ModelRegistry
from models.random_forest_model import RandomForestTradingModel
from models.xgboost_model import XGBoostTradingModel

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    action: str
    confidence: float
    predicted_class: int
    model_name: str


# HOLD sentinel returned when no model is available — avoids crashing callers
_HOLD_NO_MODEL = PredictionResult(action="HOLD", confidence=0.0, predicted_class=0, model_name="none")


class PredictionEngine:
    """Generates BUY, SELL, HOLD predictions with confidence scores."""

    def __init__(self, model: TradingModel | None = None, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self.model = model

    def load_latest_model(self) -> TradingModel:
        entry = self.registry.latest()
        if entry is None:
            raise RuntimeError(
                "No registered model found. Run --train-models first."
            )
        model = self._model_for_name(entry["model_name"])
        model.load(Path(entry["model_path"]))
        self.model = model
        return model

    def predict(self, latest_features: pd.DataFrame) -> PredictionResult:
        # ── Input validation (CRITICAL FIX) ─────────────────────────────
        if latest_features is None or latest_features.empty:
            _logger.warning("predict() received empty DataFrame — returning HOLD")
            return _HOLD_NO_MODEL

        # ── Model load ───────────────────────────────────────────────────
        try:
            model = self.model or self.load_latest_model()
        except RuntimeError as exc:
            _logger.warning("No model available (%s) — returning HOLD", exc)
            return _HOLD_NO_MODEL

        # ── Feature alignment ────────────────────────────────────────────
        if hasattr(model, "feature_columns") and model.feature_columns:
            expected = set(model.feature_columns)
            actual = set(latest_features.columns)
            missing = expected - actual
            if missing:
                _logger.warning(
                    "Feature mismatch — missing %d columns (%s…) — returning HOLD",
                    len(missing),
                    list(missing)[:3],
                )
                return _HOLD_NO_MODEL
            # Keep only expected columns in the right order
            latest_features = latest_features[list(model.feature_columns)]

        # ── Prediction ───────────────────────────────────────────────────
        try:
            probabilities = model.predict_proba(latest_features)
        except Exception as exc:
            _logger.error("model.predict_proba() failed: %s — returning HOLD", exc)
            return _HOLD_NO_MODEL

        if probabilities is None or probabilities.empty or len(probabilities.columns) == 0:
            _logger.warning("predict_proba returned empty result — returning HOLD")
            return _HOLD_NO_MODEL

        positive_column = "1" if "1" in probabilities.columns else probabilities.columns[-1]
        try:
            confidence_up = float(probabilities[positive_column].iloc[-1])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            _logger.error("Could not extract confidence: %s — returning HOLD", exc)
            return _HOLD_NO_MODEL

        # Guard against NaN / inf
        if not (0.0 <= confidence_up <= 1.0):
            confidence_up = 0.5

        predicted_class = int(confidence_up >= 0.5)
        confidence = confidence_up if predicted_class == 1 else 1 - confidence_up
        action = (
            "BUY"  if predicted_class == 1 and confidence >= 0.6 else
            "SELL" if predicted_class == 0 and confidence >= 0.6 else
            "HOLD"
        )
        return PredictionResult(
            action=action,
            confidence=round(confidence * 100, 2),
            predicted_class=predicted_class,
            model_name=model.model_name,
        )

    @staticmethod
    def _model_for_name(model_name: str) -> TradingModel:
        if model_name == RandomForestTradingModel.model_name:
            return RandomForestTradingModel()
        if model_name == XGBoostTradingModel.model_name:
            return XGBoostTradingModel()
        raise ValueError(f"Unsupported model type: {model_name}")
