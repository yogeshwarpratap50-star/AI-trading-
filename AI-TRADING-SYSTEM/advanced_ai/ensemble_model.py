"""
Ensemble Model — Phase 11A.

Combines XGBoost and Random Forest via weighted probability averaging.
Falls back gracefully if one model fails to predict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.base_model import TradingModel
from models.random_forest_model import RandomForestTradingModel
from models.xgboost_model import XGBoostTradingModel
from prediction.prediction_engine import PredictionResult


@dataclass
class EnsembleConfig:
    rf_weight: float = 0.4
    xgb_weight: float = 0.6
    min_confidence: float = 0.6    # below this → HOLD


class EnsembleModel:
    """
    Weighted ensemble of RandomForest + XGBoost.

    Usage
    -----
    ens = EnsembleModel()
    ens.load(rf_path="models/rf.pkl", xgb_path="models/xgb.pkl")
    result = ens.predict(features_df)
    """

    model_name = "Ensemble(RF+XGB)"

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self._rf = RandomForestTradingModel()
        self._xgb = XGBoostTradingModel()
        self._rf_loaded = False
        self._xgb_loaded = False
        self.logger = logging.getLogger(__name__)

    def load(self, rf_path: str | Path | None = None, xgb_path: str | Path | None = None) -> None:
        """Load persisted model files (optional — works without if models trained in-session)."""
        if rf_path and Path(rf_path).exists():
            self._rf.load(Path(rf_path))
            self._rf_loaded = True
        if xgb_path and Path(xgb_path).exists():
            self._xgb.load(Path(xgb_path))
            self._xgb_loaded = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._rf.train(X, y)
        self._xgb.train(X, y)
        self._rf_loaded = True
        self._xgb_loaded = True

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Weighted probability average → BUY / SELL / HOLD."""
        proba_positive = 0.0
        total_weight = 0.0

        for model, loaded, weight in [
            (self._rf, self._rf_loaded, self.config.rf_weight),
            (self._xgb, self._xgb_loaded, self.config.xgb_weight),
        ]:
            if not loaded:
                continue
            try:
                proba = model.predict_proba(features)
                col = "1" if "1" in proba.columns else proba.columns[-1]
                proba_positive += float(proba[col].iloc[-1]) * weight
                total_weight += weight
            except Exception as exc:
                self.logger.warning("Ensemble: %s failed: %s", model.model_name, exc)

        if total_weight == 0:
            return PredictionResult(action="HOLD", confidence=0.0, predicted_class=0, model_name=self.model_name)

        p = proba_positive / total_weight
        predicted_class = int(p >= 0.5)
        confidence = p if predicted_class == 1 else 1 - p
        action = (
            "BUY" if predicted_class == 1 and confidence >= self.config.min_confidence
            else "SELL" if predicted_class == 0 and confidence >= self.config.min_confidence
            else "HOLD"
        )
        return PredictionResult(
            action=action,
            confidence=round(confidence * 100, 2),
            predicted_class=predicted_class,
            model_name=self.model_name,
        )

    def feature_importances(self) -> pd.DataFrame:
        """Weighted average feature importance from both models."""
        rows = []
        for model, loaded, weight in [
            (self._rf, self._rf_loaded, self.config.rf_weight),
            (self._xgb, self._xgb_loaded, self.config.xgb_weight),
        ]:
            if not loaded:
                continue
            try:
                # Access sklearn model's feature_importances_ attribute
                inner = getattr(model, "model", None)
                feature_cols = getattr(model, "feature_columns", [])
                if inner is not None and hasattr(inner, "feature_importances_") and feature_cols:
                    imp = pd.DataFrame({
                        "feature": feature_cols,
                        "importance": inner.feature_importances_ * weight,
                    })
                    rows.append(imp)
            except Exception:
                pass
        if not rows:
            return pd.DataFrame(columns=["feature", "importance"])
        combined = pd.concat(rows).groupby("feature", as_index=False)["importance"].sum()
        total = combined["importance"].sum()
        if total > 0:
            combined["importance"] /= total
        return combined.sort_values("importance", ascending=False).reset_index(drop=True)
