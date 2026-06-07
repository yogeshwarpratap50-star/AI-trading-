from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class TradingModel(ABC):
    """Base wrapper for supervised trading classifiers."""

    model_name: str

    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_columns: list[str] = []

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""

    def predict(self, features: pd.DataFrame) -> list[int]:
        self._require_model()
        return [int(value) for value in self.model.predict(features[self.feature_columns])]

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        self._require_model()
        probabilities = self.model.predict_proba(features[self.feature_columns])
        return pd.DataFrame(probabilities, columns=[str(column) for column in self.model.classes_])

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, include_roc_auc: bool = False) -> dict[str, float]:
        predictions = self.predict(x_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }
        if include_roc_auc and len(set(y_test)) > 1:
            probabilities = self.predict_proba(x_test)
            positive_column = "1" if "1" in probabilities.columns else probabilities.columns[-1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, probabilities[positive_column])), 4)
        return metrics

    def save(self, path: str | Path) -> Path:
        self._require_model()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns, "model_name": self.model_name}, output_path)
        return output_path

    def load(self, path: str | Path) -> None:
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])

    def feature_importance(self) -> pd.DataFrame:
        self._require_model()
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
