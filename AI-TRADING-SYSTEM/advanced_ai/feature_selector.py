"""Feature selection — Phase 11B. Importance ranking + automatic top-N retention."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


class FeatureSelector:
    """
    Ranks features by importance and retains the top-N.

    Works with any model that exposes `feature_importances_` (sklearn-compatible)
    or with a pre-computed importance DataFrame.

    Usage
    -----
    selector = FeatureSelector(top_n=20)
    selector.fit(model, feature_names)
    selected_df = selector.transform(df)
    report = selector.importance_report()
    """

    def __init__(self, top_n: int = 20, min_importance: float = 0.001) -> None:
        self.top_n = top_n
        self.min_importance = min_importance
        self._importances: pd.DataFrame = pd.DataFrame()
        self._selected: list[str] = []
        self.logger = logging.getLogger(__name__)

    def fit(self, model: Any, feature_names: list[str]) -> "FeatureSelector":
        """Extract importances from an sklearn-compatible model."""
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model must expose `feature_importances_`")
        importances = model.feature_importances_
        self._importances = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        self._select()
        return self

    def fit_from_df(self, importance_df: pd.DataFrame) -> "FeatureSelector":
        """Accept a pre-computed DataFrame with columns ['feature', 'importance']."""
        self._importances = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        self._select()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only selected columns that exist in df."""
        cols = [c for c in self._selected if c in df.columns]
        return df[cols]

    def importance_report(self) -> pd.DataFrame:
        return self._importances.copy()

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected)

    def _select(self) -> None:
        filtered = self._importances[self._importances["importance"] >= self.min_importance]
        self._selected = filtered["feature"].head(self.top_n).tolist()
        self.logger.info("FeatureSelector: %d features selected", len(self._selected))
