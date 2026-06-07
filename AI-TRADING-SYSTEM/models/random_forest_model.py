from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from models.base_model import TradingModel


class RandomForestTradingModel(TradingModel):
    """Random Forest classifier wrapper."""

    model_name = "random_forest"

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_columns = list(x_train.columns)
        self.model.fit(x_train, y_train)
