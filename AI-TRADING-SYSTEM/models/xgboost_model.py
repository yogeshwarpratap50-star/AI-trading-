from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from models.base_model import TradingModel


class XGBoostTradingModel(TradingModel):
    """XGBoost classifier wrapper."""

    model_name = "xgboost"

    def __init__(self, n_estimators: int = 80, random_state: int = 42) -> None:
        super().__init__()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_columns = list(x_train.columns)
        self.model.fit(x_train, y_train)
