from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from models.model_registry import ModelRegistry
from models.random_forest_model import RandomForestTradingModel
from models.xgboost_model import XGBoostTradingModel
from training.dataset_builder import DatasetBuilder, MODEL_FEATURES
from training.model_comparison import ModelComparison


class TrainingPipeline:
    """End-to-end model training workflow."""

    def __init__(
        self,
        model_dir: Path = Path("models/trained"),
        reports_dir: Path = Path("reports"),
        registry: ModelRegistry | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.registry = registry or ModelRegistry()
        self.dataset_builder = DatasetBuilder()
        self.comparison = ModelComparison(reports_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        ohlcv: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        market_context: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        dataset = self.dataset_builder.build(ohlcv, sentiment, market_context)
        if len(dataset) < 20:
            raise ValueError("At least 20 labeled rows are required for model training.")

        x = dataset[MODEL_FEATURES]
        y = dataset["target_next_day_up"].astype(int)
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)

        models = {
            RandomForestTradingModel.model_name: RandomForestTradingModel(n_estimators=60),
            XGBoostTradingModel.model_name: XGBoostTradingModel(n_estimators=50),
        }
        metrics: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            model.train(x_train, y_train)
            metrics[name] = model.evaluate(x_test, y_test, include_roc_auc=name == XGBoostTradingModel.model_name)

        best_name, comparison_frame = self.comparison.compare(metrics)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        registered_entries = {}
        for name, model in models.items():
            model_path = self.model_dir / f"{name}_{version}.joblib"
            model.save(model_path)
            registered_entries[name] = self.registry.register(
                name,
                version,
                model_path,
                metrics[name],
                MODEL_FEATURES,
                is_best=name == best_name,
            )
            self._write_feature_importance(name, model.feature_importance())

        self._write_training_report(dataset, comparison_frame, best_name)
        self.logger.info("Training pipeline complete", extra={"best_model": best_name})
        return {
            "best_model": best_name,
            "metrics": metrics,
            "registry_entries": registered_entries,
            "dataset_rows": len(dataset),
        }

    def _write_feature_importance(self, model_name: str, frame: pd.DataFrame) -> None:
        path_prefix = self.reports_dir / f"{model_name}_feature_importance"
        frame.head(20).to_csv(path_prefix.with_suffix(".csv"), index=False)
        frame.head(20).to_html(path_prefix.with_suffix(".html"), index=False)

    def _write_training_report(self, dataset: pd.DataFrame, comparison: pd.DataFrame, best_model: str) -> None:
        report = pd.DataFrame(
            [
                {"metric": "dataset_rows", "value": len(dataset)},
                {"metric": "best_model", "value": best_model},
            ]
        )
        report.to_csv(self.reports_dir / "training_report.csv", index=False)
        html = "<h1>Training Report</h1>" + report.to_html(index=False) + "<h2>Model Performance</h2>" + comparison.to_html(index=False)
        (self.reports_dir / "training_report.html").write_text(html, encoding="utf-8")


def run_from_csv(path: str | Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    return TrainingPipeline().run(frame)
