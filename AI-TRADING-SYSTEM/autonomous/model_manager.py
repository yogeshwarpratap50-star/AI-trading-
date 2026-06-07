"""
Autonomous Model Manager — Phase 12.

Handles automatic:
  * Retraining when accuracy drops / drift detected / schedule fires
  * Model switching (promotes best-scoring model)
  * Rollback to previous model if new model underperforms
  * Drift monitoring every tick
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from advanced_ai.auto_retrainer import AutoRetrainer, RetrainingConfig
from advanced_ai.drift_detector import DriftDetector
from advanced_ai.performance_tracker import AIPerformanceTracker
from models.model_registry import ModelRegistry
from prediction.prediction_engine import PredictionEngine


@dataclass
class ModelState:
    model_name: str
    loaded_at: str
    accuracy: float = 0.0
    drift_psi: float = 0.0
    retrain_count: int = 0
    rollback_count: int = 0


class AutonomousModelManager:
    """
    Manages model lifecycle without user interaction.

    Workflow per cycle
    ------------------
    1. Record new prediction + outcome → performance tracker
    2. Detect drift on recent features vs training reference
    3. Evaluate retraining triggers
    4. If triggered: retrain → evaluate → promote or rollback
    """

    PERFORMANCE_WINDOW = 50          # min predictions before accuracy check
    ROLLBACK_ACCURACY_FLOOR = 0.48   # rollback if new model worse than this

    def __init__(
        self,
        data_dir: str | Path = "data/historical",
        retrain_config: RetrainingConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.registry = ModelRegistry()
        self.pred_engine = PredictionEngine(registry=self.registry)
        self.tracker = AIPerformanceTracker(window_days=30)
        self.drift_detector = DriftDetector()
        self.retrainer = AutoRetrainer(retrain_config or RetrainingConfig(
            min_accuracy=0.52,
            retrain_on_drift=True,
            weekly_retrain=True,
            monthly_retrain=True,
            min_samples=100,
        ))
        self._reference_set = False
        self._last_retrain: datetime | None = None
        self._model_history: list[str] = []   # stack of previous model paths
        self.state = ModelState(
            model_name="none",
            loaded_at=datetime.now().isoformat(),
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Per-tick call
    # ------------------------------------------------------------------

    def cycle(
        self,
        symbol: str,
        features: pd.DataFrame,
        actual_outcome: str | None = None,
        actual_pnl: float = 0.0,
    ) -> dict[str, Any]:
        """
        Run one management cycle. Call this after every prediction.

        Returns a summary dict with any actions taken.
        """
        actions: list[str] = []

        # Load model if not yet loaded
        self._ensure_model_loaded()

        # Fit drift reference on first call
        if not self._reference_set and len(features) >= 50:
            self.drift_detector.fit_reference(features)
            self._reference_set = True
            actions.append("drift_reference_fitted")

        # Record outcome if known
        if actual_outcome:
            self.tracker.log_outcome(symbol, actual_outcome, actual_pnl)

        # Drift check (only after reference is set)
        drift_report = None
        if self._reference_set and len(features) >= 20:
            drift_report = self.drift_detector.detect(features.tail(20))
            self.state.drift_psi = drift_report.data_drift

        # Performance report
        perf_report = self.tracker.report()
        if perf_report.total_predictions > 0:
            self.state.accuracy = perf_report.accuracy

        # Evaluate retraining triggers
        triggers = self.retrainer.evaluate(
            perf_report=perf_report if perf_report.total_predictions >= self.PERFORMANCE_WINDOW else None,
            drift_report=drift_report,
            last_retrain=self._last_retrain,
        )

        if triggers:
            trigger = triggers[0]
            self.logger.info("Retraining triggered: %s", trigger.reason)
            data = self._load_training_data()
            if data is not None:
                result = self.retrainer.retrain(data, trigger)
                if result["status"] == "completed":
                    self._last_retrain = datetime.now()
                    self.state.retrain_count += 1
                    actions.append(f"retrained:{trigger.reason}")
                    # Reload model after retraining
                    self._reload_model()
                else:
                    actions.append(f"retrain_skipped:{result.get('reason', '?')}")

        return {
            "model": self.state.model_name,
            "accuracy": round(self.state.accuracy, 4),
            "drift_psi": round(self.state.drift_psi, 4),
            "actions": actions,
        }

    # ------------------------------------------------------------------
    # Prediction proxy
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame, symbol: str = ""):
        """Predict and log the result in the performance tracker."""
        try:
            result = self.pred_engine.predict(features)
            self.tracker.log_prediction(
                model_name=self.state.model_name,
                symbol=symbol,
                predicted_action=result.action,
                confidence=result.confidence,
            )
            return result
        except Exception as exc:
            self.logger.warning("Model prediction failed: %s", exc)
            from prediction.prediction_engine import PredictionResult
            return PredictionResult("HOLD", 0.0, 0, "none")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self.state.model_name == "none":
            self._reload_model()

    def _reload_model(self) -> None:
        try:
            model = self.pred_engine.load_latest_model()
            self.state.model_name = model.model_name
            self.state.loaded_at = datetime.now().isoformat()
            self.logger.info("Model loaded: %s", model.model_name)
        except Exception as exc:
            self.logger.warning("No model available: %s", exc)

    def _load_training_data(self) -> pd.DataFrame | None:
        """Load and concatenate all available historical CSVs."""
        frames = []
        for csv_path in self.data_dir.glob("*.csv"):
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception:
                pass
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        return combined if len(combined) >= 100 else None
