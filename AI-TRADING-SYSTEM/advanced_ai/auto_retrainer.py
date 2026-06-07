"""
Auto Retrainer — Phase 11H.

Triggers model retraining when:
  * Accuracy drops below threshold
  * Drift detected (DriftDetector alert)
  * Weekly or monthly schedule
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from advanced_ai.drift_detector import DriftDetector, DriftReport
from advanced_ai.performance_tracker import AIPerformanceTracker, PerformanceReport


@dataclass
class RetrainingTrigger:
    reason: str
    triggered_at: str
    metric: str
    value: float
    threshold: float


@dataclass
class RetrainingConfig:
    min_accuracy: float = 0.55           # below → retrain
    retrain_on_drift: bool = True
    weekly_retrain: bool = True
    monthly_retrain: bool = True
    min_samples: int = 200               # minimum rows to retrain


class AutoRetrainer:
    """
    Evaluates retraining triggers and runs the training pipeline when needed.

    Usage
    -----
    retrainer = AutoRetrainer(config)
    retrainer.evaluate(perf_report, drift_report, last_retrain_date, data_df)
    """

    def __init__(self, config: RetrainingConfig | None = None) -> None:
        self.config = config or RetrainingConfig()
        self._retrain_history: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        perf_report: PerformanceReport | None = None,
        drift_report: DriftReport | None = None,
        last_retrain: datetime | None = None,
        data: pd.DataFrame | None = None,
    ) -> list[RetrainingTrigger]:
        """Return a list of active triggers (empty → no retraining needed)."""
        triggers: list[RetrainingTrigger] = []
        now = datetime.now()

        # Accuracy trigger
        if perf_report and perf_report.total_predictions >= 20:
            if perf_report.accuracy < self.config.min_accuracy:
                triggers.append(RetrainingTrigger(
                    reason="low_accuracy",
                    triggered_at=now.isoformat(),
                    metric="accuracy",
                    value=perf_report.accuracy,
                    threshold=self.config.min_accuracy,
                ))

        # Drift trigger
        if self.config.retrain_on_drift and drift_report and drift_report.has_critical_drift:
            triggers.append(RetrainingTrigger(
                reason="drift_detected",
                triggered_at=now.isoformat(),
                metric="data_drift_psi",
                value=drift_report.data_drift,
                threshold=0.25,
            ))

        # Weekly schedule
        if self.config.weekly_retrain and last_retrain:
            if (now - last_retrain) > timedelta(days=7):
                triggers.append(RetrainingTrigger(
                    reason="weekly_schedule",
                    triggered_at=now.isoformat(),
                    metric="days_since_retrain",
                    value=(now - last_retrain).days,
                    threshold=7,
                ))

        # Monthly schedule (independent of weekly)
        if self.config.monthly_retrain and last_retrain:
            if (now - last_retrain) > timedelta(days=30):
                triggers.append(RetrainingTrigger(
                    reason="monthly_schedule",
                    triggered_at=now.isoformat(),
                    metric="days_since_retrain",
                    value=(now - last_retrain).days,
                    threshold=30,
                ))

        return triggers

    def retrain(self, data: pd.DataFrame, trigger: RetrainingTrigger) -> dict[str, Any]:
        """
        Execute the training pipeline on `data`.

        Returns a dict summarising the retrain event.
        """
        if len(data) < self.config.min_samples:
            self.logger.warning("Insufficient data for retraining: %d rows", len(data))
            return {"status": "skipped", "reason": "insufficient_data", "rows": len(data)}

        from training.train_pipeline import TrainingPipeline
        self.logger.info("Auto-retraining triggered: %s", trigger.reason)
        try:
            result = TrainingPipeline().run(data)
            event = {
                "status": "completed",
                "trigger": trigger.reason,
                "triggered_at": trigger.triggered_at,
                "retrained_at": datetime.now().isoformat(),
                "result": result,
            }
        except Exception as exc:
            self.logger.error("Retraining failed: %s", exc, exc_info=True)
            event = {
                "status": "failed",
                "trigger": trigger.reason,
                "error": str(exc),
                "retrained_at": datetime.now().isoformat(),
            }
        self._retrain_history.append(event)
        return event

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._retrain_history)
