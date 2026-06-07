"""AI Performance Tracker — Phase 11F. Tracks prediction accuracy, win rate, and profitability."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionLog:
    timestamp: str
    model_name: str
    symbol: str
    predicted_action: str
    actual_action: str      # BUY / SELL / HOLD — filled when outcome known
    confidence: float
    pnl: float = 0.0
    correct: bool = False


@dataclass
class PerformanceReport:
    total_predictions: int
    correct_predictions: int
    accuracy: float
    win_rate: float
    avg_pnl: float
    total_pnl: float
    by_model: dict[str, dict[str, Any]]
    period_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 4),
            "total_pnl": round(self.total_pnl, 4),
            "period_days": self.period_days,
            "by_model": self.by_model,
        }


class AIPerformanceTracker:
    """
    Logs predictions and actual outcomes; computes accuracy and profitability.

    Usage
    -----
    tracker = AIPerformanceTracker()
    tracker.log_prediction(model, symbol, predicted, confidence)
    tracker.log_outcome(symbol, actual_action, pnl)
    report = tracker.report()
    """

    def __init__(self, window_days: int = 30) -> None:
        self.window_days = window_days
        self._logs: list[PredictionLog] = []

    def log_prediction(
        self,
        model_name: str,
        symbol: str,
        predicted_action: str,
        confidence: float,
    ) -> PredictionLog:
        log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            symbol=symbol,
            predicted_action=predicted_action,
            actual_action="",
            confidence=confidence,
        )
        self._logs.append(log)
        return log

    def log_outcome(self, symbol: str, actual_action: str, pnl: float = 0.0) -> None:
        """Match the most recent unresolved prediction for this symbol."""
        for log in reversed(self._logs):
            if log.symbol == symbol and not log.actual_action:
                log.actual_action = actual_action
                log.pnl = pnl
                log.correct = log.predicted_action == actual_action
                return

    def report(self) -> PerformanceReport:
        resolved = [l for l in self._logs if l.actual_action]
        if not resolved:
            return PerformanceReport(0, 0, 0.0, 0.0, 0.0, 0.0, {}, self.window_days)

        correct = sum(1 for l in resolved if l.correct)
        pnls = [l.pnl for l in resolved]
        wins = [p for p in pnls if p > 0]

        by_model: dict[str, dict] = {}
        for log in resolved:
            m = log.model_name
            if m not in by_model:
                by_model[m] = {"total": 0, "correct": 0, "pnl": 0.0}
            by_model[m]["total"] += 1
            by_model[m]["correct"] += int(log.correct)
            by_model[m]["pnl"] += log.pnl
        for m in by_model:
            t = by_model[m]["total"]
            by_model[m]["accuracy"] = round(by_model[m]["correct"] / t, 4) if t else 0.0

        return PerformanceReport(
            total_predictions=len(resolved),
            correct_predictions=correct,
            accuracy=correct / len(resolved),
            win_rate=len(wins) / len(resolved) if resolved else 0.0,
            avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
            total_pnl=float(np.sum(pnls)),
            by_model=by_model,
            period_days=self.window_days,
        )

    def as_dataframe(self) -> pd.DataFrame:
        if not self._logs:
            return pd.DataFrame()
        return pd.DataFrame([vars(l) for l in self._logs])
