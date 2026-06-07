from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from training.train_pipeline import TrainingPipeline


@dataclass(frozen=True)
class RetrainingDecision:
    should_run: bool
    reason: str


class AutomaticRetrainingService:
    """Supports weekly and monthly retraining schedules."""

    def __init__(self, state_path: Path = Path("models/retraining_state.csv")) -> None:
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def should_retrain(self, frequency: str, now: datetime | None = None) -> RetrainingDecision:
        now = now or datetime.utcnow()
        last_run = self._last_run()
        if last_run is None:
            return RetrainingDecision(True, "No previous retraining run found.")
        interval = timedelta(days=7 if frequency == "weekly" else 30 if frequency == "monthly" else 0)
        if interval.days == 0:
            raise ValueError("frequency must be weekly or monthly.")
        return RetrainingDecision(now - last_run >= interval, f"Last run was {last_run.isoformat()}.")

    def run_if_due(self, ohlcv: pd.DataFrame, frequency: str = "weekly") -> dict[str, object] | None:
        decision = self.should_retrain(frequency)
        if not decision.should_run:
            return None
        result = TrainingPipeline().run(ohlcv)
        self._record_run(frequency)
        return result

    def _last_run(self) -> datetime | None:
        if not self.state_path.exists():
            return None
        rows = pd.read_csv(self.state_path)
        if rows.empty:
            return None
        return datetime.fromisoformat(str(rows.iloc[-1]["ran_at"]))

    def _record_run(self, frequency: str) -> None:
        row = pd.DataFrame([{"frequency": frequency, "ran_at": datetime.utcnow().isoformat()}])
        if self.state_path.exists():
            row.to_csv(self.state_path, mode="a", header=False, index=False)
        else:
            row.to_csv(self.state_path, index=False)
