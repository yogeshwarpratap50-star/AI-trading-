"""
Safety System — Phase 12.

Monitors trading conditions and automatically halts the engine when:
  * Daily loss exceeds the profile limit
  * AI model confidence collapses (avg < threshold over last N predictions)
  * Broker connection fails
  * Market anomaly detected (extreme volatility spike)

All checks are non-destructive — they return a SafetyStatus without
directly killing threads. The engine polls SafetySystem.is_safe() each tick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from autonomous.risk_profile import RiskProfileConfig


class EmergencyReason(str, Enum):
    DAILY_LOSS_LIMIT    = "daily_loss_limit"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    BROKER_FAILURE      = "broker_failure"
    MARKET_ANOMALY      = "market_anomaly"
    MANUAL_STOP         = "manual_stop"


@dataclass
class SafetyStatus:
    safe: bool
    reason: EmergencyReason | None = None
    detail: str = ""
    triggered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "triggered_at": self.triggered_at,
        }


class SafetySystem:
    """
    Stateful safety guard. Call check() on each tick; call reset() to re-enable.

    Usage
    -----
    safety = SafetySystem(profile)
    safety.record_pnl(-500)                  # update running P&L
    safety.record_confidence(45.0)           # update AI confidence
    status = safety.check(broker_connected=True)
    if not status.safe:
        engine.stop()
    """

    # Confidence collapse: halt if rolling avg drops below this
    CONFIDENCE_WINDOW = 10           # last N predictions
    CONFIDENCE_FLOOR  = 45.0        # %

    # Market anomaly: halt if returns spike beyond N × historical std
    ANOMALY_Z_THRESHOLD = 4.0

    def __init__(self, profile: RiskProfileConfig, starting_capital: float) -> None:
        self.profile = profile
        self.starting_capital = starting_capital
        self._halted: SafetyStatus = SafetyStatus(safe=True)
        self._daily_pnl: float = 0.0
        self._confidence_history: list[float] = []
        self._return_history: list[float] = []
        self._manual_stop: bool = False
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # State updates (called by the engine on each event)
    # ------------------------------------------------------------------

    def record_pnl(self, pnl: float) -> None:
        """Accumulate realised P&L for the current day."""
        self._daily_pnl += pnl

    def record_confidence(self, confidence: float) -> None:
        """Log AI prediction confidence (0-100)."""
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > self.CONFIDENCE_WINDOW * 3:
            self._confidence_history.pop(0)

    def record_return(self, pct_return: float) -> None:
        """Log bar-level return for anomaly detection."""
        self._return_history.append(pct_return)
        if len(self._return_history) > 50:
            self._return_history.pop(0)

    def manual_stop(self) -> None:
        self._manual_stop = True

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_pnl = 0.0

    def reset(self) -> None:
        """Re-enable trading (e.g. after manual review)."""
        self._halted = SafetyStatus(safe=True)
        self._manual_stop = False
        self.logger.info("SafetySystem reset — trading re-enabled")

    # ------------------------------------------------------------------
    # Primary check
    # ------------------------------------------------------------------

    def check(self, broker_connected: bool = True) -> SafetyStatus:
        """Return current safety status. Short-circuits on first failure."""
        if self._manual_stop:
            return self._halt(EmergencyReason.MANUAL_STOP, "Manual stop requested")

        # 1. Daily loss limit
        daily_loss_limit = self.starting_capital * self.profile.daily_loss_limit_pct / 100
        if self._daily_pnl < -daily_loss_limit:
            return self._halt(
                EmergencyReason.DAILY_LOSS_LIMIT,
                f"Daily P&L ₹{self._daily_pnl:.0f} < limit -₹{daily_loss_limit:.0f}",
            )

        # 2. Broker failure
        if not broker_connected:
            return self._halt(EmergencyReason.BROKER_FAILURE, "Broker disconnected")

        # 3. AI confidence collapse
        if len(self._confidence_history) >= self.CONFIDENCE_WINDOW:
            recent_avg = sum(self._confidence_history[-self.CONFIDENCE_WINDOW:]) / self.CONFIDENCE_WINDOW
            if recent_avg < self.CONFIDENCE_FLOOR:
                return self._halt(
                    EmergencyReason.CONFIDENCE_COLLAPSE,
                    f"Avg confidence {recent_avg:.1f}% < {self.CONFIDENCE_FLOOR}%",
                )

        # 4. Market anomaly
        if len(self._return_history) >= 20:
            import numpy as np
            arr = np.array(self._return_history)
            mean, std = arr.mean(), arr.std()
            if std > 0:
                last_z = abs((arr[-1] - mean) / std)
                if last_z > self.ANOMALY_Z_THRESHOLD:
                    return self._halt(
                        EmergencyReason.MARKET_ANOMALY,
                        f"Return z-score={last_z:.1f} (extreme volatility spike)",
                    )

        self._halted = SafetyStatus(safe=True)
        return self._halted

    @property
    def is_safe(self) -> bool:
        return self._halted.safe

    @property
    def current_status(self) -> SafetyStatus:
        return self._halted

    # ------------------------------------------------------------------

    def _halt(self, reason: EmergencyReason, detail: str) -> SafetyStatus:
        if self._halted.safe:  # only log the first trigger
            self.logger.critical(
                "SAFETY HALT: %s — %s", reason.value, detail,
                extra={"reason": reason.value},
            )
        self._halted = SafetyStatus(
            safe=False,
            reason=reason,
            detail=detail,
            triggered_at=datetime.now().isoformat(),
        )
        return self._halted
