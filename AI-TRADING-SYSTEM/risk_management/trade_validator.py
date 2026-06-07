from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.approved = False
        self.reasons.append(reason)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        parts = [f"[{status}]"]
        if self.reasons:
            parts.append("Reasons: " + "; ".join(self.reasons))
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return " | ".join(parts)


@dataclass(frozen=True)
class RiskLimits:
    """All configurable per-trade and portfolio-level limits."""
    min_ai_confidence: float = 60.0          # reject if AI confidence < this
    require_trend_alignment: bool = True      # reject if EMA trend contradicts signal
    min_sentiment: str | None = None         # "Positive" required for BUY, "Negative" for SELL
    max_portfolio_exposure_pct: float = 80.0 # reject if used_capital / total > this
    max_open_positions: int = 5
    max_single_position_pct: float = 20.0    # max % of portfolio in one symbol
    max_sector_exposure_pct: float = 40.0
    daily_loss_limit_pct: float = 3.0        # auto-disable when breached
    max_drawdown_pct: float = 15.0           # halt trading above this drawdown


class TradeValidator:
    """
    Pre-trade gate that enforces all configurable risk limits.

    Call validate() before every order. Returns a ValidationResult —
    only submit the order if result.approved is True.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        signal: str,
        symbol: str,
        price: float,
        quantity: int,
        *,
        ai_confidence: float | None = None,
        ema_20: float | None = None,
        ema_50: float | None = None,
        sentiment: str | None = None,
        portfolio_cash: float = 0.0,
        portfolio_total_value: float = 0.0,
        portfolio_used_capital: float = 0.0,
        open_position_count: int = 0,
        symbol_exposure: float = 0.0,   # current ₹ value in this symbol
        sector_exposure: float = 0.0,   # current ₹ value in same sector
        today_pnl: float = 0.0,
        starting_capital: float = 0.0,
        current_drawdown_pct: float = 0.0,
        trading_enabled: bool = True,
    ) -> ValidationResult:
        result = ValidationResult(approved=True)

        # 0. Hard gate — trading disabled
        if not trading_enabled:
            result.reject("Trading is disabled (daily loss limit breached)")
            return result

        # 1. Price and quantity sanity
        if price <= 0:
            result.reject(f"Invalid price: {price}")
        if quantity <= 0:
            result.reject(f"Invalid quantity: {quantity}")
        if result.approved is False:
            return result

        # 2. AI confidence
        if ai_confidence is not None and ai_confidence < self.limits.min_ai_confidence:
            result.reject(f"AI confidence {ai_confidence:.1f}% < minimum {self.limits.min_ai_confidence:.1f}%")

        # 3. Trend alignment
        if self.limits.require_trend_alignment and ema_20 is not None and ema_50 is not None:
            if signal == "BUY" and ema_20 < ema_50:
                result.reject("BUY signal contradicts trend (EMA20 < EMA50)")
            elif signal == "SELL" and ema_20 > ema_50:
                result.warn("SELL signal contradicts uptrend (EMA20 > EMA50)")

        # 4. Sentiment filter
        if self.limits.min_sentiment and sentiment:
            if signal == "BUY" and self.limits.min_sentiment == "Positive" and sentiment != "Positive":
                result.reject(f"BUY requires Positive sentiment; got {sentiment}")
            elif signal == "SELL" and self.limits.min_sentiment == "Negative" and sentiment != "Negative":
                result.warn(f"SELL without Negative sentiment ({sentiment})")

        # 5. Max open positions
        if signal == "BUY" and open_position_count >= self.limits.max_open_positions:
            result.reject(f"Max open positions reached ({self.limits.max_open_positions})")

        # 6. Portfolio exposure
        if portfolio_total_value > 0:
            exposure_pct = portfolio_used_capital / portfolio_total_value * 100
            if signal == "BUY" and exposure_pct >= self.limits.max_portfolio_exposure_pct:
                result.reject(f"Portfolio exposure {exposure_pct:.1f}% ≥ limit {self.limits.max_portfolio_exposure_pct:.1f}%")

        # 7. Single position size
        if portfolio_total_value > 0:
            new_exposure = (symbol_exposure + price * quantity) / portfolio_total_value * 100
            if signal == "BUY" and new_exposure > self.limits.max_single_position_pct:
                result.reject(f"Single position exposure {new_exposure:.1f}% > limit {self.limits.max_single_position_pct:.1f}%")

        # 8. Sector exposure
        if portfolio_total_value > 0 and sector_exposure > 0:
            sec_pct = sector_exposure / portfolio_total_value * 100
            if signal == "BUY" and sec_pct >= self.limits.max_sector_exposure_pct:
                result.warn(f"Sector exposure {sec_pct:.1f}% approaching limit {self.limits.max_sector_exposure_pct:.1f}%")

        # 9. Daily loss limit
        if starting_capital > 0:
            daily_loss_pct = abs(today_pnl / starting_capital * 100) if today_pnl < 0 else 0
            if daily_loss_pct >= self.limits.daily_loss_limit_pct:
                result.reject(f"Daily loss {daily_loss_pct:.2f}% ≥ limit {self.limits.daily_loss_limit_pct:.2f}%")

        # 10. Drawdown limit
        if current_drawdown_pct >= self.limits.max_drawdown_pct:
            result.reject(f"Drawdown {current_drawdown_pct:.2f}% ≥ limit {self.limits.max_drawdown_pct:.2f}%")

        # 11. Sufficient cash for BUY
        if signal == "BUY":
            cost = price * quantity
            if cost > portfolio_cash:
                result.reject(f"Insufficient cash: need ₹{cost:,.0f}, have ₹{portfolio_cash:,.0f}")

        if not result.approved:
            self.logger.warning("Trade rejected", extra={"symbol": symbol, "reasons": result.reasons})
        else:
            self.logger.info("Trade approved", extra={"symbol": symbol, "signal": signal})
        return result
