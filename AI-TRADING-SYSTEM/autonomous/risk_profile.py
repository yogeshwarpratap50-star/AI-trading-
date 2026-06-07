"""
Risk profiles — the only input the user must provide (besides capital).

Three profiles ship out of the box:
  conservative  — preserve capital, few positions, tight stops
  moderate      — balanced growth, standard risk
  aggressive    — maximum growth, wider stops, more positions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskProfile = Literal["conservative", "moderate", "aggressive"]


@dataclass(frozen=True)
class RiskProfileConfig:
    name: RiskProfile

    # Position limits
    max_positions: int
    max_single_position_pct: float    # % of portfolio per stock
    max_sector_exposure_pct: float

    # Sizing
    risk_per_trade_pct: float         # % of capital risked per trade

    # Signal quality
    min_ai_confidence: float          # 0-100
    min_sentiment_score: float        # -1 to +1

    # Stop / target
    stop_loss_atr_multiplier: float
    take_profit_atr_multiplier: float
    trailing_stop_pct: float

    # Portfolio safety
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    cash_reserve_pct: float           # always keep this % in cash

    # Rebalance
    rebalance_threshold_pct: float    # drift % before rebalancing

    # Reporting schedule
    report_daily: bool = True
    report_weekly: bool = True
    report_monthly: bool = True


RISK_PROFILES: dict[RiskProfile, RiskProfileConfig] = {
    "conservative": RiskProfileConfig(
        name="conservative",
        max_positions=3,
        max_single_position_pct=15.0,
        max_sector_exposure_pct=30.0,
        risk_per_trade_pct=0.5,
        min_ai_confidence=70.0,
        min_sentiment_score=0.1,
        stop_loss_atr_multiplier=1.5,
        take_profit_atr_multiplier=2.0,
        trailing_stop_pct=1.5,
        daily_loss_limit_pct=1.5,
        max_drawdown_pct=8.0,
        cash_reserve_pct=30.0,
        rebalance_threshold_pct=5.0,
    ),
    "moderate": RiskProfileConfig(
        name="moderate",
        max_positions=5,
        max_single_position_pct=20.0,
        max_sector_exposure_pct=40.0,
        risk_per_trade_pct=1.5,
        min_ai_confidence=62.0,
        min_sentiment_score=0.0,
        stop_loss_atr_multiplier=2.0,
        take_profit_atr_multiplier=3.0,
        trailing_stop_pct=2.0,
        daily_loss_limit_pct=3.0,
        max_drawdown_pct=15.0,
        cash_reserve_pct=15.0,
        rebalance_threshold_pct=8.0,
    ),
    "aggressive": RiskProfileConfig(
        name="aggressive",
        max_positions=8,
        max_single_position_pct=25.0,
        max_sector_exposure_pct=50.0,
        risk_per_trade_pct=3.0,
        min_ai_confidence=55.0,
        min_sentiment_score=-0.1,
        stop_loss_atr_multiplier=2.5,
        take_profit_atr_multiplier=4.0,
        trailing_stop_pct=3.0,
        daily_loss_limit_pct=5.0,
        max_drawdown_pct=25.0,
        cash_reserve_pct=5.0,
        rebalance_threshold_pct=12.0,
    ),
}


def get_profile(name: str) -> RiskProfileConfig:
    key = name.lower().strip()
    if key not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile '{name}'. Choose: {list(RISK_PROFILES)}")
    return RISK_PROFILES[key]  # type: ignore[return-value]
