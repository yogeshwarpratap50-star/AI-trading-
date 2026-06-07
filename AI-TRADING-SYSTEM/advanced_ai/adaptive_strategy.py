"""
Adaptive Strategy Engine — Phase 11E.

Adjusts risk, confidence threshold, position size, stop loss, and take profit
based on the detected market regime.
"""
from __future__ import annotations

from dataclasses import dataclass

from advanced_ai.market_regime import Regime
from risk_management.risk_engine import RiskConfig


@dataclass
class RegimeAdjustment:
    """Multipliers/overrides applied on top of the base RiskConfig."""
    regime: Regime
    risk_pct_multiplier: float = 1.0
    min_confidence_pct: float = 65.0
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    description: str = ""


# Regime-specific defaults
_REGIME_ADJUSTMENTS: dict[Regime, RegimeAdjustment] = {
    Regime.BULL: RegimeAdjustment(
        regime=Regime.BULL,
        risk_pct_multiplier=1.2,
        min_confidence_pct=60.0,
        position_size_multiplier=1.1,
        stop_loss_multiplier=1.0,
        take_profit_multiplier=1.2,
        description="Bull market — slightly larger positions and wider targets",
    ),
    Regime.BEAR: RegimeAdjustment(
        regime=Regime.BEAR,
        risk_pct_multiplier=0.5,
        min_confidence_pct=75.0,
        position_size_multiplier=0.5,
        stop_loss_multiplier=0.8,
        take_profit_multiplier=0.8,
        description="Bear market — tight risk, higher confidence required",
    ),
    Regime.SIDEWAYS: RegimeAdjustment(
        regime=Regime.SIDEWAYS,
        risk_pct_multiplier=0.8,
        min_confidence_pct=70.0,
        position_size_multiplier=0.8,
        stop_loss_multiplier=0.9,
        take_profit_multiplier=0.9,
        description="Sideways market — smaller positions, mean-reversion approach",
    ),
    Regime.HIGH_VOLATILITY: RegimeAdjustment(
        regime=Regime.HIGH_VOLATILITY,
        risk_pct_multiplier=0.4,
        min_confidence_pct=80.0,
        position_size_multiplier=0.4,
        stop_loss_multiplier=1.5,
        take_profit_multiplier=1.5,
        description="High volatility — very small positions, wide stops",
    ),
    Regime.LOW_VOLATILITY: RegimeAdjustment(
        regime=Regime.LOW_VOLATILITY,
        risk_pct_multiplier=1.0,
        min_confidence_pct=65.0,
        position_size_multiplier=1.0,
        stop_loss_multiplier=0.8,
        take_profit_multiplier=0.8,
        description="Low volatility — normal risk, tighter stops",
    ),
}


class AdaptiveStrategy:
    """
    Wraps a base RiskConfig and applies regime-specific multipliers.

    Usage
    -----
    base_cfg = RiskConfig(risk_per_trade_pct=1.0)
    adaptive = AdaptiveStrategy(base_cfg)
    adjusted_cfg = adaptive.adjust(regime_result)
    """

    def __init__(self, base_config: RiskConfig | None = None) -> None:
        self.base = base_config or RiskConfig()

    def adjust(self, regime: Regime) -> RiskConfig:
        """Return a new RiskConfig adapted to the given regime."""
        adj = _REGIME_ADJUSTMENTS.get(regime, _REGIME_ADJUSTMENTS[Regime.SIDEWAYS])
        return RiskConfig(
            sizing_method=self.base.sizing_method,
            risk_per_trade_pct=round(self.base.risk_per_trade_pct * adj.risk_pct_multiplier, 3),
            atr_multiplier=round(self.base.atr_multiplier * adj.stop_loss_multiplier, 2),
            stop_loss_pct=round(self.base.stop_loss_pct * adj.stop_loss_multiplier, 2),
            take_profit_pct=round(self.base.take_profit_pct * adj.take_profit_multiplier, 2),
            trailing_stop_pct=self.base.trailing_stop_pct,
            risk_reward_ratio=self.base.risk_reward_ratio,
            daily_loss_limit_pct=self.base.daily_loss_limit_pct,
            max_open_positions=self.base.max_open_positions,
            max_portfolio_exposure_pct=self.base.max_portfolio_exposure_pct,
            max_single_position_pct=round(
                self.base.max_single_position_pct * adj.position_size_multiplier, 2
            ),
            max_sector_exposure_pct=self.base.max_sector_exposure_pct,
            max_drawdown_pct=self.base.max_drawdown_pct,
            min_ai_confidence=adj.min_confidence_pct,
            require_trend_alignment=self.base.require_trend_alignment,
        )

    def get_adjustment(self, regime: Regime) -> RegimeAdjustment:
        return _REGIME_ADJUSTMENTS.get(regime, _REGIME_ADJUSTMENTS[Regime.SIDEWAYS])

    @staticmethod
    def all_adjustments() -> dict[str, dict]:
        return {r.value: adj.__dict__ for r, adj in _REGIME_ADJUSTMENTS.items()}
