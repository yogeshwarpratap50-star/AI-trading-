from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from risk_management.portfolio_risk import PortfolioRiskAnalyzer, PortfolioRiskReport
from risk_management.position_sizer import PositionSizer, SizingMethod
from risk_management.trade_validator import RiskLimits, TradeValidator, ValidationResult


@dataclass(frozen=True)
class RiskConfig:
    """Single source of truth for all risk parameters."""
    # Position sizing
    sizing_method: SizingMethod = SizingMethod.ATR_BASED
    risk_per_trade_pct: float = 1.0
    atr_multiplier: float = 2.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_stop_pct: float = 2.0
    risk_reward_ratio: float = 2.0

    # Daily controls
    daily_loss_limit_pct: float = 3.0
    max_open_positions: int = 3
    max_portfolio_exposure_pct: float = 75.0
    max_single_position_pct: float = 25.0
    max_sector_exposure_pct: float = 40.0
    max_drawdown_pct: float = 15.0

    # Signal filters
    min_ai_confidence: float = 60.0
    require_trend_alignment: bool = True


class RiskEngine:
    """
    Orchestrates all risk subsystems.

    Usage
    -----
    engine = RiskEngine(config)
    qty = engine.size_position(symbol, price, atr=atr)
    result = engine.validate_trade(signal, symbol, price, qty, **portfolio_state)
    report = engine.portfolio_risk(positions, cash)
    stop = engine.stop_loss(entry_price, atr=atr)
    target = engine.take_profit(entry_price, atr=atr)
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        self._sizer: PositionSizer | None = None
        self._validator = TradeValidator(
            RiskLimits(
                min_ai_confidence=self.config.min_ai_confidence,
                require_trend_alignment=self.config.require_trend_alignment,
                max_portfolio_exposure_pct=self.config.max_portfolio_exposure_pct,
                max_open_positions=self.config.max_open_positions,
                max_single_position_pct=self.config.max_single_position_pct,
                max_sector_exposure_pct=self.config.max_sector_exposure_pct,
                daily_loss_limit_pct=self.config.daily_loss_limit_pct,
                max_drawdown_pct=self.config.max_drawdown_pct,
            )
        )
        self._portfolio_risk = PortfolioRiskAnalyzer()

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_position(
        self,
        symbol: str,
        price: float,
        capital: float,
        atr: float | None = None,
        volatility: float | None = None,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
    ) -> int:
        sizer = PositionSizer(capital, self.config.risk_per_trade_pct)
        qty = sizer.size(
            self.config.sizing_method,
            price,
            atr=atr,
            atr_multiplier=self.config.atr_multiplier,
            volatility=volatility,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
        )
        self.logger.info("Position sized", extra={"symbol": symbol, "qty": qty, "method": self.config.sizing_method})
        return qty

    # ------------------------------------------------------------------
    # Trade validation
    # ------------------------------------------------------------------

    def validate_trade(
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
        symbol_exposure: float = 0.0,
        sector_exposure: float = 0.0,
        today_pnl: float = 0.0,
        starting_capital: float = 0.0,
        current_drawdown_pct: float = 0.0,
        trading_enabled: bool = True,
    ) -> ValidationResult:
        return self._validator.validate(
            signal, symbol, price, quantity,
            ai_confidence=ai_confidence,
            ema_20=ema_20,
            ema_50=ema_50,
            sentiment=sentiment,
            portfolio_cash=portfolio_cash,
            portfolio_total_value=portfolio_total_value,
            portfolio_used_capital=portfolio_used_capital,
            open_position_count=open_position_count,
            symbol_exposure=symbol_exposure,
            sector_exposure=sector_exposure,
            today_pnl=today_pnl,
            starting_capital=starting_capital,
            current_drawdown_pct=current_drawdown_pct,
            trading_enabled=trading_enabled,
        )

    # ------------------------------------------------------------------
    # Portfolio risk
    # ------------------------------------------------------------------

    def portfolio_risk(
        self,
        positions: list[dict[str, Any]],
        cash: float,
        starting_capital: float,
        returns_df: pd.DataFrame | None = None,
        nifty_returns: pd.Series | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> PortfolioRiskReport:
        if sector_map:
            self._portfolio_risk.sector_map = sector_map
        return self._portfolio_risk.analyze(positions, cash, starting_capital, returns_df, nifty_returns)

    # ------------------------------------------------------------------
    # Stop-loss / take-profit helpers
    # ------------------------------------------------------------------

    def stop_loss(self, entry_price: float, atr: float | None = None, side: str = "BUY") -> float:
        sizer = PositionSizer(0)
        if atr:
            return sizer.atr_stop(entry_price, atr, self.config.atr_multiplier, side)
        return sizer.pct_stop(entry_price, self.config.stop_loss_pct, side)

    def take_profit(self, entry_price: float, atr: float | None = None, side: str = "BUY") -> float:
        sizer = PositionSizer(0)
        if atr:
            return sizer.atr_target(entry_price, atr, self.config.risk_reward_ratio, self.config.atr_multiplier, side)
        return sizer.fixed_target(entry_price, self.config.take_profit_pct, side)

    def trailing_stop(self, entry_price: float, highest_price: float, side: str = "BUY") -> float:
        return PositionSizer.trailing_stop(entry_price, highest_price, self.config.trailing_stop_pct, side)
