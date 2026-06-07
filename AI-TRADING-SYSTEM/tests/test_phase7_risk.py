"""Tests for Phase 7 — Risk Management Engine."""
from __future__ import annotations

import pytest

from risk_management.position_sizer import PositionSizer, SizingMethod
from risk_management.trade_validator import RiskLimits, TradeValidator
from risk_management.portfolio_risk import PortfolioRiskAnalyzer
from risk_management.risk_engine import RiskConfig, RiskEngine


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_fixed_pct_sizing(self) -> None:
        sizer = PositionSizer(capital=100_000.0, risk_per_trade_pct=2.0)
        qty = sizer.size(SizingMethod.FIXED_PCT, price=500.0)
        assert qty == 4   # 100k * 0.02 / 500 = 4

    def test_atr_sizing_returns_positive(self) -> None:
        sizer = PositionSizer(capital=100_000.0, risk_per_trade_pct=1.0)
        qty = sizer.size(SizingMethod.ATR_BASED, price=2500.0, atr=50.0, atr_multiplier=2.0)
        assert qty > 0

    def test_volatility_sizing_returns_positive(self) -> None:
        sizer = PositionSizer(capital=100_000.0, risk_per_trade_pct=1.0)
        qty = sizer.size(SizingMethod.VOLATILITY, price=1000.0, volatility=0.02)
        assert qty > 0

    def test_kelly_sizing_returns_positive(self) -> None:
        sizer = PositionSizer(capital=100_000.0, risk_per_trade_pct=1.0)
        qty = sizer.size(SizingMethod.KELLY, price=500.0, win_rate=0.6, avg_win_loss_ratio=1.5)
        assert qty > 0

    def test_atr_stop_buy(self) -> None:
        stop = PositionSizer.atr_stop(entry_price=100.0, atr=5.0, multiplier=2.0, side="BUY")
        assert stop == pytest.approx(90.0)

    def test_atr_stop_sell(self) -> None:
        stop = PositionSizer.atr_stop(entry_price=100.0, atr=5.0, multiplier=2.0, side="SELL")
        assert stop == pytest.approx(110.0)

    def test_trailing_stop(self) -> None:
        stop = PositionSizer.trailing_stop(entry_price=100.0, highest_price=120.0, trail_pct=5.0, side="BUY")
        assert stop == pytest.approx(114.0)

    def test_zero_quantity_when_price_zero(self) -> None:
        sizer = PositionSizer(capital=100_000.0, risk_per_trade_pct=2.0)
        qty = sizer.size(SizingMethod.FIXED_PCT, price=0.0)
        assert qty == 0


# ---------------------------------------------------------------------------
# TradeValidator
# ---------------------------------------------------------------------------

class TestTradeValidator:
    @pytest.fixture
    def validator(self) -> TradeValidator:
        return TradeValidator(RiskLimits())

    def test_basic_buy_approved(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 2500.0, 5,    # 5 * 2500 = 12.5k < 20% of 200k
            portfolio_cash=200_000.0,
            portfolio_total_value=200_000.0,
        )
        assert result.approved

    def test_trading_disabled_rejected(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 2500.0, 10,
            trading_enabled=False,
            portfolio_cash=100_000.0,
            portfolio_total_value=100_000.0,
        )
        assert not result.approved
        assert any("trading" in r.lower() or "disabled" in r.lower() for r in result.reasons)

    def test_zero_price_rejected(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 0.0, 10,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
        )
        assert not result.approved

    def test_zero_quantity_rejected(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 2500.0, 0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
        )
        assert not result.approved

    def test_insufficient_cash_rejected(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 2500.0, 1000,
            portfolio_cash=500.0, portfolio_total_value=500.0,
        )
        assert not result.approved

    def test_max_positions_rejected(self, validator: TradeValidator) -> None:
        limits = RiskLimits(max_open_positions=2)
        v = TradeValidator(limits)
        result = v.validate(
            "BUY", "RELIANCE", 100.0, 1,
            open_position_count=2,
            portfolio_cash=100_000.0,
            portfolio_total_value=100_000.0,
        )
        assert not result.approved

    def test_validation_result_summary(self, validator: TradeValidator) -> None:
        result = validator.validate(
            "BUY", "RELIANCE", 2500.0, 10,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
        )
        summary = result.summary()
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# PortfolioRiskAnalyzer
# ---------------------------------------------------------------------------

class TestPortfolioRiskAnalyzer:
    @pytest.fixture
    def analyzer(self) -> PortfolioRiskAnalyzer:
        return PortfolioRiskAnalyzer()

    def _positions(self) -> list[dict]:
        return [
            {"symbol": "RELIANCE", "quantity": 10, "avg_buy_price": 2500.0, "last_price": 2500.0,
             "market_value": 25000.0, "unrealized_pnl": 0.0},
            {"symbol": "TCS", "quantity": 5, "avg_buy_price": 3500.0, "last_price": 3500.0,
             "market_value": 17500.0, "unrealized_pnl": 0.0},
        ]

    def test_report_has_risk_score(self, analyzer: PortfolioRiskAnalyzer) -> None:
        report = analyzer.analyze(self._positions(), cash=57500.0, starting_capital=100_000.0)
        assert 0 <= report.risk_score <= 100

    def test_risk_level_is_valid(self, analyzer: PortfolioRiskAnalyzer) -> None:
        report = analyzer.analyze(self._positions(), cash=57500.0, starting_capital=100_000.0)
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_empty_positions_low_risk(self, analyzer: PortfolioRiskAnalyzer) -> None:
        report = analyzer.analyze([], cash=100_000.0, starting_capital=100_000.0)
        assert report.risk_level == "LOW"

    def test_exposure_pct_correct(self, analyzer: PortfolioRiskAnalyzer) -> None:
        positions = [{"symbol": "RELIANCE", "quantity": 10, "avg_buy_price": 2500.0,
                      "last_price": 2500.0, "market_value": 25000.0, "unrealized_pnl": 0.0}]
        report = analyzer.analyze(positions, cash=75_000.0, starting_capital=100_000.0)
        assert report.portfolio_exposure_pct == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------------------
# RiskEngine (integration)
# ---------------------------------------------------------------------------

class TestRiskEngine:
    @pytest.fixture
    def engine(self) -> RiskEngine:
        return RiskEngine(RiskConfig())

    def test_size_position_returns_int(self, engine: RiskEngine) -> None:
        qty = engine.size_position("RELIANCE", price=2500.0, capital=100_000.0)
        assert isinstance(qty, int)

    def test_validate_trade_approved(self, engine: RiskEngine) -> None:
        result = engine.validate_trade(
            "BUY", "RELIANCE", 2500.0, 5,
            portfolio_cash=100_000.0,
            portfolio_total_value=100_000.0,
        )
        assert result.approved

    def test_stop_loss_below_entry_for_buy(self, engine: RiskEngine) -> None:
        stop = engine.stop_loss(2500.0, atr=50.0, side="BUY")
        assert stop < 2500.0

    def test_take_profit_above_entry_for_buy(self, engine: RiskEngine) -> None:
        target = engine.take_profit(2500.0, atr=50.0, side="BUY")
        assert target > 2500.0

    def test_trailing_stop_below_high(self, engine: RiskEngine) -> None:
        stop = engine.trailing_stop(2400.0, highest_price=2600.0, side="BUY")
        assert stop < 2600.0

    def test_portfolio_risk_returns_report(self, engine: RiskEngine) -> None:
        positions = [{"symbol": "TCS", "quantity": 5, "avg_buy_price": 3500.0,
                      "last_price": 3500.0, "market_value": 17500.0, "unrealized_pnl": 0.0}]
        report = engine.portfolio_risk(positions, cash=82_500.0, starting_capital=100_000.0)
        assert report.risk_score >= 0
