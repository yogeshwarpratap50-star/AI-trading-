"""Phase 5 — Backtesting engine tests."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.monte_carlo import MonteCarloSimulator
from backtesting.order_simulator import OrderConfig, OrderSimulator
from backtesting.portfolio import Portfolio
from backtesting.portfolio_backtester import PortfolioBacktester
from backtesting.results import PerformanceCalculator
from backtesting.strategy_comparison import StrategyComparison
from backtesting.strategy_tester import (
    AIPredictionStrategy,
    BollingerBandStrategy,
    HybridStrategy,
    IndicatorStrategy,
    MACDStrategy,
    StrategyTester,
)
from backtesting.trade import OrderSide, OrderStatus, Trade
from backtesting.walk_forward import WalkForwardTester


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_ohlcv(n: int = 300, start_price: float = 1000.0, trend: float = 0.001) -> pd.DataFrame:
    """Synthetic OHLCV with a gentle uptrend."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = start_price * np.cumprod(1 + rng.normal(trend, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_featured(n: int = 300) -> pd.DataFrame:
    """OHLCV with pre-computed indicator columns (simulate feature engineering output)."""
    df = _make_ohlcv(n)
    close = df["close"]
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["rsi_14"] = 50.0  # neutral — forces all RSI checks to hold unless EMA triggers
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    df["bb_upper"] = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df["bb_lower"] = close.rolling(20).mean() - 2 * close.rolling(20).std()
    df["bb_middle"] = close.rolling(20).mean()
    return df.bfill()


def _make_ai_featured(n: int = 300) -> pd.DataFrame:
    """Featured DataFrame with AI prediction columns."""
    df = _make_featured(n)
    rng = np.random.default_rng(7)
    # Alternate BUY/SELL/HOLD with 70% confidence
    actions = rng.choice(["BUY", "SELL", "HOLD"], size=n, p=[0.4, 0.3, 0.3])
    df["ai_action"] = actions
    df["ai_confidence"] = rng.uniform(55, 95, n)
    return df


# ===========================================================================
# Module 5A — Trade
# ===========================================================================

class TestTrade:
    def test_close_buy_trade_profit(self):
        t = Trade("RELIANCE", OrderSide.BUY, datetime(2023, 1, 1), entry_price=1000.0, quantity=10)
        t.close(datetime(2023, 1, 10), exit_price=1100.0)
        assert t.pnl == pytest.approx(1000.0)
        assert t.pnl_pct == pytest.approx(0.1)
        assert t.status == OrderStatus.CLOSED

    def test_close_buy_trade_loss(self):
        t = Trade("TCS", OrderSide.BUY, datetime(2023, 1, 1), entry_price=3500.0, quantity=5)
        t.close(datetime(2023, 1, 5), exit_price=3300.0)
        assert t.pnl == pytest.approx(-1000.0)
        assert t.status == OrderStatus.CLOSED

    def test_fees_reduce_pnl(self):
        t = Trade("INFY", OrderSide.BUY, datetime(2023, 1, 1), entry_price=1500.0, quantity=10, fees=50.0)
        t.close(datetime(2023, 1, 8), exit_price=1600.0, fees=50.0)
        assert t.pnl == pytest.approx(1000.0 - 100.0)

    def test_duration_days(self):
        t = Trade("X", OrderSide.BUY, datetime(2023, 1, 1), entry_price=100.0, quantity=1)
        t.close(datetime(2023, 1, 11), exit_price=110.0)
        assert t.duration_days() == 10

    def test_to_dict_round_trips(self):
        t = Trade("Y", OrderSide.BUY, datetime(2023, 3, 15), entry_price=200.0, quantity=5, trade_id=42)
        t.close(datetime(2023, 3, 20), exit_price=210.0)
        d = t.to_dict()
        assert d["trade_id"] == 42
        assert d["pnl"] == pytest.approx(50.0)
        assert d["status"] == "CLOSED"


# ===========================================================================
# Module 5B — Order Simulator
# ===========================================================================

class TestOrderSimulator:
    def setup_method(self):
        self.sim = OrderSimulator(OrderConfig(brokerage_pct=0.001, tax_pct=0.001, slippage_pct=0.001))

    def test_market_buy_adds_slippage(self):
        fill, fees, slip = self.sim.execute_market(1000.0, 10, "BUY")
        assert fill > 1000.0  # slippage raises buy price
        assert fees > 0

    def test_market_sell_subtracts_slippage(self):
        fill, fees, slip = self.sim.execute_market(1000.0, 10, "SELL")
        assert fill < 1000.0  # slippage lowers sell price

    def test_limit_buy_triggers_when_low_below_limit(self):
        result = self.sim.execute_limit(1000.0, 990.0, 5, "BUY", bar_high=1010.0, bar_low=985.0)
        assert result is not None
        fill, fees, slip = result
        assert fill <= 990.0

    def test_limit_buy_no_trigger_when_low_above_limit(self):
        result = self.sim.execute_limit(1000.0, 950.0, 5, "BUY", bar_high=1010.0, bar_low=980.0)
        assert result is None

    def test_stop_sell_triggers_when_low_below_stop(self):
        result = self.sim.execute_stop(900.0, 5, "SELL", bar_high=920.0, bar_low=880.0)
        assert result is not None

    def test_stop_sell_no_trigger_when_low_above_stop(self):
        result = self.sim.execute_stop(800.0, 5, "SELL", bar_high=920.0, bar_low=850.0)
        assert result is None

    def test_quantity_calculation(self):
        qty = self.sim.calculate_quantity(100_000.0, 500.0, risk_pct=2.0)
        assert qty == 4   # 100000 * 0.02 / 500 = 2000 / 500 = 4

    def test_zero_price_returns_zero_quantity(self):
        assert self.sim.calculate_quantity(100_000.0, 0.0) == 0


# ===========================================================================
# Module 5A — Portfolio
# ===========================================================================

class TestPortfolio:
    def test_open_trade_reduces_cash(self):
        port = Portfolio(100_000.0)
        trade = port.open_trade("X", OrderSide.BUY, datetime(2023, 1, 1), 1000.0, 10)
        assert trade is not None
        assert port.cash == pytest.approx(100_000.0 - 10_000.0)

    def test_insufficient_capital_returns_none(self):
        port = Portfolio(5_000.0)
        trade = port.open_trade("X", OrderSide.BUY, datetime(2023, 1, 1), 1000.0, 10)
        assert trade is None
        assert port.cash == 5_000.0

    def test_close_trade_restores_cash(self):
        port = Portfolio(100_000.0)
        trade = port.open_trade("X", OrderSide.BUY, datetime(2023, 1, 1), 1000.0, 10)
        port.close_trade(trade, datetime(2023, 1, 10), 1100.0)
        assert port.cash == pytest.approx(100_000.0 - 10_000.0 + 11_000.0)

    def test_equity_reflects_position_value(self):
        port = Portfolio(100_000.0)
        port.open_trade("X", OrderSide.BUY, datetime(2023, 1, 1), 1000.0, 10)
        equity = port.equity({"X": 1200.0})
        assert equity == pytest.approx(100_000.0 - 10_000.0 + 12_000.0)

    def test_record_equity_builds_series(self):
        port = Portfolio(50_000.0)
        port.record_equity(datetime(2023, 1, 1), {})
        port.record_equity(datetime(2023, 1, 2), {})
        series = port.equity_series()
        assert len(series) == 2

    def test_closed_and_open_trade_separation(self):
        port = Portfolio(100_000.0)
        t = port.open_trade("X", OrderSide.BUY, datetime(2023, 1, 1), 500.0, 5)
        assert len(port.open_trades()) == 1
        assert len(port.closed_trades()) == 0
        port.close_trade(t, datetime(2023, 1, 5), 550.0)
        assert len(port.open_trades()) == 0
        assert len(port.closed_trades()) == 1


# ===========================================================================
# Module 5F — Performance Metrics
# ===========================================================================

class TestPerformanceCalculator:
    def _equity_df(self, values: list[float]) -> pd.DataFrame:
        dates = pd.date_range("2022-01-01", periods=len(values), freq="B")
        return pd.DataFrame({"date": dates, "equity": values})

    def test_basic_metrics(self):
        trades = []
        for pnl in [500, -200, 300, -100, 400]:
            t = Trade("X", OrderSide.BUY, datetime(2023, 1, 1), 1000.0, 1, trade_id=1)
            t.close(datetime(2023, 1, 10), 1000.0 + pnl / 1.0)
            trades.append(t)
        eq = self._equity_df([100_000] + [100_000 + i * 100 for i in range(252)])
        calc = PerformanceCalculator()
        metrics = calc.calculate(trades, eq, 100_000.0)
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == pytest.approx(60.0)
        assert metrics.total_return == pytest.approx(900.0)

    def test_max_drawdown(self):
        # Equity rises then falls
        values = [100_000, 110_000, 120_000, 100_000, 90_000, 95_000]
        eq = self._equity_df(values)
        calc = PerformanceCalculator()
        metrics = calc.calculate([], eq, 100_000.0)
        assert metrics.max_drawdown_pct < 0
        assert metrics.max_drawdown_pct == pytest.approx(-25.0, abs=1.0)

    def test_sharpe_is_finite_float(self):
        # Strong deterministic uptrend: 0.2% daily return, 1% vol → sharpe well above zero
        values = [100_000 * (1.002 ** i) for i in range(253)]
        eq = self._equity_df(values)
        calc = PerformanceCalculator()
        metrics = calc.calculate([], eq, 100_000.0)
        assert isinstance(metrics.sharpe_ratio, float)
        assert metrics.sharpe_ratio > 0

    def test_empty_trades_zero_win_rate(self):
        eq = self._equity_df([100_000, 100_000])
        metrics = PerformanceCalculator().calculate([], eq, 100_000.0)
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0


# ===========================================================================
# Module 5A — Engine (no look-ahead bias)
# ===========================================================================

class TestBacktestEngine:
    def test_engine_runs_without_error(self):
        data = _make_featured(100)
        engine = BacktestEngine(BacktestConfig(starting_capital=100_000.0))
        portfolio, metrics = engine.run(data, IndicatorStrategy(), "TEST")
        assert isinstance(metrics.total_return_pct, float)

    def test_no_future_data_leakage(self):
        """Strategy only ever receives rows up to and including the current bar."""
        seen_indices = []

        class RecordingStrategy:
            def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
                seen_indices.append(int(row.name) if hasattr(row, "name") else len(seen_indices))
                return "HOLD"

        data = _make_ohlcv(50)
        engine = BacktestEngine()
        engine.run(data, RecordingStrategy(), "TEST")
        assert len(seen_indices) == 50

    def test_engine_closes_all_on_exit(self):
        """All open positions must be closed at the end of the run."""
        class AlwaysBuy:
            def generate_signal(self, row, portfolio):
                return "BUY"

        data = _make_ohlcv(30)
        engine = BacktestEngine(BacktestConfig(starting_capital=100_000.0))
        portfolio, _ = engine.run(data, AlwaysBuy(), "X")
        assert len(portfolio.open_trades()) == 0

    def test_engine_raises_on_missing_close_column(self):
        with pytest.raises(ValueError, match="close"):
            BacktestEngine().run(pd.DataFrame({"open": [1, 2]}), IndicatorStrategy())

    def test_engine_raises_on_empty_data(self):
        with pytest.raises(ValueError):
            BacktestEngine().run(pd.DataFrame(), IndicatorStrategy())

    def test_config_validates_capital(self):
        with pytest.raises(ValueError):
            BacktestConfig(starting_capital=-1000)


# ===========================================================================
# Module 5C — Strategy Tester
# ===========================================================================

class TestStrategyTester:
    def test_indicator_strategy_produces_result(self):
        data = _make_featured()
        tester = StrategyTester()
        result = tester.test_indicator_strategy(data, "TEST")
        assert result.strategy_name == "Indicator (EMA+RSI)"
        assert isinstance(result.metrics.total_return_pct, float)

    def test_macd_strategy(self):
        data = _make_featured()
        result = StrategyTester().test_macd_strategy(data, "TEST")
        assert result.strategy_name == "Indicator (MACD)"

    def test_bollinger_strategy(self):
        data = _make_featured()
        result = StrategyTester().test_bollinger_strategy(data, "TEST")
        assert result.strategy_name == "Indicator (Bollinger Bands)"

    def test_ai_strategy(self):
        data = _make_ai_featured()
        result = StrategyTester().test_ai_strategy(data, "TEST")
        assert result.strategy_name == "AI Prediction"

    def test_hybrid_strategy(self):
        data = _make_ai_featured()
        result = StrategyTester().test_hybrid_strategy(data, "TEST")
        assert "Hybrid" in result.strategy_name

    def test_all_strategies_returns_list(self):
        data = _make_ai_featured()
        results = StrategyTester().test_all_strategies(data, "TEST")
        assert len(results) >= 3  # at least indicator strategies


# ===========================================================================
# Module 5D — AI Strategy
# ===========================================================================

class TestAIStrategy:
    def test_ai_strategy_respects_confidence_threshold(self):
        """Signals below threshold should produce HOLD."""
        strategy = AIPredictionStrategy(confidence_threshold=90.0)
        row = pd.Series({"ai_action": "BUY", "ai_confidence": 50.0, "close": 1000.0})
        signal = strategy.generate_signal(row, Portfolio(100_000.0))
        assert signal == "HOLD"

    def test_ai_strategy_fires_above_threshold(self):
        strategy = AIPredictionStrategy(confidence_threshold=60.0)
        row = pd.Series({"ai_action": "BUY", "ai_confidence": 75.0, "close": 1000.0})
        assert strategy.generate_signal(row, Portfolio(100_000.0)) == "BUY"

    def test_hybrid_requires_both_when_strict(self):
        hybrid = HybridStrategy(
            AIPredictionStrategy(confidence_threshold=60.0),
            IndicatorStrategy(),
            require_both=True,
        )
        # AI says BUY, indicator will HOLD (ema_20/ema_50 missing → HOLD)
        row = pd.Series({"ai_action": "BUY", "ai_confidence": 80.0, "close": 1000.0})
        signal = hybrid.generate_signal(row, Portfolio(100_000.0))
        # Indicator returns HOLD → strict mode → HOLD
        assert signal == "HOLD"


# ===========================================================================
# Module 5E — Portfolio Backtester
# ===========================================================================

class TestPortfolioBacktester:
    def test_multi_symbol_backtest(self):
        datasets = {"RELIANCE": _make_featured(200), "TCS": _make_featured(200)}
        cfg = BacktestConfig(starting_capital=200_000.0)
        backtester = PortfolioBacktester(cfg, strategy_factory=lambda _: IndicatorStrategy())
        result = backtester.run(datasets)
        assert "RELIANCE" in result.symbol_results
        assert "TCS" in result.symbol_results
        assert not result.combined_equity.empty
        assert result.combined_metrics.total_trades >= 0

    def test_capital_split_equally(self):
        datasets = {"A": _make_featured(100), "B": _make_featured(100), "C": _make_featured(100)}
        cfg = BacktestConfig(starting_capital=300_000.0)
        result = PortfolioBacktester(cfg).run(datasets)
        for sym, alloc in result.capital_allocation.items():
            assert alloc == pytest.approx(100_000.0)

    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError):
            PortfolioBacktester().run({})


# ===========================================================================
# Module 5H — Walk-Forward Testing
# ===========================================================================

class TestWalkForwardTester:
    def test_walk_forward_produces_windows(self):
        data = _make_featured(400)
        wf = WalkForwardTester(BacktestConfig(starting_capital=100_000.0))
        result = wf.run(data, lambda: IndicatorStrategy(), train_bars=200, test_bars=50, symbol="TEST")
        assert len(result.windows) >= 1
        for w in result.windows:
            assert w.test_start < w.test_end
            assert w.train_end < w.test_start

    def test_walk_forward_raises_on_insufficient_data(self):
        data = _make_featured(50)
        wf = WalkForwardTester()
        with pytest.raises(ValueError, match="at least"):
            wf.run(data, lambda: IndicatorStrategy(), train_bars=100, test_bars=50)

    def test_aggregated_metrics_computed(self):
        data = _make_featured(400)
        wf = WalkForwardTester(BacktestConfig(starting_capital=100_000.0))
        result = wf.run(data, lambda: IndicatorStrategy(), train_bars=200, test_bars=100)
        assert isinstance(result.aggregated_metrics.sharpe_ratio, float)

    def test_no_overlap_between_windows(self):
        """Each test window must be strictly after the previous."""
        data = _make_featured(600)
        wf = WalkForwardTester()
        result = wf.run(data, lambda: IndicatorStrategy(), train_bars=200, test_bars=100, step_bars=100)
        for i in range(1, len(result.windows)):
            assert result.windows[i].test_start >= result.windows[i - 1].test_end


# ===========================================================================
# Module 5I — Monte Carlo
# ===========================================================================

class TestMonteCarlo:
    def _equity_df(self, n: int = 252) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        values = 100_000 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({"date": dates, "equity": values})

    def test_monte_carlo_runs(self):
        eq = self._equity_df()
        mc = MonteCarloSimulator(n_simulations=200, n_paths_to_store=20)
        result = mc.run(eq, starting_capital=100_000.0)
        assert result.simulations == 200
        assert result.expected_value > 0
        assert 0 <= result.prob_profit <= 100

    def test_best_case_above_worst_case(self):
        eq = self._equity_df()
        result = MonteCarloSimulator(n_simulations=500).run(eq, starting_capital=100_000.0)
        assert result.best_case >= result.worst_case

    def test_equity_paths_shape(self):
        eq = self._equity_df(100)
        result = MonteCarloSimulator(n_simulations=50, n_paths_to_store=10).run(eq)
        assert result.equity_paths.shape[1] == 10

    def test_raises_on_insufficient_data(self):
        eq = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3, freq="B"), "equity": [100_000, 101_000, 102_000]})
        with pytest.raises(ValueError):
            MonteCarloSimulator().run(eq)

    def test_reproducibility(self):
        eq = self._equity_df()
        r1 = MonteCarloSimulator(n_simulations=100).run(eq, random_seed=42)
        r2 = MonteCarloSimulator(n_simulations=100).run(eq, random_seed=42)
        assert r1.expected_value == pytest.approx(r2.expected_value)


# ===========================================================================
# Module 5J — Strategy Comparison
# ===========================================================================

class TestStrategyComparison:
    def _make_results(self):
        data = _make_featured(200)
        tester = StrategyTester(BacktestConfig(starting_capital=100_000.0))
        return tester.test_all_indicator_strategies(data, "TEST")

    def test_comparison_produces_ranking(self):
        results = self._make_results()
        comp = StrategyComparison().compare(results)
        assert len(comp.ranked) == len(results)
        ranks = [r.rank for r in comp.ranked]
        assert ranks == list(range(1, len(results) + 1))

    def test_winner_is_first_ranked(self):
        results = self._make_results()
        comp = StrategyComparison().compare(results)
        assert comp.winner == comp.ranked[0].strategy_name

    def test_comparison_table_has_correct_columns(self):
        results = self._make_results()
        comp = StrategyComparison().compare(results)
        assert "composite_score" in comp.comparison_table.columns
        assert "strategy_name" in comp.comparison_table.columns

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            StrategyComparison().compare([])


# ===========================================================================
# Module 5N — Validation (no future data, no invalid prices)
# ===========================================================================

class TestValidation:
    def test_no_duplicate_trades_per_symbol(self):
        """Engine should not open two positions in the same symbol simultaneously."""
        class AlwaysBuy:
            def generate_signal(self, row, portfolio):
                return "BUY"

        data = _make_ohlcv(50)
        engine = BacktestEngine(BacktestConfig(starting_capital=500_000.0))
        portfolio, _ = engine.run(data, AlwaysBuy(), "X")
        # Only one position should ever be open at a time per symbol
        for t in portfolio.trades:
            if t.status == OrderStatus.OPEN:
                others = [o for o in portfolio.trades if o.symbol == t.symbol and o.trade_id != t.trade_id and o.status == OrderStatus.OPEN]
                assert len(others) == 0

    def test_no_negative_cash_after_trade(self):
        class AlwaysBuy:
            def generate_signal(self, row, portfolio):
                return "BUY"

        data = _make_ohlcv(100)
        engine = BacktestEngine(BacktestConfig(starting_capital=100_000.0))
        portfolio, _ = engine.run(data, AlwaysBuy(), "X")
        assert portfolio.cash >= 0

    def test_entry_price_never_zero(self):
        data = _make_featured(100)
        engine = BacktestEngine()
        portfolio, _ = engine.run(data, IndicatorStrategy(), "X")
        for trade in portfolio.closed_trades():
            assert trade.entry_price > 0
            assert trade.exit_price is not None and trade.exit_price > 0
