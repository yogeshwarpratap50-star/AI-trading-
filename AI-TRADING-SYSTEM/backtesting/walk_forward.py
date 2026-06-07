from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.results import BacktestMetrics, PerformanceCalculator


@dataclass
class WalkForwardWindow:
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: BacktestMetrics
    trades_df: pd.DataFrame
    equity_df: pd.DataFrame


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    aggregated_metrics: BacktestMetrics
    combined_equity: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            m = w.metrics
            rows.append(
                {
                    "window": w.window_index,
                    "train_start": str(w.train_start.date()),
                    "train_end": str(w.train_end.date()),
                    "test_start": str(w.test_start.date()),
                    "test_end": str(w.test_end.date()),
                    "total_return_pct": m.total_return_pct,
                    "annualized_return": m.annualized_return,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate": m.win_rate,
                    "total_trades": m.total_trades,
                }
            )
        return pd.DataFrame(rows)


class WalkForwardTester:
    """
    Anchored walk-forward testing.

    Each iteration:
      1. Train period: [train_start .. test_start)   (strategy can fit itself here)
      2. Test period:  [test_start .. test_end)       (out-of-sample evaluation)
      3. Advance: test window shifts forward by step_bars.

    The strategy is NOT retrained here (no ML involved at the engine level);
    walk-forward exercises each test window independently so that aggregated
    metrics reflect out-of-sample performance without look-ahead bias.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.calculator = PerformanceCalculator()
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        data: pd.DataFrame,
        strategy_factory,          # callable() -> strategy instance (fresh per window)
        train_bars: int = 252,
        test_bars: int = 63,       # ~1 quarter
        step_bars: int | None = None,
        symbol: str = "STOCK",
    ) -> WalkForwardResult:
        """
        Parameters
        ----------
        data:             Full historical OHLCV DataFrame.
        strategy_factory: Callable that returns a fresh strategy for each window.
        train_bars:       Number of bars in the training window.
        test_bars:        Number of bars in each out-of-sample test window.
        step_bars:        How far to advance between windows (defaults to test_bars).
        symbol:           Symbol name for trade records.
        """
        if step_bars is None:
            step_bars = test_bars

        engine = BacktestEngine(self.config)
        data = engine._prepare(data)   # normalise once

        min_bars = train_bars + test_bars
        if len(data) < min_bars:
            raise ValueError(
                f"Data has {len(data)} bars but walk-forward requires at least "
                f"{min_bars} (train={train_bars} + test={test_bars})."
            )

        windows: list[WalkForwardWindow] = []
        window_index = 0
        test_start_idx = train_bars

        while test_start_idx + test_bars <= len(data):
            train_slice = data.iloc[:test_start_idx]
            test_slice = data.iloc[test_start_idx : test_start_idx + test_bars]

            # Fresh strategy for each window to avoid state bleed-over
            strategy = strategy_factory()

            portfolio, metrics = engine.run(test_slice, strategy, symbol)
            trades_df = pd.DataFrame([t.to_dict() for t in portfolio.closed_trades()])
            equity_df = portfolio.equity_series()

            train_dates = pd.to_datetime(train_slice["_date"])
            test_dates = pd.to_datetime(test_slice["_date"])

            windows.append(
                WalkForwardWindow(
                    window_index=window_index,
                    train_start=train_dates.iloc[0],
                    train_end=train_dates.iloc[-1],
                    test_start=test_dates.iloc[0],
                    test_end=test_dates.iloc[-1],
                    metrics=metrics,
                    trades_df=trades_df,
                    equity_df=equity_df,
                )
            )
            self.logger.info(
                "Walk-forward window complete",
                extra={
                    "window": window_index,
                    "test_start": str(test_dates.iloc[0].date()),
                    "return_pct": metrics.total_return_pct,
                },
            )
            window_index += 1
            test_start_idx += step_bars

        if not windows:
            raise ValueError("No walk-forward windows were generated.")

        combined_equity = self._combine_equity(windows)
        all_trades = [t for w in windows for t in self._df_to_trades(w.trades_df)]
        agg_metrics = self.calculator.calculate(all_trades, combined_equity, self.config.starting_capital)

        self.logger.info(
            "Walk-forward complete",
            extra={"windows": len(windows), "total_return_pct": agg_metrics.total_return_pct},
        )
        return WalkForwardResult(windows=windows, aggregated_metrics=agg_metrics, combined_equity=combined_equity)

    @staticmethod
    def _combine_equity(windows: list[WalkForwardWindow]) -> pd.DataFrame:
        frames = [w.equity_df for w in windows if not w.equity_df.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "equity"])
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        return combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    @staticmethod
    def _df_to_trades(df: pd.DataFrame) -> list:
        from backtesting.trade import Trade, OrderSide, OrderStatus
        from datetime import datetime as dt

        trades = []
        for row in df.to_dict("records"):
            if row.get("pnl") is None:
                continue
            t = Trade.__new__(Trade)
            t.pnl = float(row["pnl"])
            t.pnl_pct = (float(row.get("pnl_pct") or 0)) / 100
            t.fees = float(row.get("fees", 0))
            t.slippage = float(row.get("slippage", 0))
            t.status = OrderStatus.CLOSED
            t.side = OrderSide.BUY
            t.symbol = str(row.get("symbol", ""))
            t.entry_date = dt.now()
            t.exit_date = dt.now()
            t.entry_price = float(row.get("entry_price", 0))
            t.exit_price = float(row.get("exit_price", 0))
            t.quantity = int(row.get("quantity", 0))
            t.reason = str(row.get("reason", ""))
            t.trade_id = int(row.get("trade_id", 0))
            trades.append(t)
        return trades
