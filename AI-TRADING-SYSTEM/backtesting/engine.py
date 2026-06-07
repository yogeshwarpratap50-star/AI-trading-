from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from backtesting.order_simulator import OrderConfig, OrderSimulator
from backtesting.portfolio import Portfolio
from backtesting.results import BacktestMetrics, PerformanceCalculator
from backtesting.trade import OrderSide

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"close"}


@dataclass
class BacktestConfig:
    """All tuneable parameters for a backtest run."""

    starting_capital: float = 100_000.0
    risk_per_trade_pct: float = 2.0      # % of current capital risked per trade
    max_open_positions: int = 5
    order_config: OrderConfig = field(default_factory=OrderConfig)

    # Validation
    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive.")
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade_pct must be in (0, 100].")


@runtime_checkable
class Strategy(Protocol):
    """Any object with a generate_signal(row, portfolio) -> str method."""

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        """Return 'BUY', 'SELL', or 'HOLD'."""
        ...


class BacktestEngine:
    """
    Bar-by-bar simulation engine.

    Iterates chronologically over an OHLCV DataFrame (no future data is
    accessible to the strategy — look-ahead bias prevention is structural).
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.simulator = OrderSimulator(self.config.order_config)
        self.calculator = PerformanceCalculator()

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "STOCK",
    ) -> tuple[Portfolio, BacktestMetrics]:
        """
        Run the backtest and return the final portfolio and performance metrics.

        Parameters
        ----------
        data:     Chronologically sorted OHLCV DataFrame (must contain 'close').
        strategy: Object implementing the Strategy protocol.
        symbol:   Symbol name attached to trades for reporting.
        """
        self._validate(data)
        data = self._prepare(data)
        portfolio = Portfolio(self.config.starting_capital)

        for _, row in data.iterrows():
            date = pd.to_datetime(row["_date"])
            close = float(row["close"])
            high = float(row.get("high", close))
            low = float(row.get("low", close))

            # 1. Generate signal using only information up to and including this bar
            signal = strategy.generate_signal(row, portfolio)

            # 2. Process signal
            open_trades = portfolio.open_trades()
            has_position = portfolio.has_open_position(symbol)

            if signal == "BUY" and not has_position and len(portfolio.positions) < self.config.max_open_positions:
                fill, fees, slippage = self.simulator.execute_market(close, 1, "BUY")
                qty = self.simulator.calculate_quantity(portfolio.cash, fill, self.config.risk_per_trade_pct)
                if qty > 0:
                    portfolio.open_trade(
                        symbol, OrderSide.BUY, date, fill, qty,
                        fees=fees * qty, slippage=slippage, reason=str(row.get("_signal_reason", "BUY signal")),
                    )

            elif signal == "SELL" and has_position:
                for trade in open_trades:
                    if trade.symbol == symbol:
                        fill, fees, slippage = self.simulator.execute_market(close, trade.quantity, "SELL")
                        portfolio.close_trade(trade, date, fill, fees, slippage)

            # 3. Mark-to-market
            portfolio.record_equity(date, {symbol: close})

        # Force-close any remaining open positions at last bar close
        self._close_all_open(portfolio, data, symbol)

        equity_df = portfolio.equity_series()
        metrics = self.calculator.calculate(
            portfolio.closed_trades(), equity_df, self.config.starting_capital
        )
        logger.info(
            "Backtest complete",
            extra={"symbol": symbol, "trades": metrics.total_trades, "return_pct": metrics.total_return_pct},
        )
        return portfolio, metrics

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        missing = _REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _prepare(data: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and ensure a parseable date column."""
        df = data.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        # Determine the date column (or fall back to the DataFrame index)
        if "date" in df.columns:
            df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["_date"] = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_values("_date").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        return df

    def _close_all_open(self, portfolio: Portfolio, data: pd.DataFrame, symbol: str) -> None:
        last = data.iloc[-1]
        last_date = pd.to_datetime(last["_date"])
        last_close = float(last["close"])
        for trade in portfolio.open_trades():
            if trade.symbol == symbol:
                fill, fees, slippage = self.simulator.execute_market(last_close, trade.quantity, "SELL")
                portfolio.close_trade(trade, last_date, fill, fees, slippage)
