from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.portfolio import Portfolio
from backtesting.results import BacktestMetrics, PerformanceCalculator
from backtesting.strategy_tester import IndicatorStrategy, StrategyResult
from backtesting.trade import Trade


@dataclass
class PortfolioBacktestResult:
    symbol_results: dict[str, StrategyResult]
    combined_equity: pd.DataFrame
    combined_metrics: BacktestMetrics
    capital_allocation: dict[str, float]

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for sym, result in self.symbol_results.items():
            m = result.metrics
            rows.append(
                {
                    "symbol": sym,
                    "total_return_pct": m.total_return_pct,
                    "annualized_return": m.annualized_return,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate": m.win_rate,
                    "total_trades": m.total_trades,
                }
            )
        rows.append(
            {
                "symbol": "PORTFOLIO",
                "total_return_pct": self.combined_metrics.total_return_pct,
                "annualized_return": self.combined_metrics.annualized_return,
                "sharpe_ratio": self.combined_metrics.sharpe_ratio,
                "max_drawdown_pct": self.combined_metrics.max_drawdown_pct,
                "win_rate": self.combined_metrics.win_rate,
                "total_trades": self.combined_metrics.total_trades,
            }
        )
        return pd.DataFrame(rows)


StrategyFactory = Callable[[str], object]


class PortfolioBacktester:
    """
    Runs a strategy across multiple symbols with equal capital allocation.

    Parameters
    ----------
    config:           Global BacktestConfig. Starting capital is split equally.
    strategy_factory: Callable(symbol) -> strategy instance.
                      Allows per-symbol strategy customisation.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        strategy_factory: StrategyFactory | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.strategy_factory: StrategyFactory = strategy_factory or (lambda _sym: IndicatorStrategy())
        self.calculator = PerformanceCalculator()
        self.logger = logging.getLogger(__name__)

    def run(self, datasets: dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        """
        Run the backtest for all symbols.

        Parameters
        ----------
        datasets: {symbol: ohlcv_dataframe}  — one DataFrame per symbol.
        """
        if not datasets:
            raise ValueError("No datasets provided to PortfolioBacktester.")

        n = len(datasets)
        per_symbol_capital = self.config.starting_capital / n
        capital_allocation = {sym: round(per_symbol_capital, 2) for sym in datasets}

        symbol_results: dict[str, StrategyResult] = {}
        all_equity_frames: list[pd.DataFrame] = []
        all_trades: list[Trade] = []

        for symbol, data in datasets.items():
            symbol_config = BacktestConfig(
                starting_capital=per_symbol_capital,
                risk_per_trade_pct=self.config.risk_per_trade_pct,
                max_open_positions=self.config.max_open_positions,
                order_config=self.config.order_config,
            )
            engine = BacktestEngine(symbol_config)
            strategy = self.strategy_factory(symbol)
            portfolio, metrics = engine.run(data, strategy, symbol)  # type: ignore[arg-type]

            result = StrategyResult(
                strategy_name=f"{strategy.__class__.__name__}_{symbol}",
                metrics=metrics,
                trades_df=pd.DataFrame([t.to_dict() for t in portfolio.closed_trades()]),
                equity_df=portfolio.equity_series(),
            )
            symbol_results[symbol] = result
            all_trades.extend(portfolio.closed_trades())

            if not portfolio.equity_series().empty:
                eq = portfolio.equity_series().copy()
                eq["symbol"] = symbol
                all_equity_frames.append(eq)

            self.logger.info("Symbol backtest complete", extra={"symbol": symbol, "return_pct": metrics.total_return_pct})

        combined_equity = self._combine_equity(all_equity_frames)
        combined_metrics = self.calculator.calculate(all_trades, combined_equity, self.config.starting_capital)

        return PortfolioBacktestResult(
            symbol_results=symbol_results,
            combined_equity=combined_equity,
            combined_metrics=combined_metrics,
            capital_allocation=capital_allocation,
        )

    @staticmethod
    def _combine_equity(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Sum per-symbol equity on each date to get total portfolio equity."""
        if not frames:
            return pd.DataFrame(columns=["date", "equity"])
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        summed = combined.groupby("date")["equity"].sum().reset_index()
        return summed.sort_values("date").reset_index(drop=True)
