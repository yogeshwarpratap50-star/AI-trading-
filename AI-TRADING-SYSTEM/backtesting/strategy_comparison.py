from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from backtesting.results import BacktestMetrics
from backtesting.strategy_tester import StrategyResult


@dataclass
class RankedStrategy:
    rank: int
    strategy_name: str
    composite_score: float
    metrics: BacktestMetrics


@dataclass
class ComparisonResult:
    ranked: list[RankedStrategy]
    comparison_table: pd.DataFrame
    winner: str

    def summary(self) -> str:
        lines = [f"{'Rank':<6}{'Strategy':<40}{'Score':<10}{'Return%':<12}{'Sharpe':<10}{'MaxDD%':<10}{'WinRate%':<10}"]
        lines.append("-" * 88)
        for r in self.ranked:
            m = r.metrics
            lines.append(
                f"{r.rank:<6}{r.strategy_name:<40}{r.composite_score:<10.3f}"
                f"{m.total_return_pct:<12.2f}{m.sharpe_ratio:<10.3f}"
                f"{m.max_drawdown_pct:<10.2f}{m.win_rate:<10.1f}"
            )
        return "\n".join(lines)


class StrategyComparison:
    """
    Ranks backtest results by a composite score:

        score = 0.30 * annualised_return_rank
              + 0.25 * sharpe_rank
              + 0.20 * max_drawdown_rank  (lower drawdown = higher rank)
              + 0.15 * win_rate_rank
              + 0.10 * profit_factor_rank

    Each component is normalised to [0, 1] within the comparison set.
    A higher composite score is better.
    """

    WEIGHTS = {
        "annualized_return": 0.30,
        "sharpe_ratio": 0.25,
        "max_drawdown_pct": 0.20,   # will be inverted (less is better)
        "win_rate": 0.15,
        "profit_factor": 0.10,
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def compare(self, results: list[StrategyResult]) -> ComparisonResult:
        if not results:
            raise ValueError("No strategy results to compare.")

        rows = []
        for r in results:
            m = r.metrics
            rows.append(
                {
                    "strategy_name": r.strategy_name,
                    "annualized_return": m.annualized_return,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate": m.win_rate,
                    "profit_factor": min(m.profit_factor, 100.0),
                    "total_return_pct": m.total_return_pct,
                    "sortino_ratio": m.sortino_ratio,
                    "calmar_ratio": m.calmar_ratio,
                    "total_trades": m.total_trades,
                    "expectancy": m.expectancy,
                    "_result": r,
                }
            )
        df = pd.DataFrame(rows)
        df["composite_score"] = self._composite_score(df)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        ranked = [
            RankedStrategy(
                rank=i + 1,
                strategy_name=row["strategy_name"],
                composite_score=round(float(row["composite_score"]), 4),
                metrics=row["_result"].metrics,
            )
            for i, row in df.iterrows()
        ]

        comparison_table = df.drop(columns=["_result"]).round(4)

        winner = ranked[0].strategy_name
        self.logger.info("Strategy comparison complete", extra={"winner": winner, "strategies": len(results)})
        return ComparisonResult(ranked=ranked, comparison_table=comparison_table, winner=winner)

    def _composite_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        for metric, weight in self.WEIGHTS.items():
            col = df[metric].astype(float).fillna(0)
            col_min, col_max = col.min(), col.max()
            if col_max == col_min:
                normalised = pd.Series(0.5, index=df.index)
            else:
                normalised = (col - col_min) / (col_max - col_min)
            # Invert drawdown: a smaller (more negative) drawdown is better
            if metric == "max_drawdown_pct":
                normalised = 1.0 - normalised
            score += normalised * weight
        return score
