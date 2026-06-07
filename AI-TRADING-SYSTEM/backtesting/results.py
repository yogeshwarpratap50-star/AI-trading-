from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.trade import Trade

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.065  # 6.5% — approximate Indian risk-free rate


@dataclass
class BacktestMetrics:
    """Complete set of performance metrics for a backtest run."""

    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    loss_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float

    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float

    # Costs
    total_fees: float
    total_slippage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_str(self) -> str:
        lines = [
            "=" * 48,
            f"  Total Return      : {self.total_return_pct:+.2f}%",
            f"  Annualized Return : {self.annualized_return:+.2f}%",
            f"  Sharpe Ratio      : {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio     : {self.sortino_ratio:.3f}",
            f"  Max Drawdown      : {self.max_drawdown_pct:.2f}%",
            f"  Calmar Ratio      : {self.calmar_ratio:.3f}",
            f"  Win Rate          : {self.win_rate:.1f}%",
            f"  Profit Factor     : {self.profit_factor:.3f}",
            f"  Expectancy        : {self.expectancy:.2f}",
            f"  Total Trades      : {self.total_trades}",
            f"  Total Fees        : {self.total_fees:.2f}",
            "=" * 48,
        ]
        return "\n".join(lines)


class PerformanceCalculator:
    """Computes all backtest performance metrics from trades and equity curve."""

    def calculate(
        self,
        trades: list[Trade],
        equity_series: pd.DataFrame,
        starting_capital: float,
    ) -> BacktestMetrics:
        closed = [t for t in trades if t.pnl is not None]
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed)
        total_return_pct = (total_pnl / starting_capital * 100) if starting_capital > 0 else 0.0

        win_rate = (len(wins) / len(closed) * 100) if closed else 0.0
        loss_rate = (len(losses) / len(closed) * 100) if closed else 0.0
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        expectancy = (win_rate / 100.0 * avg_win) + (loss_rate / 100.0 * avg_loss)

        ann_return = self._annualized_return(equity_series, starting_capital)
        sharpe, sortino = self._risk_ratios(equity_series)
        max_dd, max_dd_pct = self._max_drawdown(equity_series)

        calmar = (ann_return / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0
        recovery = (total_pnl / abs(max_dd)) if max_dd != 0 else 0.0

        total_fees = sum(t.fees for t in closed)
        total_slippage = sum(t.slippage for t in closed)

        return BacktestMetrics(
            total_return=round(total_pnl, 2),
            total_return_pct=round(total_return_pct, 4),
            annualized_return=round(ann_return, 4),
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 2),
            loss_rate=round(loss_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(min(profit_factor, 9999.0), 4),
            expectancy=round(expectancy, 2),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            recovery_factor=round(recovery, 4),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            total_fees=round(total_fees, 2),
            total_slippage=round(total_slippage, 2),
        )

    # ------------------------------------------------------------------
    # Equity-curve helpers
    # ------------------------------------------------------------------

    def _annualized_return(self, equity_series: pd.DataFrame, starting_capital: float) -> float:
        if equity_series.empty or len(equity_series) < 2 or starting_capital <= 0:
            return 0.0
        final_eq = float(equity_series["equity"].iloc[-1])
        start_date = pd.to_datetime(equity_series["date"].iloc[0])
        end_date = pd.to_datetime(equity_series["date"].iloc[-1])
        n_days = (end_date - start_date).days
        if n_days <= 0:
            return 0.0
        years = n_days / 365.25
        total_return = final_eq / starting_capital
        if total_return <= 0:
            return -100.0
        return (total_return ** (1.0 / years) - 1.0) * 100.0

    def _risk_ratios(self, equity_series: pd.DataFrame) -> tuple[float, float]:
        if equity_series.empty or len(equity_series) < 3:
            return 0.0, 0.0
        equity = equity_series["equity"].values.astype(float)
        daily_returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        if len(daily_returns) == 0:
            return 0.0, 0.0
        rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        excess = daily_returns - rf_daily
        std_excess = float(np.std(excess, ddof=1))
        if std_excess == 0:
            return 0.0, 0.0
        sharpe = float(np.mean(excess) / std_excess * np.sqrt(TRADING_DAYS_PER_YEAR))
        downside = excess[excess < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0.0
        return sharpe, sortino

    def _max_drawdown(self, equity_series: pd.DataFrame) -> tuple[float, float]:
        if equity_series.empty:
            return 0.0, 0.0
        equity = equity_series["equity"].values.astype(float)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        drawdown_pct = np.where(peak > 0, drawdown / peak, 0.0)
        return float(np.min(drawdown)), float(np.min(drawdown_pct)) * 100.0


class EquityCurveGenerator:
    """Generates equity curve, drawdown, and portfolio value charts."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_charts(
        self,
        equity_series: pd.DataFrame,
        metrics: BacktestMetrics,
        strategy_name: str,
        save: bool = True,
    ) -> dict[str, str]:
        """Generate PNG and interactive HTML equity curve charts. Returns paths."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("matplotlib not installed — skipping chart generation")
            return {}

        if equity_series.empty:
            return {}

        eq = equity_series.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.sort_values("date").reset_index(drop=True)
        equity = eq["equity"].values.astype(float)
        peak = np.maximum.accumulate(equity)
        drawdown_pct = np.where(peak > 0, (equity - peak) / peak * 100, 0.0)

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Backtest — {strategy_name}", fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

        # Equity curve
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(eq["date"], equity, color="#1f77b4", linewidth=1.5, label="Portfolio Value")
        ax1.axhline(equity[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Starting Capital")
        ax1.set_title("Equity Curve", fontsize=10)
        ax1.set_ylabel("Portfolio Value (₹)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[1])
        ax2.fill_between(eq["date"], drawdown_pct, 0, color="#d62728", alpha=0.7, label="Drawdown %")
        ax2.set_title(f"Drawdown  (Max: {metrics.max_drawdown_pct:.2f}%)", fontsize=10)
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Daily returns
        ax3 = fig.add_subplot(gs[2])
        if len(equity) > 1:
            daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1) * 100
            colors = ["#2ca02c" if r >= 0 else "#d62728" for r in daily_ret]
            ax3.bar(eq["date"].iloc[1:], daily_ret, color=colors, alpha=0.8, width=1.0)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_title("Daily Returns (%)", fontsize=10)
        ax3.set_ylabel("Return (%)")
        ax3.grid(alpha=0.3)

        safe_name = strategy_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
        png_path = self.output_dir / f"equity_{safe_name}.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Interactive HTML via plotly if available
        html_path_str = ""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_html = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=("Equity Curve", "Drawdown %", "Daily Returns %"),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25],
            )
            fig_html.add_trace(go.Scatter(x=eq["date"], y=equity, name="Portfolio Value", line=dict(color="#1f77b4")), row=1, col=1)
            fig_html.add_trace(go.Scatter(x=eq["date"], y=drawdown_pct, name="Drawdown %", fill="tozeroy", line=dict(color="#d62728")), row=2, col=1)
            if len(equity) > 1:
                daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1) * 100
                bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in daily_ret]
                fig_html.add_trace(go.Bar(x=eq["date"].iloc[1:], y=daily_ret, name="Daily Return %", marker_color=bar_colors), row=3, col=1)
            fig_html.update_layout(
                title=f"Backtest — {strategy_name}",
                height=800,
                showlegend=True,
                template="plotly_white",
            )
            html_path = self.output_dir / f"equity_{safe_name}.html"
            fig_html.write_html(str(html_path))
            html_path_str = str(html_path)
        except ImportError:
            logger.info("plotly not installed — HTML chart skipped")

        logger.info("Charts generated", extra={"strategy": strategy_name})
        return {"png": str(png_path), "html": html_path_str}
