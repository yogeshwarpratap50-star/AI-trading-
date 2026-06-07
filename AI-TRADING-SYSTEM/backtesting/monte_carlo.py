from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    starting_capital: float
    # Distribution of terminal portfolio values
    expected_value: float
    median_value: float
    best_case: float          # 95th percentile
    worst_case: float         # 5th percentile
    value_at_risk_5: float    # VaR — loss at 5th percentile
    cvar_5: float             # CVaR / Expected Shortfall at 5%
    prob_profit: float        # % of simulations that end above starting capital
    prob_loss_10: float       # % that lose > 10%
    prob_loss_25: float       # % that lose > 25%
    # Drawdown distribution
    expected_max_drawdown: float
    worst_max_drawdown: float
    # Equity paths (sampled subset for charting)
    equity_paths: pd.DataFrame

    def summary_str(self) -> str:
        lines = [
            f"Monte Carlo ({self.simulations:,} simulations)",
            "=" * 48,
            f"  Expected Final Value  : ₹{self.expected_value:,.2f}",
            f"  Median Final Value    : ₹{self.median_value:,.2f}",
            f"  Best Case  (95th %)  : ₹{self.best_case:,.2f}",
            f"  Worst Case  (5th %)  : ₹{self.worst_case:,.2f}",
            f"  Value at Risk (5%)   : ₹{self.value_at_risk_5:,.2f}",
            f"  CVaR (5%)            : ₹{self.cvar_5:,.2f}",
            f"  P(profit)            : {self.prob_profit:.1f}%",
            f"  P(loss > 10%)        : {self.prob_loss_10:.1f}%",
            f"  P(loss > 25%)        : {self.prob_loss_25:.1f}%",
            f"  Exp. Max Drawdown    : {self.expected_max_drawdown:.2f}%",
            f"  Worst Max Drawdown   : {self.worst_max_drawdown:.2f}%",
            "=" * 48,
        ]
        return "\n".join(lines)


class MonteCarloSimulator:
    """
    Bootstrapped Monte Carlo simulation of a trading strategy's equity curve.

    Method
    ------
    1. Compute the empirical daily-return distribution from the backtest equity curve.
    2. Resample (with replacement) ``n_simulations`` independent paths of
       ``n_days`` daily returns.
    3. Compound each path from ``starting_capital`` to generate terminal values.

    This avoids assumptions about return distributions and implicitly captures
    autocorrelation and fat-tail events present in the original strategy.
    """

    def __init__(self, n_simulations: int = 1000, n_paths_to_store: int = 100) -> None:
        self.n_simulations = n_simulations
        self.n_paths_to_store = min(n_paths_to_store, n_simulations)
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        equity_series: pd.DataFrame,
        starting_capital: float | None = None,
        random_seed: int | None = 42,
    ) -> MonteCarloResult:
        """
        Parameters
        ----------
        equity_series:    DataFrame with 'date' and 'equity' columns (backtest output).
        starting_capital: Override starting value; defaults to first equity observation.
        random_seed:      For reproducibility; pass None for true randomness.
        """
        if equity_series.empty or len(equity_series) < 5:
            raise ValueError("Need at least 5 equity observations for Monte Carlo simulation.")

        rng = np.random.default_rng(random_seed)
        equity = equity_series["equity"].values.astype(float)
        capital = float(starting_capital or equity[0])
        n_days = len(equity) - 1

        daily_returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)

        # Bootstrap: sample n_days returns with replacement for each simulation
        sampled = rng.choice(daily_returns, size=(self.n_simulations, n_days), replace=True)
        # Shape: (n_simulations, n_days)

        # Compound returns to produce equity paths
        cum_returns = np.cumprod(1 + sampled, axis=1)          # (S, D)
        paths = capital * np.hstack([np.ones((self.n_simulations, 1)), cum_returns])  # include t=0

        terminal = paths[:, -1]

        # --- Terminal value statistics ---
        expected_value = float(np.mean(terminal))
        median_value = float(np.median(terminal))
        best_case = float(np.percentile(terminal, 95))
        worst_case = float(np.percentile(terminal, 5))
        var_5 = capital - worst_case
        below_5pct = terminal[terminal <= worst_case]
        cvar_5 = capital - float(np.mean(below_5pct)) if len(below_5pct) > 0 else var_5

        prob_profit = float(np.mean(terminal > capital) * 100)
        prob_loss_10 = float(np.mean(terminal < capital * 0.90) * 100)
        prob_loss_25 = float(np.mean(terminal < capital * 0.75) * 100)

        # --- Drawdown per path ---
        peak = np.maximum.accumulate(paths, axis=1)
        dd_pct = (paths - peak) / np.where(peak > 0, peak, 1) * 100
        max_dd_per_sim = dd_pct.min(axis=1)  # most negative value per sim

        expected_max_drawdown = float(np.mean(max_dd_per_sim))
        worst_max_drawdown = float(np.percentile(max_dd_per_sim, 5))

        # --- Store a random subset of paths for charting ---
        sample_idx = rng.choice(self.n_simulations, size=self.n_paths_to_store, replace=False)
        sampled_paths = paths[sample_idx, :]  # (n_paths_to_store, n_days+1)

        equity_paths_df = pd.DataFrame(
            sampled_paths.T,
            columns=[f"sim_{i}" for i in range(self.n_paths_to_store)],
        )
        # Attach the original date index if we have one
        if "date" in equity_series.columns:
            dates = pd.to_datetime(equity_series["date"])
            equity_paths_df.index = dates
        equity_paths_df.index.name = "date"

        self.logger.info(
            "Monte Carlo complete",
            extra={
                "simulations": self.n_simulations,
                "expected_value": expected_value,
                "prob_profit": prob_profit,
            },
        )

        return MonteCarloResult(
            simulations=self.n_simulations,
            starting_capital=capital,
            expected_value=round(expected_value, 2),
            median_value=round(median_value, 2),
            best_case=round(best_case, 2),
            worst_case=round(worst_case, 2),
            value_at_risk_5=round(var_5, 2),
            cvar_5=round(cvar_5, 2),
            prob_profit=round(prob_profit, 2),
            prob_loss_10=round(prob_loss_10, 2),
            prob_loss_25=round(prob_loss_25, 2),
            expected_max_drawdown=round(expected_max_drawdown, 2),
            worst_max_drawdown=round(worst_max_drawdown, 2),
            equity_paths=equity_paths_df,
        )

    def generate_chart(
        self,
        result: MonteCarloResult,
        output_path: str = "reports/monte_carlo.png",
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed — Monte Carlo chart skipped")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Monte Carlo Analysis ({result.simulations:,} Simulations)", fontsize=12)

        # Left: equity path fan chart
        ax = axes[0]
        for col in result.equity_paths.columns:
            ax.plot(result.equity_paths[col].values, color="steelblue", alpha=0.05, linewidth=0.5)
        ax.axhline(result.starting_capital, color="black", linewidth=1, linestyle="--", label="Starting Capital")
        ax.axhline(result.expected_value, color="green", linewidth=1.5, linestyle="-", label=f"Expected: ₹{result.expected_value:,.0f}")
        ax.axhline(result.worst_case, color="red", linewidth=1.5, linestyle="--", label=f"5th pct: ₹{result.worst_case:,.0f}")
        ax.set_title("Simulated Equity Paths")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: terminal value distribution
        ax2 = axes[1]
        # Re-generate terminal values from stored paths
        terminal = result.equity_paths.iloc[-1].values
        ax2.hist(terminal, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax2.axvline(result.starting_capital, color="black", linewidth=1.5, linestyle="--", label="Starting Capital")
        ax2.axvline(result.expected_value, color="green", linewidth=1.5, label=f"Expected: ₹{result.expected_value:,.0f}")
        ax2.axvline(result.worst_case, color="red", linewidth=1.5, linestyle="--", label=f"5th pct: ₹{result.worst_case:,.0f}")
        ax2.set_title("Terminal Value Distribution")
        ax2.set_xlabel("Final Portfolio Value (₹)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Monte Carlo chart saved", extra={"path": output_path})
        return output_path
