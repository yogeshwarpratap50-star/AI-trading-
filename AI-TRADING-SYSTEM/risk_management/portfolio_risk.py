from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioRiskReport:
    total_value: float
    cash: float
    used_capital: float
    portfolio_exposure_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    sector_exposure: dict[str, float]     # sector → ₹ value
    sector_exposure_pct: dict[str, float] # sector → % of portfolio
    correlation_risk: float               # average pairwise correlation (0–1)
    market_beta: float                    # estimated portfolio beta vs Nifty
    risk_score: float                     # composite 0–100 (lower is safer)
    risk_level: str                       # LOW / MEDIUM / HIGH / CRITICAL
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "used_capital": round(self.used_capital, 2),
            "portfolio_exposure_pct": round(self.portfolio_exposure_pct, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "correlation_risk": round(self.correlation_risk, 4),
            "market_beta": round(self.market_beta, 4),
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "warnings": self.warnings,
        }


class PortfolioRiskAnalyzer:
    """
    Computes portfolio-level risk metrics from current positions
    and optional historical returns.
    """

    def __init__(self, sector_map: dict[str, str] | None = None) -> None:
        self.sector_map = sector_map or {}
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        positions: list[dict[str, Any]],   # from PaperPortfolio.positions[s].to_dict()
        cash: float,
        starting_capital: float,
        returns_df: pd.DataFrame | None = None,   # columns = symbols, rows = daily returns
        nifty_returns: pd.Series | None = None,
    ) -> PortfolioRiskReport:
        total_mkt_val = sum(float(p["market_value"]) for p in positions)
        used_capital = sum(float(p.get("avg_buy_price", 0)) * int(p["quantity"]) for p in positions)
        total_value = cash + total_mkt_val
        exposure_pct = (total_mkt_val / total_value * 100) if total_value > 0 else 0.0
        unrealized_pnl = sum(float(p.get("unrealized_pnl", 0)) for p in positions)
        unrealized_pnl_pct = (unrealized_pnl / used_capital * 100) if used_capital > 0 else 0.0

        # Sector exposure
        sector_exp: dict[str, float] = {}
        for p in positions:
            sym = p["symbol"]
            sector = self.sector_map.get(sym, "Unknown")
            sector_exp[sector] = sector_exp.get(sector, 0.0) + float(p["market_value"])
        sector_exp_pct = {s: round(v / total_value * 100, 2) if total_value > 0 else 0.0 for s, v in sector_exp.items()}

        # Correlation and beta from returns
        correlation_risk = self._correlation_risk(returns_df, positions)
        market_beta = self._market_beta(returns_df, nifty_returns, positions)

        # Composite risk score (0–100, lower = safer)
        score, warnings = self._risk_score(
            exposure_pct, correlation_risk, market_beta, sector_exp_pct, unrealized_pnl_pct
        )
        risk_level = "LOW" if score < 30 else "MEDIUM" if score < 55 else "HIGH" if score < 75 else "CRITICAL"

        self.logger.info("Portfolio risk analyzed", extra={"risk_score": score, "risk_level": risk_level})
        return PortfolioRiskReport(
            total_value=total_value,
            cash=cash,
            used_capital=used_capital,
            portfolio_exposure_pct=round(exposure_pct, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            sector_exposure=sector_exp,
            sector_exposure_pct=sector_exp_pct,
            correlation_risk=round(correlation_risk, 4),
            market_beta=round(market_beta, 4),
            risk_score=round(score, 2),
            risk_level=risk_level,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_risk(returns_df: pd.DataFrame | None, positions: list[dict]) -> float:
        if returns_df is None or returns_df.empty or len(positions) < 2:
            return 0.0
        syms = [p["symbol"] for p in positions if p["symbol"] in returns_df.columns]
        if len(syms) < 2:
            return 0.0
        corr = returns_df[syms].corr()
        # Average of upper triangle (excluding diagonal)
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        values = upper.stack().values
        return float(np.mean(np.abs(values))) if len(values) > 0 else 0.0

    @staticmethod
    def _market_beta(
        returns_df: pd.DataFrame | None,
        nifty_returns: pd.Series | None,
        positions: list[dict],
    ) -> float:
        if returns_df is None or nifty_returns is None or not positions:
            return 1.0   # assume market beta of 1 when no data
        betas = []
        for p in positions:
            sym = p["symbol"]
            if sym not in returns_df.columns:
                continue
            aligned = pd.concat([returns_df[sym], nifty_returns], axis=1).dropna()
            if len(aligned) < 10:
                continue
            cov = aligned.cov().iloc[0, 1]
            var = aligned.iloc[:, 1].var()
            if var > 0:
                betas.append(cov / var)
        return float(np.mean(betas)) if betas else 1.0

    @staticmethod
    def _risk_score(
        exposure_pct: float,
        correlation: float,
        beta: float,
        sector_pct: dict[str, float],
        unrealized_pnl_pct: float,
    ) -> tuple[float, list[str]]:
        warnings = []
        score = 0.0

        # Exposure contribution (0–30)
        score += min(30.0, exposure_pct * 0.375)
        if exposure_pct > 70:
            warnings.append(f"High portfolio exposure: {exposure_pct:.1f}%")

        # Correlation contribution (0–25)
        score += correlation * 25.0
        if correlation > 0.7:
            warnings.append(f"High position correlation: {correlation:.2f}")

        # Beta contribution (0–20)
        score += min(20.0, abs(beta) * 10.0)
        if abs(beta) > 1.5:
            warnings.append(f"High market beta: {beta:.2f}")

        # Sector concentration (0–15)
        max_sector = max(sector_pct.values()) if sector_pct else 0
        score += min(15.0, max_sector * 0.375)
        if max_sector > 35:
            warnings.append(f"High sector concentration: {max_sector:.1f}%")

        # Unrealized loss penalty (0–10)
        if unrealized_pnl_pct < -5:
            score += min(10.0, abs(unrealized_pnl_pct) * 0.5)
            warnings.append(f"Significant unrealized loss: {unrealized_pnl_pct:.1f}%")

        return min(100.0, score), warnings
