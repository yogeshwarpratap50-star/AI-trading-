"""
Capital Allocator — Phase 12.

Given a total capital amount and a risk profile, automatically determines
how much to allocate to each stock opportunity.

Allocation method: inverse-volatility weighting, capped by profile limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autonomous.risk_profile import RiskProfileConfig
from autonomous.stock_scanner import ScanResult


@dataclass
class Allocation:
    symbol: str
    allocated_capital: float      # ₹ amount
    allocation_pct: float         # % of total deployable capital
    suggested_qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "allocated_capital": round(self.allocated_capital, 2),
            "allocation_pct": round(self.allocation_pct, 2),
            "suggested_qty": self.suggested_qty,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "score": round(self.score, 2),
        }


@dataclass
class AllocationPlan:
    total_capital: float
    deployable_capital: float     # after cash reserve
    cash_reserve: float
    allocations: list[Allocation]
    unallocated: float            # capital not deployed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_capital": round(self.total_capital, 2),
            "deployable_capital": round(self.deployable_capital, 2),
            "cash_reserve": round(self.cash_reserve, 2),
            "unallocated": round(self.unallocated, 2),
            "allocations": [a.to_dict() for a in self.allocations],
        }

    def summary(self) -> str:
        lines = [
            f"Total Capital:      ₹{self.total_capital:>12,.0f}",
            f"Cash Reserve:       ₹{self.cash_reserve:>12,.0f}",
            f"Deployable Capital: ₹{self.deployable_capital:>12,.0f}",
            f"Unallocated:        ₹{self.unallocated:>12,.0f}",
            f"Positions:          {len(self.allocations)}",
        ]
        for a in self.allocations:
            lines.append(
                f"  {a.symbol:<20} ₹{a.allocated_capital:>10,.0f}  "
                f"({a.allocation_pct:.1f}%)  qty={a.suggested_qty}"
            )
        return "\n".join(lines)


class CapitalAllocator:
    """
    Builds an AllocationPlan from scan results and a risk profile.

    Algorithm
    ---------
    1. Keep cash reserve per profile (e.g. 15%)
    2. Score-weight remaining capital across top candidates
    3. Cap each position at max_single_position_pct
    4. Compute quantity from allocated capital and entry price
    5. Compute stop-loss and take-profit using ATR × profile multipliers
    """

    def __init__(self, profile: RiskProfileConfig) -> None:
        self.profile = profile

    def allocate(
        self,
        total_capital: float,
        candidates: list[ScanResult],
    ) -> AllocationPlan:
        cash_reserve = total_capital * self.profile.cash_reserve_pct / 100
        deployable = total_capital - cash_reserve

        # Limit to max_positions
        top = candidates[: self.profile.max_positions]
        if not top:
            return AllocationPlan(total_capital, deployable, cash_reserve, [], deployable)

        # Inverse-volatility + score weighting
        weights = self._compute_weights(top)

        # Cap each position
        max_pct = self.profile.max_single_position_pct / 100
        weights = np.minimum(weights, max_pct)
        # Renormalise so they don't exceed 1.0
        total_w = weights.sum()
        if total_w > 1.0:
            weights /= total_w

        allocations: list[Allocation] = []
        for candidate, weight in zip(top, weights):
            amount = deployable * float(weight)
            if amount < candidate.current_price:   # can't afford even 1 share
                continue

            qty = max(1, int(amount / candidate.current_price))
            actual_amount = qty * candidate.current_price

            # Stops using ATR
            atr = candidate.atr if candidate.atr > 0 else candidate.current_price * 0.02
            stop = candidate.current_price - atr * self.profile.stop_loss_atr_multiplier
            target = candidate.current_price + atr * self.profile.take_profit_atr_multiplier
            stop = max(stop, candidate.current_price * 0.90)    # never more than 10% below
            target = min(target, candidate.current_price * 1.30) # never more than 30% above

            allocations.append(Allocation(
                symbol=candidate.symbol,
                allocated_capital=round(actual_amount, 2),
                allocation_pct=round(float(weight) * 100, 2),
                suggested_qty=qty,
                entry_price=candidate.current_price,
                stop_loss=round(stop, 2),
                take_profit=round(target, 2),
                score=candidate.score,
            ))

        deployed = sum(a.allocated_capital for a in allocations)
        unallocated = deployable - deployed

        return AllocationPlan(
            total_capital=total_capital,
            deployable_capital=deployable,
            cash_reserve=cash_reserve,
            allocations=allocations,
            unallocated=unallocated,
        )

    # ------------------------------------------------------------------

    def _compute_weights(self, candidates: list[ScanResult]) -> np.ndarray:
        """Score-based weights, normalised to sum to 1."""
        scores = np.array([c.score for c in candidates], dtype=float)
        scores = np.clip(scores, 1, None)
        weights = scores / scores.sum()
        return weights
