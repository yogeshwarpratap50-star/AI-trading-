from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass(frozen=True)
class OrderConfig:
    """Configurable cost model for order simulation."""

    brokerage_pct: float = 0.0003   # 0.03% per leg — Zerodha-style flat brokerage
    tax_pct: float = 0.001          # STT + exchange fees on sell leg
    slippage_pct: float = 0.0005    # 0.05% market impact / slippage per leg


ExecutionResult = tuple[float, float, float]   # (execution_price, fees, slippage_cost)


class OrderSimulator:
    """Simulates order execution with realistic costs for Indian markets."""

    def __init__(self, config: OrderConfig | None = None) -> None:
        self.config = config or OrderConfig()

    # ------------------------------------------------------------------
    # Public execution methods
    # ------------------------------------------------------------------

    def execute_market(self, price: float, quantity: int, side: str) -> ExecutionResult:
        """Fill at price ± slippage, return (fill_price, fees, slippage_cost)."""
        slip = price * self.config.slippage_pct
        fill = price + slip if side == "BUY" else price - slip
        fill = max(fill, 0.01)
        fees = self._fees(fill, quantity, side)
        return fill, fees, slip * quantity

    def execute_limit(
        self,
        price: float,
        limit_price: float,
        quantity: int,
        side: str,
        bar_high: float,
        bar_low: float,
    ) -> ExecutionResult | None:
        """Fill only when the bar's range touches the limit price."""
        if side == "BUY" and bar_low <= limit_price:
            fill = min(limit_price, price)
            fees = self._fees(fill, quantity, side)
            return fill, fees, 0.0
        if side == "SELL" and bar_high >= limit_price:
            fill = max(limit_price, price)
            fees = self._fees(fill, quantity, side)
            return fill, fees, 0.0
        return None

    def execute_stop(
        self,
        stop_price: float,
        quantity: int,
        side: str,
        bar_high: float,
        bar_low: float,
    ) -> ExecutionResult | None:
        """Trigger stop order when bar crosses stop_price, fill at stop + slippage."""
        if side == "SELL" and bar_low <= stop_price:
            return self.execute_market(stop_price, quantity, "SELL")
        if side == "BUY" and bar_high >= stop_price:
            return self.execute_market(stop_price, quantity, "BUY")
        return None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_quantity(
        self,
        available_capital: float,
        price: float,
        risk_pct: float = 2.0,
    ) -> int:
        """Integer shares affordable within risk_pct of available_capital."""
        if price <= 0:
            return 0
        allocation = available_capital * (risk_pct / 100.0)
        return max(1, int(allocation / price))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fees(self, fill: float, quantity: int, side: str) -> float:
        """Brokerage on both legs, tax only on sell leg (Indian STT model)."""
        brokerage = fill * quantity * self.config.brokerage_pct
        tax = fill * quantity * self.config.tax_pct if side == "SELL" else 0.0
        return brokerage + tax
