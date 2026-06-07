from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from paper_trading.paper_order_manager import OrderSide, OrderStatus, OrderType, PaperOrder, PaperOrderManager
from paper_trading.paper_portfolio import PaperPortfolio


@dataclass(frozen=True)
class BrokerConfig:
    """Cost model and limits for the paper broker."""
    brokerage_pct: float = 0.0003       # 0.03% per leg
    tax_pct: float = 0.001              # STT on sell
    slippage_pct: float = 0.0005        # Market impact
    starting_capital: float = 100_000.0


class PaperBroker:
    """
    Virtual broker that routes all orders to in-memory state.

    No real exchange calls are ever made from this class.
    All orders are simulated with configurable brokerage, tax, and slippage.
    """

    def __init__(self, config: BrokerConfig | None = None) -> None:
        self.config = config or BrokerConfig()
        self.portfolio = PaperPortfolio(self.config.starting_capital)
        self.order_manager = PaperOrderManager()
        self._trading_enabled = True
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        tags: dict[str, Any] | None = None,
    ) -> PaperOrder:
        order = self.order_manager.create_order(
            symbol, OrderSide.BUY, quantity, order_type, limit_price=limit_price, tags=tags or {}
        )
        if not self._trading_enabled:
            self.order_manager.reject_order(order.order_id, "Trading disabled — daily loss limit reached")
            return order
        fill_price = self._apply_slippage(price, "BUY")
        fees = self._calculate_fees(fill_price, quantity, "BUY")
        total_cost = fill_price * quantity + fees
        if total_cost > self.portfolio.cash:
            self.order_manager.reject_order(order.order_id, f"Insufficient cash: need {total_cost:.2f}, have {self.portfolio.cash:.2f}")
            return order
        self.order_manager.fill_order(order.order_id, fill_price, fees)
        self.portfolio.open_position(symbol, quantity, fill_price, fees, datetime.now())
        return order

    def place_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        reason: str = "",
        tags: dict[str, Any] | None = None,
    ) -> PaperOrder:
        order = self.order_manager.create_order(
            symbol, OrderSide.SELL, quantity, order_type, limit_price=limit_price, tags=tags or {}
        )
        if not self._trading_enabled:
            self.order_manager.reject_order(order.order_id, "Trading disabled")
            return order
        if symbol not in self.portfolio.positions:
            self.order_manager.reject_order(order.order_id, f"No open position in {symbol}")
            return order
        fill_price = self._apply_slippage(price, "SELL")
        fees = self._calculate_fees(fill_price, quantity, "SELL")
        self.order_manager.fill_order(order.order_id, fill_price, fees)
        self.portfolio.close_position(symbol, quantity, fill_price, fees, datetime.now(), reason=reason)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return self.order_manager.cancel_order(order_id)

    # ------------------------------------------------------------------
    # Account / position queries
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, float]:
        return {
            "cash": round(self.portfolio.cash, 2),
            "total_value": round(self.portfolio.total_value(), 2),
            "used_capital": round(self.portfolio.used_capital(), 2),
            "unrealized_pnl": round(self.portfolio.unrealized_pnl(), 2),
            "realized_pnl": round(self.portfolio.realized_pnl(), 2),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.portfolio.positions.values()]

    def get_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.order_manager.all_orders(symbol)]

    def get_price(self, symbol: str) -> float | None:
        pos = self.portfolio.positions.get(symbol)
        return pos.last_price if pos else None

    # ------------------------------------------------------------------
    # Risk controls
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        self.portfolio.update_prices(prices)

    def check_daily_loss_limit(self, limit_pct: float = 3.0) -> bool:
        """Disable trading if today's loss exceeds limit_pct of starting capital."""
        today_pnl = self.portfolio.today_pnl()
        limit = self.portfolio.starting_capital * (limit_pct / 100)
        if today_pnl < -abs(limit):
            self._trading_enabled = False
            self.logger.warning("Daily loss limit reached — trading disabled", extra={"today_pnl": today_pnl})
            return False
        self._trading_enabled = True
        return True

    def enable_trading(self, enabled: bool = True) -> None:
        self._trading_enabled = enabled

    def is_trading_enabled(self) -> bool:
        return self._trading_enabled

    def snapshot(self) -> dict[str, Any]:
        port_snap = self.portfolio.snapshot()
        return {
            **port_snap,
            "balance": {
                "cash": port_snap["cash"],
                "total_value": port_snap["total_value"],
                "unrealized_pnl": port_snap["unrealized_pnl"],
                "today_pnl": self.portfolio.today_pnl(),
            },
            "positions": [p.to_dict() for p in self.portfolio.positions.values()],
            "closed_positions": [c.to_dict() for c in self.portfolio.closed_positions],
            "trading_enabled": self._trading_enabled,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * self.config.slippage_pct
        return price + slip if side == "BUY" else max(price - slip, 0.01)

    def _calculate_fees(self, price: float, quantity: int, side: str) -> float:
        brokerage = price * quantity * self.config.brokerage_pct
        tax = price * quantity * self.config.tax_pct if side == "SELL" else 0.0
        return brokerage + tax
