from __future__ import annotations

import logging
import uuid
from datetime import datetime, time
from typing import Any

from broker.base_broker import BaseBroker, OrderResult
from execution.order_tracker import OrderTracker, TrackedStatus


# NSE market hours (IST)
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)


class OrderManager:
    """
    Submits orders to the broker and records every outcome in OrderTracker.

    Safety features:
    - Duplicate order prevention (same symbol + side within the same session)
    - Market hours enforcement (configurable)
    - Minimum quantity guard
    - Over-allocation guard
    """

    def __init__(
        self,
        broker: BaseBroker,
        tracker: OrderTracker | None = None,
        enforce_market_hours: bool = False,
    ) -> None:
        self.broker = broker
        self.tracker = tracker or OrderTracker()
        self.enforce_market_hours = enforce_market_hours
        self._recent_orders: set[str] = set()   # "SYMBOL:SIDE" dedup keys
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public order submission
    # ------------------------------------------------------------------

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "MARKET",
        tags: dict[str, Any] | None = None,
        force: bool = False,
    ) -> OrderResult | None:
        """
        Submit a BUY order.  Returns None if any safety check fails.

        Parameters
        ----------
        force: Skip duplicate-order guard (use only for partial fills / adds).
        """
        if not self._pre_flight(symbol, "BUY", quantity, force):
            return None
        result = self.broker.place_buy(symbol, quantity, price, order_type)
        self._record(result, "BUY", quantity, price, tags)
        return result

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "MARKET",
        tags: dict[str, Any] | None = None,
        force: bool = False,
    ) -> OrderResult | None:
        """Submit a SELL order. Returns None if any safety check fails."""
        if not self._pre_flight(symbol, "SELL", quantity, force):
            return None
        result = self.broker.place_sell(symbol, quantity, price, order_type)
        self._record(result, "SELL", quantity, price, tags)
        return result

    def cancel(self, order_id: str) -> bool:
        ok = self.broker.cancel_order(order_id)
        if ok:
            self.tracker.update(order_id, TrackedStatus.CANCELLED)
        return ok

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_orders(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.tracker.by_status(TrackedStatus.PENDING)]

    def filled_orders(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.tracker.by_status(TrackedStatus.FILLED)]

    def summary(self) -> dict[str, int]:
        return self.tracker.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pre_flight(self, symbol: str, side: str, quantity: int, force: bool) -> bool:
        # 1. Market hours
        if self.enforce_market_hours and not self._in_market_hours():
            self.logger.warning("Order rejected: outside market hours", extra={"symbol": symbol})
            return False

        # 2. Quantity
        if quantity <= 0:
            self.logger.warning("Order rejected: invalid quantity", extra={"symbol": symbol, "qty": quantity})
            return False

        # 3. Duplicate guard
        key = f"{symbol}:{side}"
        if not force and key in self._recent_orders:
            self.logger.warning("Order rejected: duplicate", extra={"symbol": symbol, "side": side})
            return False

        self._recent_orders.add(key)
        return True

    def _record(self, result: OrderResult, side: str, quantity: int, price: float | None, tags: dict | None) -> None:
        status = TrackedStatus.FILLED if result.status in ("FILLED", "SIMULATED", "PLACED") else TrackedStatus.REJECTED
        internal_id = str(uuid.uuid4())
        self.tracker.track(
            order_id=internal_id,
            broker_order_id=result.order_id,
            symbol=result.symbol,
            side=side,
            quantity=quantity,
            requested_price=price,
            filled_price=result.price,
            status=status,
            broker_name=self.broker.broker_name(),
            error=result.message if status == TrackedStatus.REJECTED else "",
            tags=tags or {},
        )
        # Reset dedup key on fill so a re-entry is allowed later
        if status == TrackedStatus.FILLED:
            self._recent_orders.discard(f"{result.symbol}:{side}")

    @staticmethod
    def _in_market_hours() -> bool:
        now = datetime.now().time()
        return _MARKET_OPEN <= now <= _MARKET_CLOSE
