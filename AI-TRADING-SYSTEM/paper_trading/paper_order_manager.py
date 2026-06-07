from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: float | None
    stop_price: float | None
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_at: datetime | None = None
    fees: float = 0.0
    reject_reason: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def fill(self, price: float, fees: float, timestamp: datetime) -> None:
        self.filled_price = price
        self.filled_at = timestamp
        self.fees = fees
        self.status = OrderStatus.FILLED

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def reject(self, reason: str) -> None:
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "filled_price": round(self.filled_price, 4) if self.filled_price else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fees": round(self.fees, 4),
            "reject_reason": self.reject_reason,
        }


class PaperOrderManager:
    """
    Manages the full lifecycle of paper orders:
    creation → pending → filled / cancelled / rejected.
    """

    def __init__(self) -> None:
        self._orders: dict[str, PaperOrder] = {}
        self.logger = logging.getLogger(__name__)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tags: dict[str, Any] | None = None,
    ) -> PaperOrder:
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=datetime.now(),
            tags=tags or {},
        )
        self._orders[order.order_id] = order
        self.logger.info("Order created", extra={"order_id": order.order_id, "symbol": symbol, "side": side.value})
        return order

    def fill_order(self, order_id: str, price: float, fees: float) -> PaperOrder | None:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return None
        order.fill(price, fees, datetime.now())
        self.logger.info("Order filled", extra={"order_id": order_id, "price": price})
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.cancel()
            self.logger.info("Order cancelled", extra={"order_id": order_id})
            return True
        return False

    def reject_order(self, order_id: str, reason: str) -> None:
        order = self._orders.get(order_id)
        if order:
            order.reject(reason)
            self.logger.warning("Order rejected", extra={"order_id": order_id, "reason": reason})

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> PaperOrder | None:
        return self._orders.get(order_id)

    def pending_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def filled_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def cancelled_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == OrderStatus.CANCELLED]

    def rejected_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == OrderStatus.REJECTED]

    def all_orders(self, symbol: str | None = None) -> list[PaperOrder]:
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def has_open_order(self, symbol: str) -> bool:
        return any(o.symbol == symbol and o.status == OrderStatus.PENDING for o in self._orders.values())

    def cancel_all_pending(self) -> int:
        count = 0
        for order in self.pending_orders():
            order.cancel()
            count += 1
        return count
