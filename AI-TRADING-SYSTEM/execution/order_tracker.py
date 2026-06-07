from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrackedStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class TrackedOrder:
    order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: int
    requested_price: float | None
    filled_price: float | None
    status: TrackedStatus
    created_at: datetime
    updated_at: datetime
    broker_name: str
    error: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "requested_price": self.requested_price,
            "filled_price": self.filled_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "broker_name": self.broker_name,
            "error": self.error,
        }


class OrderTracker:
    """
    Maintains a complete audit log of every order submitted through the
    ExecutionEngine, across broker restarts and sessions.
    """

    def __init__(self) -> None:
        self._orders: dict[str, TrackedOrder] = {}
        self.logger = logging.getLogger(__name__)

    def track(
        self,
        order_id: str,
        broker_order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        requested_price: float | None,
        filled_price: float | None,
        status: TrackedStatus,
        broker_name: str,
        error: str = "",
        tags: dict[str, Any] | None = None,
    ) -> TrackedOrder:
        now = datetime.now()
        order = TrackedOrder(
            order_id=order_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            requested_price=requested_price,
            filled_price=filled_price,
            status=status,
            created_at=now,
            updated_at=now,
            broker_name=broker_name,
            error=error,
            tags=tags or {},
        )
        self._orders[order_id] = order
        self.logger.info("Order tracked", extra={"order_id": order_id, "status": status.value})
        return order

    def update(self, order_id: str, status: TrackedStatus, filled_price: float | None = None, error: str = "") -> None:
        if order_id in self._orders:
            self._orders[order_id].status = status
            self._orders[order_id].updated_at = datetime.now()
            if filled_price is not None:
                self._orders[order_id].filled_price = filled_price
            if error:
                self._orders[order_id].error = error

    def get(self, order_id: str) -> TrackedOrder | None:
        return self._orders.get(order_id)

    def all(self) -> list[TrackedOrder]:
        return sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)

    def by_status(self, status: TrackedStatus) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.status == status]

    def by_symbol(self, symbol: str) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.symbol == symbol]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in TrackedStatus}
        for o in self._orders.values():
            counts[o.status.value] += 1
        counts["total"] = len(self._orders)
        return counts
