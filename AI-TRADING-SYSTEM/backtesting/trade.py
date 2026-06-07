from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Trade:
    """Represents a single round-trip or one-legged trade."""

    symbol: str
    side: OrderSide
    entry_date: datetime
    entry_price: float
    quantity: int
    exit_date: datetime | None = None
    exit_price: float | None = None
    fees: float = 0.0
    slippage: float = 0.0
    pnl: float | None = None
    pnl_pct: float | None = None
    status: OrderStatus = field(default=OrderStatus.OPEN)
    trade_id: int = field(default=0)
    reason: str = ""

    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.fees += fees
        self.slippage += slippage
        self.status = OrderStatus.CLOSED
        if self.side == OrderSide.BUY:
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - exit_price) * self.quantity
        self.pnl = gross_pnl - self.fees
        cost_basis = self.entry_price * self.quantity
        self.pnl_pct = self.pnl / cost_basis if cost_basis > 0 else 0.0

    def duration_days(self) -> int | None:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_date": self.entry_date.isoformat() if isinstance(self.entry_date, datetime) else str(self.entry_date),
            "entry_price": round(self.entry_price, 4),
            "quantity": self.quantity,
            "exit_date": self.exit_date.isoformat() if isinstance(self.exit_date, datetime) else (str(self.exit_date) if self.exit_date else None),
            "exit_price": round(self.exit_price, 4) if self.exit_price is not None else None,
            "fees": round(self.fees, 4),
            "slippage": round(self.slippage, 4),
            "pnl": round(self.pnl, 4) if self.pnl is not None else None,
            "pnl_pct": round(self.pnl_pct * 100, 4) if self.pnl_pct is not None else None,
            "status": self.status.value,
            "duration_days": self.duration_days(),
            "reason": self.reason,
        }
