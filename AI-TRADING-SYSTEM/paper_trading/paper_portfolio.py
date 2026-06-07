from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass
class PaperPosition:
    symbol: str
    quantity: int
    avg_buy_price: float
    entry_date: datetime
    last_price: float = 0.0

    def market_value(self) -> float:
        return self.quantity * (self.last_price or self.avg_buy_price)

    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_buy_price) * self.quantity

    def unrealized_pnl_pct(self) -> float:
        cost = self.avg_buy_price * self.quantity
        return (self.unrealized_pnl() / cost * 100) if cost > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_buy_price": round(self.avg_buy_price, 4),
            "entry_date": self.entry_date.isoformat(),
            "last_price": round(self.last_price, 4),
            "market_value": round(self.market_value(), 4),
            "unrealized_pnl": round(self.unrealized_pnl(), 4),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct(), 4),
        }


@dataclass
class ClosedPosition:
    symbol: str
    quantity: int
    buy_price: float
    sell_price: float
    entry_date: datetime
    exit_date: datetime
    realized_pnl: float
    fees: float
    reason: str = ""

    def realized_pnl_pct(self) -> float:
        cost = self.buy_price * self.quantity
        return (self.realized_pnl / cost * 100) if cost > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": round(self.buy_price, 4),
            "sell_price": round(self.sell_price, 4),
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat(),
            "realized_pnl": round(self.realized_pnl, 4),
            "realized_pnl_pct": round(self.realized_pnl_pct(), 4),
            "fees": round(self.fees, 4),
            "reason": self.reason,
        }


class PaperPortfolio:
    """
    Tracks all paper-trading state: cash, open positions, closed positions,
    and daily equity snapshots.  Thread-safe for single-threaded polling loops.
    """

    def __init__(self, starting_capital: float = 100_000.0) -> None:
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: dict[str, PaperPosition] = {}
        self.closed_positions: list[ClosedPosition] = []
        self.equity_history: list[dict[str, Any]] = []
        self.daily_pnl: dict[str, float] = {}   # date-str → realized P&L that day
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        fees: float,
        timestamp: datetime,
    ) -> bool:
        """Deduct cost from cash and record open position. Returns False if insufficient funds."""
        total_cost = price * quantity + fees
        if total_cost > self.cash:
            self.logger.warning("Insufficient cash for position", extra={"symbol": symbol, "required": total_cost, "available": self.cash})
            return False
        self.cash -= total_cost
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            pos.avg_buy_price = (pos.avg_buy_price * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                avg_buy_price=price,
                entry_date=timestamp,
                last_price=price,
            )
        self.logger.info("Position opened", extra={"symbol": symbol, "qty": quantity, "price": price})
        return True

    def close_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        fees: float,
        timestamp: datetime,
        reason: str = "",
    ) -> float:
        """Credit proceeds to cash, record closed position. Returns realized P&L."""
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        actual_qty = min(quantity, pos.quantity)
        proceeds = price * actual_qty - fees
        self.cash += proceeds
        realized = (price - pos.avg_buy_price) * actual_qty - fees

        closed = ClosedPosition(
            symbol=symbol,
            quantity=actual_qty,
            buy_price=pos.avg_buy_price,
            sell_price=price,
            entry_date=pos.entry_date,
            exit_date=timestamp,
            realized_pnl=realized,
            fees=fees,
            reason=reason,
        )
        self.closed_positions.append(closed)

        # Track daily P&L
        day_key = timestamp.strftime("%Y-%m-%d")
        self.daily_pnl[day_key] = self.daily_pnl.get(day_key, 0.0) + realized

        pos.quantity -= actual_qty
        if pos.quantity <= 0:
            del self.positions[symbol]

        self.logger.info("Position closed", extra={"symbol": symbol, "pnl": realized})
        return realized

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update last_price for all held positions (mark-to-market)."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].last_price = price

    # ------------------------------------------------------------------
    # Equity and metrics
    # ------------------------------------------------------------------

    def total_value(self) -> float:
        position_value = sum(p.market_value() for p in self.positions.values())
        return self.cash + position_value

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl() for p in self.positions.values())

    def realized_pnl(self) -> float:
        return sum(c.realized_pnl for c in self.closed_positions)

    def total_pnl(self) -> float:
        return self.total_value() - self.starting_capital

    def total_return_pct(self) -> float:
        return (self.total_pnl() / self.starting_capital * 100) if self.starting_capital > 0 else 0.0

    def used_capital(self) -> float:
        return sum(p.avg_buy_price * p.quantity for p in self.positions.values())

    def today_pnl(self) -> float:
        return self.daily_pnl.get(date.today().isoformat(), 0.0)

    def snapshot(self, timestamp: datetime | None = None) -> dict[str, Any]:
        ts = timestamp or datetime.now()
        snap = {
            "timestamp": ts.isoformat(),
            "total_value": round(self.total_value(), 2),
            "cash": round(self.cash, 2),
            "used_capital": round(self.used_capital(), 2),
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "realized_pnl": round(self.realized_pnl(), 2),
            "total_pnl": round(self.total_pnl(), 2),
            "total_return_pct": round(self.total_return_pct(), 4),
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
        }
        self.equity_history.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = "paper_trading/state.json") -> None:
        state = {
            "starting_capital": self.starting_capital,
            "cash": self.cash,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "closed_positions": [c.to_dict() for c in self.closed_positions],
            "daily_pnl": self.daily_pnl,
            "equity_history": self.equity_history,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = "paper_trading/state.json") -> "PaperPortfolio":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved state at {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        portfolio = cls(starting_capital=state["starting_capital"])
        portfolio.cash = state["cash"]
        portfolio.daily_pnl = state.get("daily_pnl", {})
        portfolio.equity_history = state.get("equity_history", [])
        for sym, p in state.get("positions", {}).items():
            portfolio.positions[sym] = PaperPosition(
                symbol=sym,
                quantity=p["quantity"],
                avg_buy_price=p["avg_buy_price"],
                entry_date=datetime.fromisoformat(p["entry_date"]),
                last_price=p.get("last_price", p["avg_buy_price"]),
            )
        for c in state.get("closed_positions", []):
            portfolio.closed_positions.append(
                ClosedPosition(
                    symbol=c["symbol"],
                    quantity=c["quantity"],
                    buy_price=c["buy_price"],
                    sell_price=c["sell_price"],
                    entry_date=datetime.fromisoformat(c["entry_date"]),
                    exit_date=datetime.fromisoformat(c["exit_date"]),
                    realized_pnl=c["realized_pnl"],
                    fees=c["fees"],
                    reason=c.get("reason", ""),
                )
            )
        return portfolio
