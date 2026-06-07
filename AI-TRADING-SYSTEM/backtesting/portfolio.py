from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from backtesting.trade import OrderSide, OrderStatus, Trade


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_entry_price: float

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry_price) * self.quantity


class Portfolio:
    """Tracks cash, positions, and equity curve during a backtest."""

    def __init__(self, starting_capital: float) -> None:
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self._trade_counter = 0

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        side: OrderSide,
        date: datetime,
        price: float,
        quantity: int,
        fees: float = 0.0,
        slippage: float = 0.0,
        reason: str = "",
    ) -> Trade | None:
        """Open a new position. Returns None when insufficient capital."""
        cost = price * quantity + fees + slippage
        if side == OrderSide.BUY and cost > self.cash:
            return None

        self._trade_counter += 1
        trade = Trade(
            symbol=symbol,
            side=side,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            fees=fees,
            slippage=slippage,
            reason=reason,
            trade_id=self._trade_counter,
        )

        if side == OrderSide.BUY:
            self.cash -= cost
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.quantity + quantity
                pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + price * quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(symbol, quantity, price)

        self.trades.append(trade)
        return trade

    def close_trade(
        self,
        trade: Trade,
        date: datetime,
        price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Close an open trade and credit proceeds to cash."""
        trade.close(date, price, fees, slippage)
        proceeds = price * trade.quantity - fees
        self.cash += max(proceeds, 0.0)

        if trade.symbol in self.positions:
            pos = self.positions[trade.symbol]
            pos.quantity -= trade.quantity
            if pos.quantity <= 0:
                del self.positions[trade.symbol]

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def equity(self, prices: dict[str, float]) -> float:
        position_value = sum(
            pos.market_value(prices.get(pos.symbol, pos.avg_entry_price))
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def record_equity(self, date: datetime, prices: dict[str, float]) -> None:
        eq = self.equity(prices)
        unrealized = sum(
            pos.unrealized_pnl(prices.get(pos.symbol, pos.avg_entry_price))
            for pos in self.positions.values()
        )
        self.equity_curve.append(
            {
                "date": date,
                "equity": round(eq, 4),
                "cash": round(self.cash, 4),
                "unrealized_pnl": round(unrealized, 4),
                "open_positions": len(self.positions),
            }
        )

    def equity_series(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=["date", "equity", "cash", "unrealized_pnl", "open_positions"])
        return pd.DataFrame(self.equity_curve)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == OrderStatus.CLOSED]

    def open_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == OrderStatus.OPEN]

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.positions and self.positions[symbol].quantity > 0

    def total_invested(self) -> float:
        return sum(pos.avg_entry_price * pos.quantity for pos in self.positions.values())
