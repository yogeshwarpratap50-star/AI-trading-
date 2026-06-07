"""
Trade Executor — converts a validated signal into a broker order and records it
in the trade journal. All execution defaults to SIMULATION mode.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from broker.base_broker import BaseBroker, OrderResult
from execution.order_tracker import OrderTracker, TrackedStatus
from risk_management.risk_engine import RiskEngine


@dataclass
class TradeRecord:
    """Immutable record written to the trade journal after each fill."""
    trade_id: str
    symbol: str
    side: str                   # BUY | SELL
    quantity: int
    entry_price: float
    exit_price: float | None    # None while position open
    entry_time: datetime
    exit_time: datetime | None
    stop_loss: float | None
    take_profit: float | None
    reason: str                 # signal source or exit reason
    ai_confidence: float        # 0-100
    risk_score: float           # 0-100
    realized_pnl: float = 0.0
    duration_seconds: float = 0.0
    broker_order_id: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def close(self, exit_price: float, reason: str = "") -> None:
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        mult = 1 if self.side == "BUY" else -1
        self.realized_pnl = mult * (exit_price - self.entry_price) * self.quantity
        self.duration_seconds = (self.exit_time - self.entry_time).total_seconds()
        if reason:
            self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "stop_loss": round(self.stop_loss, 4) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 4) if self.take_profit else None,
            "reason": self.reason,
            "ai_confidence": round(self.ai_confidence, 2),
            "risk_score": round(self.risk_score, 2),
            "realized_pnl": round(self.realized_pnl, 4),
            "duration_seconds": round(self.duration_seconds, 1),
            "broker_order_id": self.broker_order_id,
        }


class TradeJournal:
    """
    In-memory trade journal with CSV persistence.

    Appends one row per closed trade so the file grows incrementally.
    """

    def __init__(self, path: str | Path = "reports/trade_journal.csv") -> None:
        self._path = Path(path)
        self._trades: list[TradeRecord] = []
        self.logger = logging.getLogger(__name__)

    def add(self, record: TradeRecord) -> None:
        self._trades.append(record)

    def close_trade(self, trade_id: str, exit_price: float, reason: str = "") -> TradeRecord | None:
        record = self.get(trade_id)
        if record is None:
            return None
        record.close(exit_price, reason)
        self._append_to_csv(record)
        return record

    def get(self, trade_id: str) -> TradeRecord | None:
        return next((t for t in self._trades if t.trade_id == trade_id), None)

    def open_trades(self) -> list[TradeRecord]:
        return [t for t in self._trades if t.exit_price is None]

    def closed_trades(self) -> list[TradeRecord]:
        return [t for t in self._trades if t.exit_price is not None]

    def all_as_df(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self._trades])

    def today_pnl(self) -> float:
        today = datetime.now().date()
        return sum(
            t.realized_pnl for t in self.closed_trades()
            if t.exit_time and t.exit_time.date() == today
        )

    def _append_to_csv(self, record: TradeRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([record.to_dict()])
        write_header = not self._path.exists()
        row.to_csv(self._path, mode="a", header=write_header, index=False)


class TradeExecutor:
    """
    Converts a validated signal + risk params into a broker order and
    creates a TradeRecord in the journal.

    All execution is simulation-safe — broker.SIMULATION_MODE controls
    whether real orders are placed.
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_engine: RiskEngine,
        journal: TradeJournal | None = None,
        tracker: OrderTracker | None = None,
    ) -> None:
        self.broker = broker
        self.risk = risk_engine
        self.journal = journal or TradeJournal()
        self.tracker = tracker or OrderTracker()
        self.logger = logging.getLogger(__name__)

    def execute_buy(
        self,
        symbol: str,
        price: float,
        ai_confidence: float = 0.0,
        risk_score: float = 0.0,
        reason: str = "signal",
        atr: float | None = None,
        tags: dict[str, Any] | None = None,
    ) -> TradeRecord | None:
        """Place a BUY order, compute stops, record in journal."""
        capital = self._portfolio_value()
        qty = self.risk.size_position(symbol, price, capital=capital, atr=atr)
        if qty <= 0:
            self.logger.info("TradeExecutor: zero quantity — skipping BUY", extra={"symbol": symbol})
            return None

        stop = self.risk.stop_loss(price, atr=atr, side="BUY")
        target = self.risk.take_profit(price, atr=atr, side="BUY")

        result: OrderResult = self.broker.place_buy(symbol, qty, price)
        if result.status in ("REJECTED", "ERROR"):
            self.logger.warning("Buy order rejected by broker", extra={"symbol": symbol, "msg": result.message})
            return None

        record = TradeRecord(
            trade_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            side="BUY",
            quantity=qty,
            entry_price=result.price or price,
            exit_price=None,
            entry_time=datetime.now(),
            exit_time=None,
            stop_loss=stop,
            take_profit=target,
            reason=reason,
            ai_confidence=ai_confidence,
            risk_score=risk_score,
            broker_order_id=result.order_id,
            tags=tags or {},
        )
        self.journal.add(record)
        self.tracker.track(
            order_id=str(uuid.uuid4()),
            broker_order_id=result.order_id,
            symbol=symbol,
            side="BUY",
            quantity=qty,
            requested_price=price,
            filled_price=result.price,
            status=TrackedStatus.FILLED,
            broker_name=self.broker.broker_name(),
        )
        return record

    def execute_sell(
        self,
        trade_record: TradeRecord,
        price: float,
        reason: str = "signal",
    ) -> float:
        """Close an open trade. Returns realized P&L."""
        result: OrderResult = self.broker.place_sell(
            trade_record.symbol, trade_record.quantity, price
        )
        exit_price = result.price or price
        self.journal.close_trade(trade_record.trade_id, exit_price, reason)
        self.tracker.track(
            order_id=str(uuid.uuid4()),
            broker_order_id=result.order_id,
            symbol=trade_record.symbol,
            side="SELL",
            quantity=trade_record.quantity,
            requested_price=price,
            filled_price=exit_price,
            status=TrackedStatus.FILLED,
            broker_name=self.broker.broker_name(),
        )
        return trade_record.realized_pnl

    def _portfolio_value(self) -> float:
        try:
            return float(self.broker.get_balance().get("total_value", 100_000.0))
        except Exception:
            return 100_000.0
