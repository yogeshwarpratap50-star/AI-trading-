"""
Trade Monitor — watches open positions and triggers exits based on:
  * Fixed stop loss / take profit
  * ATR trailing stop
  * Trailing stop (percentage)
  * Time-based / end-of-day exit
  * AI exit signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Callable

from execution.trade_executor import TradeExecutor, TradeRecord


# NSE market close time (IST)
_EOD_EXIT_TIME = time(15, 15)   # 15 minutes before close


@dataclass
class MonitorConfig:
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = False
    trailing_stop_pct: float = 2.0          # % below highest price
    enable_time_exit: bool = True           # exit N minutes before close
    time_exit_minutes: int = 15
    enable_eod_exit: bool = True            # force close all at EOD
    max_hold_hours: float = 6.0            # time-based exit
    ai_exit_threshold: float = 0.6         # confidence to accept AI SELL signal


class TradeMonitor:
    """
    Checks every open TradeRecord against exit conditions.

    Call `check_all(price_fn)` on each tick.
    `price_fn(symbol) -> float` supplies current prices.
    """

    def __init__(
        self,
        executor: TradeExecutor,
        config: MonitorConfig | None = None,
    ) -> None:
        self.executor = executor
        self.config = config or MonitorConfig()
        self._highest: dict[str, float] = {}   # trade_id → highest seen price
        self.logger = logging.getLogger(__name__)

    def check_all(
        self,
        price_fn: Callable[[str], float],
        ai_signal_fn: Callable[[str], tuple[str, float]] | None = None,
    ) -> list[dict]:
        """
        Evaluate every open position. Returns list of exit events.

        ai_signal_fn(symbol) -> ("BUY"|"SELL"|"HOLD", confidence 0-1)
        """
        exits = []
        for record in self.executor.journal.open_trades():
            price = price_fn(record.symbol)
            if not price or price <= 0:
                continue
            exit_reason = self._check_exit(record, price, ai_signal_fn)
            if exit_reason:
                pnl = self.executor.execute_sell(record, price, reason=exit_reason)
                exits.append({"trade_id": record.trade_id, "symbol": record.symbol,
                               "exit_price": price, "reason": exit_reason, "pnl": pnl})
                self.logger.info("Exit triggered", extra={
                    "symbol": record.symbol, "reason": exit_reason, "pnl": round(pnl, 2)
                })
        return exits

    def update_trailing(self, trade_id: str, price: float) -> None:
        """Update highest seen price for trailing stop calculation."""
        current = self._highest.get(trade_id, 0.0)
        if price > current:
            self._highest[trade_id] = price

    # ------------------------------------------------------------------
    # Exit condition checks (returns reason string or None)
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        record: TradeRecord,
        price: float,
        ai_signal_fn: Callable | None,
    ) -> str | None:
        self.update_trailing(record.trade_id, price)

        if record.side == "BUY":
            # Stop loss
            if self.config.enable_stop_loss and record.stop_loss and price <= record.stop_loss:
                return "stop_loss"
            # Take profit
            if self.config.enable_take_profit and record.take_profit and price >= record.take_profit:
                return "take_profit"
            # Trailing stop
            if self.config.enable_trailing_stop:
                high = self._highest.get(record.trade_id, price)
                trail = high * (1 - self.config.trailing_stop_pct / 100)
                if price <= trail:
                    return "trailing_stop"

        # Time-based exit
        if self.config.enable_time_exit:
            hold_hours = (datetime.now() - record.entry_time).total_seconds() / 3600
            if hold_hours >= self.config.max_hold_hours:
                return "time_exit"

        # End-of-day exit
        if self.config.enable_eod_exit:
            now = datetime.now().time()
            eod = (datetime.now() - timedelta(minutes=self.config.time_exit_minutes)).time()
            if now >= _EOD_EXIT_TIME:
                return "eod_exit"

        # AI exit signal
        if ai_signal_fn:
            ai_action, ai_conf = ai_signal_fn(record.symbol)
            if ai_action == "SELL" and ai_conf >= self.config.ai_exit_threshold:
                return f"ai_exit(conf={ai_conf:.2f})"

        return None
