from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from paper_trading.paper_broker import BrokerConfig, PaperBroker
from paper_trading.paper_order_manager import OrderType


@dataclass
class ExecutionConfig:
    """Controls for the paper execution loop."""
    poll_interval_seconds: float = 60.0   # how often to poll for new signals
    daily_loss_limit_pct: float = 3.0
    max_open_positions: int = 5
    risk_per_trade_pct: float = 2.0


SignalFn = Callable[[str], str]   # symbol → "BUY" | "SELL" | "HOLD"
PriceFn = Callable[[str], float | None]   # symbol → current price or None


class PaperExecutionEngine:
    """
    Drives the paper trading loop.

    Usage
    -----
    engine = PaperExecutionEngine(symbols=["RELIANCE.NS", "TCS.NS"])
    engine.run_once(signal_fn=my_signal_fn, price_fn=my_price_fn)
    """

    def __init__(
        self,
        symbols: list[str],
        broker_config: BrokerConfig | None = None,
        exec_config: ExecutionConfig | None = None,
    ) -> None:
        self.symbols = symbols
        self.broker = PaperBroker(broker_config)
        self.config = exec_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)
        self._stop_requested = False

    def run_once(self, signal_fn: SignalFn, price_fn: PriceFn) -> dict[str, Any]:
        """
        Process one round of signals for all symbols.
        Suitable for calling from a scheduler or manually.
        Returns a summary dict of actions taken.
        """
        results: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "actions": []}

        # 1. Refresh prices
        prices = {}
        for sym in self.symbols:
            price = price_fn(sym)
            if price:
                prices[sym] = price
        self.broker.update_prices(prices)

        # 2. Check daily loss limit before doing anything
        if not self.broker.check_daily_loss_limit(self.config.daily_loss_limit_pct):
            self.logger.warning("Daily loss limit breached — skipping all signals this cycle")
            results["daily_loss_limit_breached"] = True
            return results

        # 3. Process each symbol
        for sym in self.symbols:
            price = prices.get(sym)
            if price is None:
                self.logger.warning("No price available", extra={"symbol": sym})
                continue

            signal = signal_fn(sym)
            action = self._execute_signal(sym, signal, price)
            if action:
                results["actions"].append(action)

        # 4. Snapshot
        results["portfolio"] = self.broker.snapshot()
        self.logger.info("Execution cycle complete", extra={"actions": len(results["actions"])})
        return results

    def run_loop(self, signal_fn: SignalFn, price_fn: PriceFn, max_cycles: int | None = None) -> None:
        """
        Blocking polling loop. Call stop() from another thread to exit cleanly.
        Intended for live paper-trading daemons.
        """
        cycle = 0
        self.logger.info("Paper execution loop started", extra={"symbols": self.symbols})
        while not self._stop_requested:
            if max_cycles is not None and cycle >= max_cycles:
                break
            self.run_once(signal_fn, price_fn)
            cycle += 1
            if not self._stop_requested:
                time.sleep(self.config.poll_interval_seconds)
        self.logger.info("Paper execution loop stopped", extra={"cycles": cycle})

    def stop(self) -> None:
        self._stop_requested = True

    def get_summary(self) -> dict[str, Any]:
        return {
            "balance": self.broker.get_balance(),
            "positions": self.broker.get_positions(),
            "orders": self.broker.get_orders(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_signal(self, symbol: str, signal: str, price: float) -> dict[str, Any] | None:
        portfolio = self.broker.portfolio
        has_position = symbol in portfolio.positions
        open_count = len(portfolio.positions)

        if signal == "BUY" and not has_position and open_count < self.config.max_open_positions:
            qty = self._position_size(price)
            if qty > 0:
                order = self.broker.place_buy(symbol, qty, price, tags={"signal": signal})
                return {"symbol": symbol, "action": "BUY", "qty": qty, "price": price, "order_id": order.order_id, "status": order.status.value}

        elif signal == "SELL" and has_position:
            pos = portfolio.positions[symbol]
            order = self.broker.place_sell(symbol, pos.quantity, price, reason="SELL signal", tags={"signal": signal})
            return {"symbol": symbol, "action": "SELL", "qty": pos.quantity, "price": price, "order_id": order.order_id, "status": order.status.value}

        return None

    def _position_size(self, price: float) -> int:
        capital = self.broker.portfolio.cash
        allocation = capital * (self.config.risk_per_trade_pct / 100.0)
        return max(1, int(allocation / price)) if price > 0 else 0
