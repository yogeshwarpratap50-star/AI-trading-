from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from broker.base_broker import BaseBroker
from execution.order_manager import OrderManager
from execution.order_tracker import OrderTracker, TrackedStatus
from risk_management.risk_engine import RiskEngine


@dataclass
class ExecutionConfig:
    poll_interval_seconds: int = 60
    dry_run: bool = False          # log orders but never submit
    max_retries: int = 2           # retry failed orders up to N times


class ExecutionEngine:
    """
    Top-level execution orchestrator for Phase 6-8.

    Connects broker, risk engine, order manager, and order tracker.
    Operates in simulation mode by default (broker.SIMULATION_MODE is True
    for all adapters created via BrokerFactory).

    Usage (single tick)
    -------------------
    engine = ExecutionEngine(broker, risk_engine)
    engine.execute(signal, price)        # one trade decision

    Usage (polling loop)
    --------------------
    engine.run_loop(signal_fn, price_fn) # blocking — call in a thread
    engine.stop()
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_engine: RiskEngine | None = None,
        config: ExecutionConfig | None = None,
        tracker: OrderTracker | None = None,
    ) -> None:
        self.broker = broker
        self.risk = risk_engine
        self.config = config or ExecutionConfig()
        self.tracker = tracker or OrderTracker()
        self.order_manager = OrderManager(broker, self.tracker)
        self._running = False
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Single-tick execution
    # ------------------------------------------------------------------

    def execute(
        self,
        signal: dict[str, Any],
        price: float,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process one trade signal and return a result dict.

        Signal schema
        -------------
        {
            "action": "BUY" | "SELL" | "HOLD",
            "symbol": str,
            "quantity": int,           # optional; overrides risk sizing
            "confidence": float,       # 0-1, used by risk validator
        }
        """
        action = signal.get("action", "HOLD").upper()
        symbol = signal.get("symbol", "")
        confidence = float(signal.get("confidence", 1.0))

        if action == "HOLD" or not symbol:
            return {"status": "SKIPPED", "reason": "HOLD or no symbol"}

        # Risk sizing / validation
        quantity = signal.get("quantity")
        if quantity is None and self.risk is not None:
            quantity = self.risk.size_position(
                symbol=symbol,
                price=price,
                capital=self._portfolio_value(),
            )

        if not quantity or quantity <= 0:
            return {"status": "SKIPPED", "reason": "Zero quantity from risk sizer"}

        if self.risk is not None:
            result = self.risk.validate_trade(
                signal=action,
                symbol=symbol,
                price=price,
                quantity=int(quantity),
                ai_confidence=confidence * 100,   # risk engine expects 0-100
                portfolio_cash=self._cash(),
                portfolio_total_value=self._portfolio_value(),
            )
            if not result.approved:
                self.logger.info("Trade blocked by risk engine: %s", result.summary())
                return {"status": "BLOCKED", "reasons": result.reasons}

        if self.config.dry_run:
            self.logger.info("[DRY-RUN] %s %s x%d @ %.2f", action, symbol, quantity, price)
            return {"status": "DRY_RUN", "action": action, "symbol": symbol, "quantity": quantity, "price": price}

        # Submit order
        order_result = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                if action == "BUY":
                    order_result = self.order_manager.buy(symbol, int(quantity), price, tags=tags)
                else:
                    order_result = self.order_manager.sell(symbol, int(quantity), price, tags=tags)
                if order_result is not None:
                    break
            except Exception as exc:
                self.logger.warning("Order attempt %d failed: %s", attempt, exc)
                if attempt >= self.config.max_retries + 1:
                    return {"status": "ERROR", "error": str(exc)}

        if order_result is None:
            return {"status": "REJECTED", "reason": "Pre-flight checks failed"}

        return {
            "status": "SUBMITTED",
            "order_id": order_result.order_id,
            "broker_status": order_result.status,
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": order_result.price,
        }

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        signal_fn: Callable[[], dict[str, Any]],
        price_fn: Callable[[str], float],
    ) -> None:
        """
        Blocking poll loop. Call signal_fn() each tick to get a new signal,
        then price_fn(symbol) to get the current price.

        Stop by calling engine.stop() from another thread.
        """
        self._running = True
        self.logger.info("ExecutionEngine loop started (interval=%ds)", self.config.poll_interval_seconds)
        while self._running:
            try:
                signal = signal_fn()
                symbol = signal.get("symbol", "")
                price = price_fn(symbol) if symbol else 0.0
                if price and price > 0:
                    result = self.execute(signal, price)
                    self.logger.info("Tick result: %s", result)
            except Exception as exc:
                self.logger.error("Execution loop error: %s", exc, exc_info=True)
            time.sleep(self.config.poll_interval_seconds)
        self.logger.info("ExecutionEngine loop stopped")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        broker_balance = {}
        try:
            broker_balance = self.broker.get_balance()
        except Exception:
            pass

        return {
            "simulation_mode": bool(getattr(self.broker, "SIMULATION_MODE", True)),
            "broker_connected": self.broker.is_connected,
            "order_summary": self.tracker.summary(),
            "balance": broker_balance,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _portfolio_value(self) -> float:
        try:
            bal = self.broker.get_balance()
            return float(bal.get("total_value", bal.get("cash", 100_000.0)))
        except Exception:
            return 100_000.0

    def _cash(self) -> float:
        try:
            bal = self.broker.get_balance()
            return float(bal.get("cash", 100_000.0))
        except Exception:
            return 100_000.0
