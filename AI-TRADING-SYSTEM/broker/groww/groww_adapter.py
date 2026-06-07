from __future__ import annotations

"""
Groww broker adapter — placeholder implementation.

Groww does not currently publish an official public trading API for retail
algorithmic traders. This adapter is designed to be production-ready as soon
as Groww releases an official API.

Integration requirements (when Groww API becomes available):
  1. Obtain API credentials from Groww developer portal.
  2. Install the official Groww SDK (not yet released as of 2025).
  3. Replace the _live_* stub methods below with real SDK calls.
  4. All simulation behaviour is already implemented and tested.

References:
  - Groww Pro platform: https://pro.groww.in
  - Groww developer updates: https://groww.in/blog
"""

import logging
import uuid
from typing import Any

from broker.base_broker import BaseBroker, OrderResult

_GROWW_API_NOTE = (
    "Groww does not have a public trading API as of 2025. "
    "This adapter operates in SIMULATION_MODE only until an official API is available."
)


class GrowwAdapter(BaseBroker):
    """
    Groww adapter — simulation-only until an official Groww trading API is released.

    All live methods raise NotImplementedError with a clear explanation.
    Simulation mode is fully functional for paper-trading and testing workflows.
    """

    SIMULATION_MODE: bool = True   # immutable — Groww has no public API yet

    def __init__(self, simulation_mode: bool = True) -> None:
        if not simulation_mode:
            raise NotImplementedError(
                "GrowwAdapter: live trading is not supported yet. " + _GROWW_API_NOTE
            )
        self.SIMULATION_MODE = True
        self._connected = False
        self._mock_prices: dict[str, float] = {}
        self._mock_orders: list[dict] = []
        self.logger = logging.getLogger(__name__)

    def login(self) -> bool:
        self._connected = True
        self.logger.info("Groww adapter: SIMULATION login (no real API available)")
        return True

    def logout(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_price(self, symbol: str) -> float | None:
        return self._mock_prices.get(symbol)

    def set_mock_price(self, symbol: str, price: float) -> None:
        self._mock_prices[symbol] = price

    def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        return self._mock_order("BUY", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))

    def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        return self._mock_order("SELL", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))

    def cancel_order(self, order_id: str) -> bool:
        return True   # simulation: always succeeds

    def get_positions(self) -> list[dict[str, Any]]:
        return []

    def get_orders(self) -> list[dict[str, Any]]:
        return self._mock_orders

    def get_balance(self) -> dict[str, float]:
        return {"cash": 100_000.0, "total_value": 100_000.0, "used_capital": 0.0}

    def _mock_order(self, side: str, symbol: str, quantity: int, price: float) -> OrderResult:
        order_id = str(uuid.uuid4())[:8]
        self._mock_orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": quantity, "price": price})
        self.logger.info("Groww SIM order", extra={"side": side, "symbol": symbol, "qty": quantity})
        return OrderResult(
            order_id, "SIMULATED", symbol, side, quantity, price,
            message="Simulation mode — " + _GROWW_API_NOTE,
        )

    # ------------------------------------------------------------------
    # Live stubs — raise clearly when called without simulation
    # ------------------------------------------------------------------

    def _live_login(self) -> None:
        raise NotImplementedError(_GROWW_API_NOTE)

    def _live_place_order(self) -> None:
        raise NotImplementedError(_GROWW_API_NOTE)
