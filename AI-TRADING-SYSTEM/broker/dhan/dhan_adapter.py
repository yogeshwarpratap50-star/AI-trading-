from __future__ import annotations

"""
Dhan broker adapter.

Simulation mode by default — no real orders placed.
Live mode requires the dhanhq SDK:  pip install dhanhq

Dhan API documentation: https://dhanhq.co/docs/v2/
"""

import logging
import uuid
from typing import Any

from broker.base_broker import BaseBroker, OrderResult


class DhanAdapter(BaseBroker):
    """
    Dhan HQ adapter.

    Simulation mode: all methods return plausible mock data.
    Live mode: delegates to dhanhq.Dhan (SDK must be installed and
    CLIENT_ID + ACCESS_TOKEN must be provided).
    """

    SIMULATION_MODE: bool = True

    def __init__(
        self,
        client_id: str = "",
        access_token: str = "",
        simulation_mode: bool = True,
    ) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self.SIMULATION_MODE = simulation_mode
        self._connected = False
        self._mock_prices: dict[str, float] = {}
        self._mock_orders: list[dict] = []
        self._dhan: Any = None
        self.logger = logging.getLogger(__name__)

    def login(self) -> bool:
        if self.SIMULATION_MODE:
            self._connected = True
            self.logger.info("Dhan adapter: SIMULATION login successful")
            return True
        try:
            from dhanhq import Dhan
            self._dhan = Dhan(self.client_id, self.access_token)
            self._connected = True
            self.logger.info("Dhan: live login successful")
            return True
        except Exception as exc:
            self.logger.error("Dhan login failed", extra={"error": str(exc)})
            return False

    def logout(self) -> None:
        self._connected = False
        self.logger.info("Dhan: logged out")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_price(self, symbol: str) -> float | None:
        if self.SIMULATION_MODE:
            return self._mock_prices.get(symbol)
        try:
            resp = self._dhan.get_ltp_data(
                security_id=symbol, exchange_segment="NSE_EQ", instrument_type="EQUITY"
            )
            return float(resp["data"]["last_price"])
        except Exception:
            return None

    def set_mock_price(self, symbol: str, price: float) -> None:
        self._mock_prices[symbol] = price

    def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("BUY", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            resp = self._dhan.place_order(
                security_id=symbol,
                exchange_segment="NSE_EQ",
                transaction_type="BUY",
                quantity=quantity,
                order_type=order_type,
                product_type="CNC",
                price=price or 0,
            )
            return OrderResult(str(resp.get("orderId", "")), "PLACED", symbol, "BUY", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "BUY", quantity, price, message=str(exc))

    def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("SELL", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            resp = self._dhan.place_order(
                security_id=symbol,
                exchange_segment="NSE_EQ",
                transaction_type="SELL",
                quantity=quantity,
                order_type=order_type,
                product_type="CNC",
                price=price or 0,
            )
            return OrderResult(str(resp.get("orderId", "")), "PLACED", symbol, "SELL", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "SELL", quantity, price, message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        if self.SIMULATION_MODE:
            return True
        try:
            self._dhan.cancel_order(order_id)
            return True
        except Exception:
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return []
        try:
            return self._dhan.get_positions().get("data", [])
        except Exception:
            return []

    def get_orders(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return self._mock_orders
        try:
            return self._dhan.get_order_list().get("data", [])
        except Exception:
            return []

    def get_balance(self) -> dict[str, float]:
        if self.SIMULATION_MODE:
            return {"cash": 100_000.0, "total_value": 100_000.0, "used_capital": 0.0}
        try:
            resp = self._dhan.get_fund_limits()
            cash = float(resp["data"]["availabelBalance"])
            return {"cash": cash, "total_value": cash, "used_capital": 0.0}
        except Exception:
            return {"cash": 0.0, "total_value": 0.0, "used_capital": 0.0}

    def _mock_order(self, side: str, symbol: str, quantity: int, price: float) -> OrderResult:
        order_id = str(uuid.uuid4())[:8]
        self._mock_orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": quantity, "price": price})
        self.logger.info("Dhan SIM order", extra={"side": side, "symbol": symbol, "qty": quantity})
        return OrderResult(order_id, "SIMULATED", symbol, side, quantity, price, message="Simulation mode")
