from __future__ import annotations

"""
Angel One (SmartAPI) adapter.

Simulation mode by default — no real orders placed.
Live mode requires:  pip install smartapi-python

SmartAPI documentation: https://smartapi.angelbroking.com/docs
"""

import logging
import uuid
from typing import Any

from broker.base_broker import BaseBroker, OrderResult


class AngelOneAdapter(BaseBroker):
    """
    Angel One SmartAPI adapter.

    Simulation mode: returns plausible mock responses with no network calls.
    Live mode: delegates to SmartConnect (pip install smartapi-python).
    """

    SIMULATION_MODE: bool = True

    def __init__(
        self,
        api_key: str = "",
        client_code: str = "",
        password: str = "",
        totp_secret: str = "",
        simulation_mode: bool = True,
    ) -> None:
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.SIMULATION_MODE = simulation_mode
        self._connected = False
        self._mock_prices: dict[str, float] = {}
        self._mock_orders: list[dict] = []
        self._smart: Any = None
        self.logger = logging.getLogger(__name__)

    def login(self) -> bool:
        if self.SIMULATION_MODE:
            self._connected = True
            self.logger.info("AngelOne adapter: SIMULATION login successful")
            return True
        try:
            import pyotp
            from SmartApi import SmartConnect
            self._smart = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self._smart.generateSession(self.client_code, self.password, totp)
            if data["status"]:
                self._connected = True
                self.logger.info("AngelOne: live login successful")
                return True
            return False
        except Exception as exc:
            self.logger.error("AngelOne login failed", extra={"error": str(exc)})
            return False

    def logout(self) -> None:
        if not self.SIMULATION_MODE and self._smart:
            try:
                self._smart.terminateSession(self.client_code)
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_price(self, symbol: str) -> float | None:
        if self.SIMULATION_MODE:
            return self._mock_prices.get(symbol)
        try:
            ltp_data = self._smart.ltpData("NSE", symbol, "")
            return float(ltp_data["data"]["ltp"])
        except Exception:
            return None

    def set_mock_price(self, symbol: str, price: float) -> None:
        self._mock_prices[symbol] = price

    def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("BUY", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": "",
                "transactiontype": "BUY",
                "exchange": "NSE",
                "ordertype": order_type,
                "producttype": "DELIVERY",
                "duration": "DAY",
                "price": str(price or 0),
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(quantity),
            }
            resp = self._smart.placeOrder(params)
            return OrderResult(str(resp.get("orderid", "")), "PLACED", symbol, "BUY", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "BUY", quantity, price, message=str(exc))

    def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("SELL", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": "",
                "transactiontype": "SELL",
                "exchange": "NSE",
                "ordertype": order_type,
                "producttype": "DELIVERY",
                "duration": "DAY",
                "price": str(price or 0),
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(quantity),
            }
            resp = self._smart.placeOrder(params)
            return OrderResult(str(resp.get("orderid", "")), "PLACED", symbol, "SELL", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "SELL", quantity, price, message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        if self.SIMULATION_MODE:
            return True
        try:
            self._smart.cancelOrder(order_id, "NORMAL")
            return True
        except Exception:
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return []
        try:
            resp = self._smart.position()
            return resp.get("data", []) or []
        except Exception:
            return []

    def get_orders(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return self._mock_orders
        try:
            resp = self._smart.orderBook()
            return resp.get("data", []) or []
        except Exception:
            return []

    def get_balance(self) -> dict[str, float]:
        if self.SIMULATION_MODE:
            return {"cash": 100_000.0, "total_value": 100_000.0, "used_capital": 0.0}
        try:
            resp = self._smart.rmsLimit()
            cash = float(resp["data"]["net"])
            return {"cash": cash, "total_value": cash, "used_capital": 0.0}
        except Exception:
            return {"cash": 0.0, "total_value": 0.0, "used_capital": 0.0}

    def _mock_order(self, side: str, symbol: str, quantity: int, price: float) -> OrderResult:
        order_id = str(uuid.uuid4())[:8]
        self._mock_orders.append({"order_id": order_id, "symbol": symbol, "side": side, "qty": quantity, "price": price})
        self.logger.info("AngelOne SIM order", extra={"side": side, "symbol": symbol, "qty": quantity})
        return OrderResult(order_id, "SIMULATED", symbol, side, quantity, price, message="Simulation mode")
