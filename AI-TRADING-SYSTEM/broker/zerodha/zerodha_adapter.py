from __future__ import annotations

"""
Zerodha / Kite Connect adapter.

In SIMULATION_MODE (default) every call is handled locally without any
real network request.  Set SIMULATION_MODE=False and supply real
credentials only when you intend to trade live.

Real integration requires:
    pip install kiteconnect
and a valid API key + access token from https://kite.trade/
"""

import logging
import uuid
from typing import Any

from broker.base_broker import BaseBroker, OrderResult


class ZerodhaAdapter(BaseBroker):
    """
    Kite Connect adapter for Zerodha.

    Simulation mode — all methods return plausible mock responses.
    Live mode — delegates to kiteconnect.KiteConnect (must be installed
    and credentials must be provided via environment variables).
    """

    SIMULATION_MODE: bool = True

    def __init__(
        self,
        api_key: str = "",
        access_token: str = "",
        simulation_mode: bool = True,
    ) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.SIMULATION_MODE = simulation_mode
        self._connected = False
        self._mock_prices: dict[str, float] = {}
        self._mock_orders: list[dict] = []
        self._mock_positions: list[dict] = []
        self._mock_cash = 100_000.0
        self.logger = logging.getLogger(__name__)
        self._kite: Any = None

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def login(self) -> bool:
        if self.SIMULATION_MODE:
            self._connected = True
            self.logger.info("Zerodha adapter: SIMULATION login successful")
            return True
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)
            self._connected = True
            self.logger.info("Zerodha: live login successful")
            return True
        except Exception as exc:
            self.logger.error("Zerodha login failed", extra={"error": str(exc)})
            return False

    def logout(self) -> None:
        if not self.SIMULATION_MODE and self._kite:
            try:
                self._kite.invalidate_access_token()
            except Exception:
                pass
        self._connected = False
        self.logger.info("Zerodha: logged out")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> float | None:
        if self.SIMULATION_MODE:
            return self._mock_prices.get(symbol)
        try:
            quote = self._kite.quote([symbol])
            return float(quote[symbol]["last_price"])
        except Exception:
            return None

    def set_mock_price(self, symbol: str, price: float) -> None:
        """Helper for tests and simulation."""
        self._mock_prices[symbol] = price

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("BUY", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            from kiteconnect import KiteConnect
            order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type="BUY",
                quantity=quantity,
                order_type=order_type,
                product="CNC",
                price=price,
                variety="regular",
            )
            return OrderResult(str(order_id), "PLACED", symbol, "BUY", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "BUY", quantity, price, message=str(exc))

    def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        if self.SIMULATION_MODE:
            return self._mock_order("SELL", symbol, quantity, price or self._mock_prices.get(symbol, 0.0))
        try:
            order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type="SELL",
                quantity=quantity,
                order_type=order_type,
                product="CNC",
                price=price,
                variety="regular",
            )
            return OrderResult(str(order_id), "PLACED", symbol, "SELL", quantity, price)
        except Exception as exc:
            return OrderResult("", "REJECTED", symbol, "SELL", quantity, price, message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        if self.SIMULATION_MODE:
            return True
        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return self._mock_positions
        try:
            return self._kite.positions().get("net", [])
        except Exception:
            return []

    def get_orders(self) -> list[dict[str, Any]]:
        if self.SIMULATION_MODE:
            return self._mock_orders
        try:
            return self._kite.orders()
        except Exception:
            return []

    def get_balance(self) -> dict[str, float]:
        if self.SIMULATION_MODE:
            return {"cash": self._mock_cash, "total_value": self._mock_cash, "used_capital": 0.0}
        try:
            margins = self._kite.margins()
            cash = float(margins["equity"]["available"]["live_balance"])
            return {"cash": cash, "total_value": cash, "used_capital": 0.0}
        except Exception:
            return {"cash": 0.0, "total_value": 0.0, "used_capital": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mock_order(self, side: str, symbol: str, quantity: int, price: float) -> OrderResult:
        order_id = str(uuid.uuid4())[:8]
        self._mock_orders.append({"order_id": order_id, "symbol": symbol, "side": side, "quantity": quantity, "price": price, "status": "SIMULATED"})
        self.logger.info("Zerodha SIM order", extra={"side": side, "symbol": symbol, "qty": quantity, "price": price})
        return OrderResult(order_id, "SIMULATED", symbol, side, quantity, price, message="Simulation mode — no real order placed")
