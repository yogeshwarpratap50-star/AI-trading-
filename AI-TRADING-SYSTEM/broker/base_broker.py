from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class OrderResult:
    order_id: str
    status: str          # PLACED / REJECTED / SIMULATED
    symbol: str
    side: str            # BUY / SELL
    quantity: int
    price: float | None
    message: str = ""


class BaseBroker(ABC):
    """
    Abstract interface that every broker adapter must implement.

    All concrete adapters (Zerodha, Dhan, AngelOne, Groww, Paper) must
    satisfy this contract so that the ExecutionEngine can swap brokers
    without changing application logic.
    """

    SIMULATION_MODE: bool = True   # class-level default — always safe

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @abstractmethod
    def login(self) -> bool:
        """Authenticate and return True on success."""

    @abstractmethod
    def logout(self) -> None:
        """Close the session."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True when a valid session exists."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def get_price(self, symbol: str) -> float | None:
        """Return latest LTP for symbol, or None on failure."""

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        """Place a buy order. price=None means market order."""

    @abstractmethod
    def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> OrderResult:
        """Place a sell order. price=None means market order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Return list of current open positions."""

    @abstractmethod
    def get_orders(self) -> list[dict[str, Any]]:
        """Return list of today's orders."""

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Return {'cash': ..., 'total_value': ..., ...}."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def broker_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        mode = "SIMULATION" if self.SIMULATION_MODE else "LIVE"
        return f"{self.broker_name()}[{mode}]"
