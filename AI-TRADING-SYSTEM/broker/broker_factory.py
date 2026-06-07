from __future__ import annotations

import os
from typing import Any

from broker.base_broker import BaseBroker

# Supported broker names
BROKERS = ("zerodha", "dhan", "angelone", "groww", "paper")


class BrokerFactory:
    """
    Creates broker instances from a name string and optional config dict.

    Environment variable BROKER_NAME overrides the default.
    SIMULATION_MODE env var (true/false) controls simulation vs live.

    Usage
    -----
    broker = BrokerFactory.create("zerodha")          # simulation
    broker = BrokerFactory.create("dhan", live=True)  # live (credentials required)
    broker = BrokerFactory.create()                   # from env vars
    """

    @staticmethod
    def create(
        broker_name: str | None = None,
        config: dict[str, Any] | None = None,
        live: bool = False,
    ) -> BaseBroker:
        """
        Parameters
        ----------
        broker_name: One of 'zerodha', 'dhan', 'angelone', 'groww', 'paper'.
                     Falls back to BROKER_NAME env var, then 'paper'.
        config:      Optional dict of broker-specific credentials/settings.
        live:        If True, attempt live trading (simulation_mode=False).
                     Credentials must be present in config or env vars.
        """
        name = (broker_name or os.getenv("BROKER_NAME", "paper")).lower().strip()
        simulation = not live and os.getenv("SIMULATION_MODE", "true").lower() != "false"
        cfg = config or {}

        if name == "zerodha":
            from broker.zerodha.zerodha_adapter import ZerodhaAdapter
            return ZerodhaAdapter(
                api_key=cfg.get("api_key", os.getenv("ZERODHA_API_KEY", "")),
                access_token=cfg.get("access_token", os.getenv("ZERODHA_ACCESS_TOKEN", "")),
                simulation_mode=simulation,
            )

        if name == "dhan":
            from broker.dhan.dhan_adapter import DhanAdapter
            return DhanAdapter(
                client_id=cfg.get("client_id", os.getenv("DHAN_CLIENT_ID", "")),
                access_token=cfg.get("access_token", os.getenv("DHAN_ACCESS_TOKEN", "")),
                simulation_mode=simulation,
            )

        if name == "angelone":
            from broker.angelone.angel_adapter import AngelOneAdapter
            return AngelOneAdapter(
                api_key=cfg.get("api_key", os.getenv("ANGEL_API_KEY", "")),
                client_code=cfg.get("client_code", os.getenv("ANGEL_CLIENT_CODE", "")),
                password=cfg.get("password", os.getenv("ANGEL_PASSWORD", "")),
                totp_secret=cfg.get("totp_secret", os.getenv("ANGEL_TOTP_SECRET", "")),
                simulation_mode=simulation,
            )

        if name == "groww":
            from broker.groww.groww_adapter import GrowwAdapter
            return GrowwAdapter(simulation_mode=True)  # always simulation

        # Default: paper broker backed by PaperBroker
        from paper_trading.paper_broker import BrokerConfig, PaperBroker

        class _PaperBrokerAdapter(BaseBroker):
            """Thin adapter that wraps PaperBroker to satisfy BaseBroker interface."""

            SIMULATION_MODE = True

            def __init__(self) -> None:
                from broker.base_broker import OrderResult as OR
                paper_cfg = BrokerConfig(
                    starting_capital=float(cfg.get("starting_capital", os.getenv("STARTING_CAPITAL", "100000")))
                )
                self._pb = PaperBroker(paper_cfg)
                self._OR = OR

            def login(self) -> bool:
                return True

            def logout(self) -> None:
                pass

            @property
            def is_connected(self) -> bool:
                return True

            def get_price(self, symbol: str) -> float | None:
                return self._pb.get_price(symbol)

            def place_buy(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> Any:
                order = self._pb.place_buy(symbol, quantity, price or 0.0)
                return self._OR(order.order_id, order.status.value, symbol, "BUY", quantity, price, message="Paper")

            def place_sell(self, symbol: str, quantity: int, price: float | None = None, order_type: str = "MARKET") -> Any:
                order = self._pb.place_sell(symbol, quantity, price or 0.0)
                return self._OR(order.order_id, order.status.value, symbol, "SELL", quantity, price, message="Paper")

            def cancel_order(self, order_id: str) -> bool:
                return self._pb.cancel_order(order_id)

            def get_positions(self) -> list[dict]:
                return self._pb.get_positions()

            def get_orders(self) -> list[dict]:
                return self._pb.get_orders()

            def get_balance(self) -> dict[str, float]:
                return self._pb.get_balance()

        return _PaperBrokerAdapter()

    @staticmethod
    def available_brokers() -> list[str]:
        return list(BROKERS)
