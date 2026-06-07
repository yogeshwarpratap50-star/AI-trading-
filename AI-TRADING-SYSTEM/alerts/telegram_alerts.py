"""
Telegram alerter — Phase 10E.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars to activate.
Falls back to logging when not configured (no crash).
"""
from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request
from typing import Any


class TelegramAlerter:
    """
    Sends messages to a Telegram chat via Bot API.

    If credentials are missing, alerts are only logged (safe for CI / testing).
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)
        self.logger = logging.getLogger(__name__)
        if not self._enabled:
            self.logger.info("TelegramAlerter: no credentials — alert-only mode (logging)")

    # ------------------------------------------------------------------
    # Typed alert helpers
    # ------------------------------------------------------------------

    def trade_executed(self, symbol: str, side: str, qty: int, price: float, pnl: float | None = None) -> None:
        emoji = "🟢" if side == "BUY" else "🔴"
        pnl_str = f"  P&L: ₹{pnl:+.2f}" if pnl is not None else ""
        self.send(f"{emoji} *Trade Executed*\n{side} {symbol} ×{qty} @ ₹{price:,.2f}{pnl_str}")

    def risk_breach(self, reason: str, detail: str = "") -> None:
        self.send(f"⚠️ *Risk Breach*\n{reason}\n{detail}")

    def system_failure(self, component: str, error: str) -> None:
        self.send(f"🔥 *System Failure*\nComponent: {component}\nError: {error}")

    def model_failure(self, model: str, error: str) -> None:
        self.send(f"🤖 *Model Failure*\nModel: {model}\nError: {error}")

    def broker_failure(self, broker: str, error: str) -> None:
        self.send(f"🏦 *Broker Failure*\nBroker: {broker}\nError: {error}")

    def daily_report(self, pnl: float, trades: int, win_rate: float) -> None:
        self.send(
            f"📊 *Daily Report*\n"
            f"P&L: ₹{pnl:+,.2f}\n"
            f"Trades: {trades}\n"
            f"Win Rate: {win_rate:.1f}%"
        )

    # ------------------------------------------------------------------
    # Raw send
    # ------------------------------------------------------------------

    def send(self, message: str) -> bool:
        if not self._enabled:
            self.logger.info("[Telegram] %s", message.replace("*", ""))
            return True
        try:
            url = self.BASE_URL.format(token=self.token)
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }).encode()
            with urllib.request.urlopen(url, data=data, timeout=5) as resp:
                ok = resp.status == 200
            if not ok:
                self.logger.warning("Telegram send failed: status %s", resp.status)
            return ok
        except Exception as exc:
            self.logger.error("Telegram send error: %s", exc)
            return False
