"""
Discord alerter — Phase 10E.

Set DISCORD_WEBHOOK_URL env var to activate.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request


class DiscordAlerter:
    """Posts embed messages to a Discord channel via webhook. Falls back to logging if unconfigured."""

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self._enabled = bool(self.webhook_url)
        self.logger = logging.getLogger(__name__)

    def trade_executed(self, symbol: str, side: str, qty: int, price: float, pnl: float | None = None) -> None:
        color = 0x2ECC71 if side == "BUY" else 0xE74C3C
        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side, "inline": True},
            {"name": "Qty", "value": str(qty), "inline": True},
            {"name": "Price", "value": f"₹{price:,.2f}", "inline": True},
        ]
        if pnl is not None:
            fields.append({"name": "P&L", "value": f"₹{pnl:+.2f}", "inline": True})
        self._send_embed(title="Trade Executed", color=color, fields=fields)

    def risk_breach(self, reason: str, detail: str = "") -> None:
        self._send_embed(title="⚠️ Risk Breach", description=f"{reason}\n{detail}", color=0xF39C12)

    def system_failure(self, component: str, error: str) -> None:
        self._send_embed(title=f"🔥 System Failure: {component}", description=error, color=0xE74C3C)

    def daily_report(self, pnl: float, trades: int, win_rate: float) -> None:
        color = 0x2ECC71 if pnl >= 0 else 0xE74C3C
        self._send_embed(
            title="📊 Daily Report",
            color=color,
            fields=[
                {"name": "P&L", "value": f"₹{pnl:+,.2f}", "inline": True},
                {"name": "Trades", "value": str(trades), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            ],
        )

    def send(self, content: str) -> bool:
        if not self._enabled:
            self.logger.info("[Discord] %s", content)
            return True
        try:
            payload = json.dumps({"content": content}).encode()
            req = urllib.request.Request(
                self.webhook_url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status in (200, 204)
        except Exception as exc:
            self.logger.error("Discord send error: %s", exc)
            return False

    def _send_embed(
        self,
        title: str,
        description: str = "",
        color: int = 0x3498DB,
        fields: list | None = None,
    ) -> bool:
        if not self._enabled:
            self.logger.info("[Discord] %s — %s", title, description or (fields or []))
            return True
        embed: dict = {"title": title, "color": color}
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = fields
        try:
            payload = json.dumps({"embeds": [embed]}).encode()
            req = urllib.request.Request(
                self.webhook_url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status in (200, 204)
        except Exception as exc:
            self.logger.error("Discord embed error: %s", exc)
            return False
