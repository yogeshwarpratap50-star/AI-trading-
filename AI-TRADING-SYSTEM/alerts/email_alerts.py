"""
Email alerter — Phase 10E.

Configure via env vars:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailAlerter:
    """Sends HTML or plain-text alerts via SMTP. Falls back to logging if unconfigured."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        recipient: str | None = None,
    ) -> None:
        self.host = host or os.getenv("SMTP_HOST", "")
        self.port = int(port or os.getenv("SMTP_PORT", "587"))
        self.user = user or os.getenv("SMTP_USER", "")
        self.password = password or os.getenv("SMTP_PASS", "")
        self.recipient = recipient or os.getenv("ALERT_EMAIL_TO", "")
        self._enabled = bool(self.host and self.user and self.recipient)
        self.logger = logging.getLogger(__name__)

    def trade_executed(self, symbol: str, side: str, qty: int, price: float, pnl: float | None = None) -> None:
        pnl_str = f"P&L: ₹{pnl:+.2f}" if pnl is not None else ""
        self.send(
            subject=f"Trade: {side} {symbol}",
            body=f"{side} {symbol} ×{qty} @ ₹{price:,.2f}\n{pnl_str}",
        )

    def risk_breach(self, reason: str, detail: str = "") -> None:
        self.send(subject="⚠ Risk Breach", body=f"{reason}\n{detail}")

    def system_failure(self, component: str, error: str) -> None:
        self.send(subject=f"🔥 System Failure: {component}", body=error)

    def daily_report(self, pnl: float, trades: int, win_rate: float) -> None:
        self.send(
            subject="Daily Trading Report",
            body=f"P&L: ₹{pnl:+,.2f}\nTrades: {trades}\nWin Rate: {win_rate:.1f}%",
        )

    def send(self, subject: str, body: str, html: bool = False) -> bool:
        if not self._enabled:
            self.logger.info("[Email] %s: %s", subject, body[:80])
            return True
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.user
            msg["To"] = self.recipient
            mime_type = "html" if html else "plain"
            msg.attach(MIMEText(body, mime_type, "utf-8"))
            with smtplib.SMTP(self.host, self.port, timeout=10) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, [self.recipient], msg.as_string())
            return True
        except Exception as exc:
            self.logger.error("Email send error: %s", exc)
            return False
