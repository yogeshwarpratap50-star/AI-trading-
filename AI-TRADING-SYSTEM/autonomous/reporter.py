"""
Autonomous Reporter — Phase 12.

Generates and sends daily / weekly / monthly reports via Telegram and Email.
Reports are generated automatically without user intervention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from autonomous.risk_profile import RiskProfileConfig


@dataclass
class ReportData:
    period: str                      # "daily" | "weekly" | "monthly"
    start_date: str
    end_date: str
    total_capital: float
    portfolio_value: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    best_trade: dict = field(default_factory=dict)
    worst_trade: dict = field(default_factory=dict)
    positions: list[dict] = field(default_factory=list)
    regime: str = "UNKNOWN"
    model_accuracy: float = 0.0
    safety_events: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        sign = "+" if self.total_pnl >= 0 else ""
        lines = [
            f"{'='*50}",
            f"  AI TRADING SYSTEM — {self.period.upper()} REPORT",
            f"{'='*50}",
            f"Period: {self.start_date} → {self.end_date}",
            f"",
            f"PORTFOLIO SUMMARY",
            f"  Capital:         ₹{self.total_capital:>12,.0f}",
            f"  Portfolio Value: ₹{self.portfolio_value:>12,.0f}",
            f"  Cash:            ₹{self.cash:>12,.0f}",
            f"  Realized P&L:    ₹{self.realized_pnl:>+12,.0f}",
            f"  Unrealized P&L:  ₹{self.unrealized_pnl:>+12,.0f}",
            f"  Total P&L:       ₹{self.total_pnl:>+12,.0f}  ({sign}{self.total_return_pct:.2f}%)",
            f"",
            f"TRADING STATS",
            f"  Trades:          {self.num_trades}",
            f"  Win Rate:        {self.win_rate:.1f}%",
            f"  Model Accuracy:  {self.model_accuracy:.1f}%",
            f"  Market Regime:   {self.regime}",
        ]
        if self.best_trade:
            lines.append(f"  Best Trade:      {self.best_trade.get('symbol','?')} "
                         f"₹{self.best_trade.get('pnl',0):+,.0f}")
        if self.worst_trade:
            lines.append(f"  Worst Trade:     {self.worst_trade.get('symbol','?')} "
                         f"₹{self.worst_trade.get('pnl',0):+,.0f}")
        if self.positions:
            lines.append(f"")
            lines.append(f"OPEN POSITIONS ({len(self.positions)})")
            for pos in self.positions[:5]:  # top 5
                lines.append(
                    f"  {pos.get('symbol','?'):<20} "
                    f"qty={pos.get('quantity',0)}  "
                    f"pnl=₹{pos.get('unrealized_pnl',0):+,.0f}"
                )
        if self.safety_events:
            lines.append(f"")
            lines.append(f"SAFETY EVENTS")
            for evt in self.safety_events:
                lines.append(f"  ! {evt}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def to_html(self) -> str:
        sign = "+" if self.total_pnl >= 0 else ""
        color = "green" if self.total_pnl >= 0 else "red"
        return f"""
<h2>AI Trading System — {self.period.capitalize()} Report</h2>
<p><b>Period:</b> {self.start_date} → {self.end_date}</p>
<h3>Portfolio Summary</h3>
<table border="1" cellpadding="4">
<tr><td>Capital</td><td>₹{self.total_capital:,.0f}</td></tr>
<tr><td>Portfolio Value</td><td>₹{self.portfolio_value:,.0f}</td></tr>
<tr><td>Cash</td><td>₹{self.cash:,.0f}</td></tr>
<tr><td>Total P&amp;L</td><td style="color:{color}">₹{self.total_pnl:+,.0f} ({sign}{self.total_return_pct:.2f}%)</td></tr>
</table>
<h3>Trading Stats</h3>
<table border="1" cellpadding="4">
<tr><td>Trades</td><td>{self.num_trades}</td></tr>
<tr><td>Win Rate</td><td>{self.win_rate:.1f}%</td></tr>
<tr><td>Model Accuracy</td><td>{self.model_accuracy:.1f}%</td></tr>
<tr><td>Market Regime</td><td>{self.regime}</td></tr>
</table>
"""


class AutonomousReporter:
    """
    Builds and sends periodic reports using available alert channels.

    Usage
    -----
    reporter = AutonomousReporter(profile, telegram_alerter, email_alerter)
    reporter.send_daily(data)
    reporter.send_weekly(data)
    reporter.send_monthly(data)
    """

    def __init__(
        self,
        profile: RiskProfileConfig,
        telegram_alerter=None,
        email_alerter=None,
    ) -> None:
        self.profile = profile
        self.telegram = telegram_alerter
        self.email = email_alerter
        self.logger = logging.getLogger(__name__)
        self._last_daily: datetime | None = None
        self._last_weekly: datetime | None = None
        self._last_monthly: datetime | None = None

    def send_daily(self, data: ReportData) -> None:
        if not self.profile.report_daily:
            return
        data.period = "daily"
        self._dispatch(data)
        self._last_daily = datetime.now()

    def send_weekly(self, data: ReportData) -> None:
        if not self.profile.report_weekly:
            return
        data.period = "weekly"
        self._dispatch(data)
        self._last_weekly = datetime.now()

    def send_monthly(self, data: ReportData) -> None:
        if not self.profile.report_monthly:
            return
        data.period = "monthly"
        self._dispatch(data)
        self._last_monthly = datetime.now()

    def maybe_send(self, data: ReportData) -> list[str]:
        """
        Automatically send whichever reports are due.
        Returns list of report types sent.
        """
        sent: list[str] = []
        now = datetime.now()

        if self._daily_due(now):
            self.send_daily(data)
            sent.append("daily")

        if self._weekly_due(now):
            self.send_weekly(data)
            sent.append("weekly")

        if self._monthly_due(now):
            self.send_monthly(data)
            sent.append("monthly")

        return sent

    # ------------------------------------------------------------------

    def _dispatch(self, data: ReportData) -> None:
        text = data.to_text()
        self.logger.info("Sending %s report", data.period)

        if self.telegram:
            try:
                self.telegram.daily_report(text)
            except Exception as exc:
                self.logger.warning("Telegram report failed: %s", exc)

        if self.email:
            try:
                subject = (
                    f"AI Trading — {data.period.capitalize()} Report "
                    f"({data.end_date})  P&L ₹{data.total_pnl:+,.0f}"
                )
                html = data.to_html()
                self.email.send(subject, html, html=True)
            except Exception as exc:
                self.logger.warning("Email report failed: %s", exc)

    def _daily_due(self, now: datetime) -> bool:
        if not self.profile.report_daily:
            return False
        if self._last_daily is None:
            return True
        return (now - self._last_daily) >= timedelta(hours=20)

    def _weekly_due(self, now: datetime) -> bool:
        if not self.profile.report_weekly:
            return False
        if self._last_weekly is None:
            return now.weekday() == 4   # Friday
        return (now - self._last_weekly) >= timedelta(days=6)

    def _monthly_due(self, now: datetime) -> bool:
        if not self.profile.report_monthly:
            return False
        if self._last_monthly is None:
            return now.day == 1
        return (now - self._last_monthly) >= timedelta(days=28)

    # ------------------------------------------------------------------

    @staticmethod
    def build_report(
        period: str,
        portfolio_snapshot: dict[str, Any],
        trade_journal=None,
        safety_system=None,
        model_manager=None,
        regime_str: str = "UNKNOWN",
        starting_capital: float = 0.0,
    ) -> ReportData:
        """
        Convenience constructor that extracts fields from live system state.
        """
        bal = portfolio_snapshot.get("balance", {})
        cash = float(bal.get("cash", 0))
        total_value = float(bal.get("total_value", 0)) or cash
        unrealized = float(bal.get("unrealized_pnl", 0))
        today_pnl = float(bal.get("today_pnl", 0))

        # Realised P&L from journal
        realized = 0.0
        num_trades = 0
        win_rate = 0.0
        best_trade: dict = {}
        worst_trade: dict = {}
        if trade_journal:
            closed = trade_journal.closed_trades()
            num_trades = len(closed)
            realized = sum(t.realized_pnl for t in closed)
            if closed:
                winning = sum(1 for t in closed if t.realized_pnl > 0)
                win_rate = winning / len(closed) * 100
                best = max(closed, key=lambda t: t.realized_pnl)
                worst_t = min(closed, key=lambda t: t.realized_pnl)
                best_trade = {"symbol": best.symbol, "pnl": best.realized_pnl}
                worst_trade = {"symbol": worst_t.symbol, "pnl": worst_t.realized_pnl}

        # Model accuracy
        model_accuracy = 0.0
        if model_manager:
            model_accuracy = model_manager.state.accuracy * 100

        # Safety events
        safety_events: list[str] = []
        if safety_system and not safety_system.is_safe:
            st = safety_system.current_status
            safety_events.append(f"{st.reason.value}: {st.detail}")

        total_pnl = realized + unrealized
        ret_pct = total_pnl / starting_capital * 100 if starting_capital else 0.0

        now = datetime.now()
        return ReportData(
            period=period,
            start_date=now.strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
            total_capital=starting_capital or total_value,
            portfolio_value=total_value,
            cash=cash,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=total_pnl,
            total_return_pct=ret_pct,
            num_trades=num_trades,
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            positions=portfolio_snapshot.get("positions", []),
            regime=regime_str,
            model_accuracy=model_accuracy,
            safety_events=safety_events,
        )
