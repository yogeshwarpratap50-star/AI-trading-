from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from paper_trading.paper_portfolio import PaperPortfolio

logger = logging.getLogger(__name__)

_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; color: #222; }
  h1 { color: #1b5e20; border-bottom: 2px solid #1b5e20; padding-bottom: 8px; }
  h2 { color: #2e7d32; margin-top: 28px; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }
  th { background: #2e7d32; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f1f8e9; }
  .card { background: #e8f5e9; border-radius: 8px; padding: 14px 20px; display: inline-block; min-width: 140px; margin: 6px; }
  .card .lbl { font-size: 11px; color: #555; text-transform: uppercase; }
  .card .val { font-size: 20px; font-weight: bold; color: #1b5e20; }
  .pos { color: #2e7d32; } .neg { color: #c62828; }
  .footer { margin-top: 40px; font-size: 11px; color: #999; }
</style>
"""


def _card(label: str, value: str, positive: bool | None = None) -> str:
    css = ("pos" if positive else "neg") if positive is not None else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="val {css}">{value}</div></div>'


class PaperReportGenerator:
    """Generates daily and summary paper-trading reports in CSV and HTML."""

    def __init__(self, output_dir: str | Path = "reports/paper_trading") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def daily_report(self, portfolio: PaperPortfolio, date_str: str | None = None) -> dict[str, str]:
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Positions
        positions_df = pd.DataFrame([p.to_dict() for p in portfolio.positions.values()])
        closed_today = [
            c for c in portfolio.closed_positions
            if c.exit_date.strftime("%Y-%m-%d") == date_str
        ]
        closed_df = pd.DataFrame([c.to_dict() for c in closed_today])

        snap = portfolio.snapshot()

        # CSV
        csv_path = self.output_dir / f"daily_{date_str}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in snap.items():
                writer.writerow([k, v])

        # HTML
        pos_table = self._df_table(positions_df, "No open positions.")
        closed_table = self._df_table(closed_df, "No trades closed today.")
        today_pnl = portfolio.today_pnl()
        pnl_sign = today_pnl >= 0

        html = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Paper Trading Daily Report — {date_str}</h1>"
            f"<div>"
            + _card("Portfolio Value", f"₹{snap['total_value']:,.2f}")
            + _card("Cash", f"₹{snap['cash']:,.2f}")
            + _card("Unrealized P&L", f"₹{snap['unrealized_pnl']:+,.2f}", snap["unrealized_pnl"] >= 0)
            + _card("Today's P&L", f"₹{today_pnl:+,.2f}", pnl_sign)
            + _card("Realized P&L", f"₹{snap['realized_pnl']:+,.2f}", snap["realized_pnl"] >= 0)
            + _card("Total Return", f"{snap['total_return_pct']:+.2f}%", snap["total_return_pct"] >= 0)
            + f"</div>"
            f"<h2>Open Positions</h2>{pos_table}"
            f"<h2>Trades Closed Today</h2>{closed_table}"
            f"<div class='footer'>AI Trading System — Paper Trading Report</div>"
            f"</body></html>"
        )
        html_path = self.output_dir / f"daily_{date_str}_{ts_tag}.html"
        html_path.write_text(html, encoding="utf-8")

        logger.info("Daily report generated", extra={"date": date_str})
        return {"csv": str(csv_path), "html": str(html_path)}

    def summary_report(self, portfolio: PaperPortfolio) -> dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = portfolio.snapshot()

        all_closed = pd.DataFrame([c.to_dict() for c in portfolio.closed_positions])
        equity_df = pd.DataFrame(portfolio.equity_history)

        csv_path = self.output_dir / f"summary_{ts}.csv"
        if not all_closed.empty:
            all_closed.to_csv(csv_path, index=False)
        else:
            pd.DataFrame([snap]).to_csv(csv_path, index=False)

        equity_section = ""
        if not equity_df.empty:
            equity_section = f"<h2>Equity History</h2>{self._df_table(equity_df)}"

        html = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Paper Trading Summary Report</h1>"
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p><div>"
            + _card("Starting Capital", f"₹{portfolio.starting_capital:,.2f}")
            + _card("Portfolio Value", f"₹{snap['total_value']:,.2f}")
            + _card("Total P&L", f"₹{snap['total_pnl']:+,.2f}", snap["total_pnl"] >= 0)
            + _card("Total Return", f"{snap['total_return_pct']:+.2f}%", snap["total_return_pct"] >= 0)
            + _card("Realized P&L", f"₹{snap['realized_pnl']:+,.2f}", snap["realized_pnl"] >= 0)
            + _card("Trades", str(snap["closed_positions"]))
            + f"</div>"
            f"<h2>All Closed Trades</h2>{self._df_table(all_closed)}"
            + equity_section
            + f"<div class='footer'>AI Trading System — Paper Trading</div></body></html>"
        )
        html_path = self.output_dir / f"summary_{ts}.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info("Summary report generated")
        return {"csv": str(csv_path), "html": str(html_path)}

    @staticmethod
    def _df_table(df: pd.DataFrame, empty_msg: str = "No data.") -> str:
        if df is None or df.empty:
            return f"<p>{empty_msg}</p>"
        headers = "".join(f"<th>{c}</th>" for c in df.columns)
        rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values) + "</tr>"
            for _, row in df.iterrows()
        )
        return f"<table><tr>{headers}</tr>{rows}</table>"
