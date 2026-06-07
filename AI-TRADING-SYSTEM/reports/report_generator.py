from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.results import BacktestMetrics
from backtesting.strategy_tester import StrategyResult


logger = logging.getLogger(__name__)

_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; color: #222; }
  h1 { color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }
  h2 { color: #283593; margin-top: 32px; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
  th { background: #283593; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f5f7ff; }
  .metric-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }
  .metric-card { background: #e8eaf6; border-radius: 8px; padding: 16px 24px; min-width: 160px; }
  .metric-card .label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-card .value { font-size: 22px; font-weight: bold; color: #1a237e; }
  .positive { color: #2e7d32; } .negative { color: #c62828; }
  img { max-width: 100%; border-radius: 4px; margin-top: 12px; }
  .footer { margin-top: 48px; font-size: 11px; color: #999; }
</style>
"""


def _metric_card(label: str, value: Any, positive_good: bool = True) -> str:
    str_val = str(value)
    css = ""
    try:
        num = float(str(value).replace("%", "").replace("₹", "").replace(",", ""))
        if num > 0:
            css = "positive" if positive_good else "negative"
        elif num < 0:
            css = "negative" if positive_good else "positive"
    except ValueError:
        pass
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value {css}">{str_val}</div>'
        f'</div>'
    )


class ReportGenerator:
    """Generates backtest, strategy, portfolio, and risk reports in CSV, HTML (and optionally PDF)."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Backtest report
    # ------------------------------------------------------------------

    def backtest_report(
        self,
        result: StrategyResult,
        chart_paths: dict[str, str] | None = None,
        symbol: str = "",
    ) -> dict[str, str]:
        """Generate CSV + HTML backtest report. Returns {format: path}."""
        safe = result.strategy_name.replace(" ", "_").replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"backtest_{safe}_{ts}"

        csv_path = self._write_metrics_csv(result.metrics, stem)
        html_path = self._write_backtest_html(result, chart_paths or {}, symbol, stem)

        if not result.trades_df.empty:
            trades_path = self.output_dir / f"{stem}_trades.csv"
            result.trades_df.to_csv(trades_path, index=False)

        logger.info("Backtest report generated", extra={"strategy": result.strategy_name})
        return {"csv": str(csv_path), "html": str(html_path)}

    # ------------------------------------------------------------------
    # Strategy comparison report
    # ------------------------------------------------------------------

    def strategy_comparison_report(
        self,
        comparison_table: pd.DataFrame,
        winner: str,
    ) -> dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"strategy_comparison_{ts}"

        csv_path = self.output_dir / f"{stem}.csv"
        comparison_table.to_csv(csv_path, index=False)

        rows_html = ""
        for _, row in comparison_table.iterrows():
            highlight = ' style="background:#e8f5e9; font-weight:bold;"' if row.get("strategy_name") == winner else ""
            cells = "".join(f"<td>{v}</td>" for v in row.values)
            rows_html += f"<tr{highlight}>{cells}</tr>"

        html_content = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Strategy Comparison Report</h1>"
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
            f"<p><strong>Winner:</strong> {winner}</p>"
            f"<table><tr>{''.join(f'<th>{c}</th>' for c in comparison_table.columns)}</tr>{rows_html}</table>"
            f"<div class='footer'>AI Trading System — Phase 5 Backtesting</div>"
            f"</body></html>"
        )
        html_path = self.output_dir / f"{stem}.html"
        html_path.write_text(html_content, encoding="utf-8")

        logger.info("Strategy comparison report generated")
        return {"csv": str(csv_path), "html": str(html_path)}

    # ------------------------------------------------------------------
    # Portfolio report
    # ------------------------------------------------------------------

    def portfolio_report(
        self,
        symbol_results: dict[str, StrategyResult],
        combined_metrics: BacktestMetrics,
        capital_allocation: dict[str, float],
    ) -> dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"portfolio_report_{ts}"

        # Build summary table
        rows = []
        for sym, res in symbol_results.items():
            m = res.metrics
            rows.append({
                "symbol": sym,
                "allocation": capital_allocation.get(sym, 0),
                "total_return_pct": m.total_return_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown_pct": m.max_drawdown_pct,
                "win_rate": m.win_rate,
                "total_trades": m.total_trades,
            })
        summary_df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{stem}.csv"
        summary_df.to_csv(csv_path, index=False)

        # HTML
        table_html = self._df_to_html_table(summary_df)
        combined_cards = self._metrics_cards(combined_metrics)
        html_content = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Portfolio Backtest Report</h1>"
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
            f"<h2>Portfolio Metrics</h2><div class='metric-grid'>{combined_cards}</div>"
            f"<h2>Per-Symbol Summary</h2>{table_html}"
            f"<div class='footer'>AI Trading System — Phase 5 Backtesting</div>"
            f"</body></html>"
        )
        html_path = self.output_dir / f"{stem}.html"
        html_path.write_text(html_content, encoding="utf-8")

        logger.info("Portfolio report generated")
        return {"csv": str(csv_path), "html": str(html_path)}

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    def risk_report(
        self,
        metrics: BacktestMetrics,
        monte_carlo_summary: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"risk_report_{ts}"

        risk_data = {
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "profit_factor": metrics.profit_factor,
            "total_fees": metrics.total_fees,
        }
        if monte_carlo_summary:
            risk_data.update(monte_carlo_summary)

        csv_path = self.output_dir / f"{stem}.csv"
        pd.DataFrame([risk_data]).to_csv(csv_path, index=False)

        mc_section = ""
        if monte_carlo_summary:
            mc_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in monte_carlo_summary.items())
            mc_section = f"<h2>Monte Carlo Analysis</h2><table><tr><th>Metric</th><th>Value</th></tr>{mc_rows}</table>"

        html_content = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Risk Report</h1>"
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
            f"<h2>Risk Metrics</h2><div class='metric-grid'>"
            + _metric_card("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%", positive_good=False)
            + _metric_card("Sharpe", f"{metrics.sharpe_ratio:.3f}")
            + _metric_card("Sortino", f"{metrics.sortino_ratio:.3f}")
            + _metric_card("Calmar", f"{metrics.calmar_ratio:.3f}")
            + _metric_card("Profit Factor", f"{metrics.profit_factor:.3f}")
            + f"</div>{mc_section}"
            f"<div class='footer'>AI Trading System — Phase 5 Backtesting</div>"
            f"</body></html>"
        )
        html_path = self.output_dir / f"{stem}.html"
        html_path.write_text(html_content, encoding="utf-8")

        logger.info("Risk report generated")
        return {"csv": str(csv_path), "html": str(html_path)}

    # ------------------------------------------------------------------
    # PDF export (optional — requires weasyprint or pdfkit)
    # ------------------------------------------------------------------

    def export_pdf(self, html_path: str | Path) -> str | None:
        html_path = Path(html_path)
        pdf_path = html_path.with_suffix(".pdf")
        try:
            import weasyprint
            weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            logger.info("PDF exported", extra={"path": str(pdf_path)})
            return str(pdf_path)
        except ImportError:
            pass
        try:
            import pdfkit
            pdfkit.from_file(str(html_path), str(pdf_path))
            logger.info("PDF exported via pdfkit", extra={"path": str(pdf_path)})
            return str(pdf_path)
        except (ImportError, OSError):
            logger.warning("PDF export skipped (install weasyprint or pdfkit).")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_metrics_csv(self, metrics: BacktestMetrics, stem: str) -> Path:
        path = self.output_dir / f"{stem}_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.to_dict().keys()))
            writer.writeheader()
            writer.writerow(metrics.to_dict())
        return path

    def _write_backtest_html(
        self,
        result: StrategyResult,
        chart_paths: dict[str, str],
        symbol: str,
        stem: str,
    ) -> Path:
        m = result.metrics
        cards = self._metrics_cards(m)
        trades_html = self._df_to_html_table(result.trades_df) if not result.trades_df.empty else "<p>No closed trades.</p>"

        png_tag = ""
        if chart_paths.get("png") and Path(chart_paths["png"]).exists():
            png_rel = Path(chart_paths["png"]).name
            png_tag = f"<img src='{png_rel}' alt='Equity Curve'>"

        html_link = ""
        if chart_paths.get("html"):
            html_link = f"<p><a href='{Path(chart_paths['html']).name}'>Open Interactive Chart</a></p>"

        html_content = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"
            f"<h1>Backtest Report: {result.strategy_name}</h1>"
            f"<p>Symbol: <strong>{symbol or 'N/A'}</strong> &nbsp;|&nbsp; "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
            f"<h2>Performance Metrics</h2><div class='metric-grid'>{cards}</div>"
            f"<h2>Equity Curve</h2>{png_tag}{html_link}"
            f"<h2>Trade Log</h2>{trades_html}"
            f"<div class='footer'>AI Trading System — Phase 5 Backtesting</div>"
            f"</body></html>"
        )
        path = self.output_dir / f"{stem}.html"
        path.write_text(html_content, encoding="utf-8")
        return path

    @staticmethod
    def _metrics_cards(m: BacktestMetrics) -> str:
        return (
            _metric_card("Total Return", f"{m.total_return_pct:+.2f}%")
            + _metric_card("Annualised Return", f"{m.annualized_return:+.2f}%")
            + _metric_card("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
            + _metric_card("Sortino Ratio", f"{m.sortino_ratio:.3f}")
            + _metric_card("Max Drawdown", f"{m.max_drawdown_pct:.2f}%", positive_good=False)
            + _metric_card("Win Rate", f"{m.win_rate:.1f}%")
            + _metric_card("Profit Factor", f"{m.profit_factor:.3f}")
            + _metric_card("Total Trades", str(m.total_trades))
            + _metric_card("Expectancy", f"{m.expectancy:.2f}")
            + _metric_card("Total Fees", f"₹{m.total_fees:,.2f}", positive_good=False)
        )

    @staticmethod
    def _df_to_html_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No data.</p>"
        headers = "".join(f"<th>{c}</th>" for c in df.columns)
        rows = ""
        for _, row in df.iterrows():
            cells = "".join(f"<td>{v}</td>" for v in row.values)
            rows += f"<tr>{cells}</tr>"
        return f"<table><tr>{headers}</tr>{rows}</table>"
