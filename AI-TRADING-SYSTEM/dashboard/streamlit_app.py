from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import pandas as pd
import streamlit as st

from ai_trading_system.config.settings import get_settings
from features.feature_engineering import FeatureEngineeringService
from sentiment.sentiment_engine import SentimentEngine


st.set_page_config(page_title="AI Trading System", layout="wide")


# ===========================================================================
# Shared utilities
# ===========================================================================

# Whitelist of tables that load_table() may query — prevents SQL injection
_ALLOWED_TABLES = frozenset({
    "market_data", "news_articles", "sentiment_results",
    "predictions", "trades", "orders",
})


def load_table(table: str, limit: int = 500) -> pd.DataFrame:
    # SECURITY: reject any table name not in the allow-list (prevents SQL injection
    # if table param ever becomes user-controlled via st.selectbox / URL params).
    if table not in _ALLOWED_TABLES:
        return pd.DataFrame()
    settings = get_settings()
    db_path = Path(settings.database_url.replace("sqlite:///", "", 1))
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as connection:
        try:
            # Table name is safe (whitelist-validated above); LIMIT param is bound
            return pd.read_sql_query(
                f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?",  # noqa: S608
                connection,
                params=(limit,),
            )
        except Exception:
            return pd.DataFrame()


def load_historical_csv(ticker: str) -> pd.DataFrame:
    path = Path("data/historical") / f"{ticker.replace('.', '_')}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def auto_refresh_control() -> None:
    settings = get_settings()
    interval = st.sidebar.number_input(
        "Refresh seconds",
        min_value=10,
        max_value=300,
        value=settings.dashboard_refresh_seconds,
        step=5,
    )
    if st.sidebar.toggle("Live refresh", value=True):
        # interval is always an int (number_input enforced); clamp for safety
        safe_interval = max(10, min(300, int(interval)))
        st.caption(f"Auto refresh enabled every {safe_interval} seconds.")
        # Use st.empty + st.rerun via query param rather than raw HTML meta tag
        st.markdown(
            f"<meta http-equiv='refresh' content='{safe_interval}'>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Existing pages (unchanged)
# ===========================================================================

def overview_page() -> None:
    st.header("Overview")
    sentiments = load_table("sentiment_results", 100)
    live_quotes = load_table("live_quotes", 50)
    market_health = 50.0
    if not sentiments.empty:
        market_health = round(50 + sentiments["sentiment_score"].astype(float).mean() * 50, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Market Status", "Open / Monitoring")
    col2.metric("NIFTY Trend", "Bullish" if market_health >= 55 else "Bearish" if market_health <= 45 else "Neutral")
    col3.metric("Market Health Score", market_health)

    st.subheader("Latest Predictions")
    st.dataframe(pd.DataFrame(columns=["ticker", "signal", "confidence", "risk_score"]), use_container_width=True)
    st.subheader("Latest Signals")
    st.dataframe(live_quotes, use_container_width=True)


def news_page() -> None:
    st.header("News")
    news = load_table("news_articles", 500)
    sentiments = load_table("sentiment_results", 500)
    if news.empty:
        st.info("No news available yet.")
        return
    search = st.text_input("Search News")
    ticker = st.selectbox("Ticker", ["All"] + sorted(news["ticker"].dropna().unique().tolist()))
    if not sentiments.empty:
        sentiment_filter = st.selectbox("Sentiment", ["All"] + sorted(sentiments["sentiment"].dropna().unique().tolist()))
        news = news.merge(sentiments[["headline", "sentiment", "sentiment_score"]], on="headline", how="left")
        if sentiment_filter != "All":
            news = news[news["sentiment"] == sentiment_filter]
    if ticker != "All":
        news = news[news["ticker"] == ticker]
    if search:
        news = news[news["headline"].str.contains(search, case=False, na=False)]
    st.dataframe(news, use_container_width=True)


def sentiment_page() -> None:
    st.header("Sentiment")
    sentiments = load_table("sentiment_results", 500)
    if sentiments.empty:
        st.info("No sentiment data available yet.")
        return
    distribution = sentiments["sentiment"].value_counts().reset_index()
    distribution.columns = ["sentiment", "count"]
    st.bar_chart(distribution, x="sentiment", y="count")
    col1, col2 = st.columns(2)
    col1.subheader("Positive News")
    col1.dataframe(sentiments[sentiments["sentiment"] == "Positive"].head(25), use_container_width=True)
    col2.subheader("Negative News")
    col2.dataframe(sentiments[sentiments["sentiment"] == "Negative"].head(25), use_container_width=True)
    st.subheader("Confidence Scores")
    st.line_chart(sentiments[["confidence_score"]].astype(float))


def market_intelligence_page() -> None:
    st.header("Market Intelligence")
    live_quotes = load_table("live_quotes", 500)
    if live_quotes.empty:
        st.info("No live quote data available yet.")
        return
    numeric = live_quotes.copy()
    numeric["last_price"] = pd.to_numeric(numeric["last_price"], errors="coerce")
    numeric["open"] = pd.to_numeric(numeric["open"], errors="coerce")
    numeric["volume"] = pd.to_numeric(numeric["volume"], errors="coerce")
    numeric["return_pct"] = ((numeric["last_price"] - numeric["open"]) / numeric["open"].replace(0, pd.NA)).fillna(0)
    st.subheader("Top Gainers")
    st.dataframe(numeric.sort_values("return_pct", ascending=False).head(10), use_container_width=True)
    st.subheader("Top Losers")
    st.dataframe(numeric.sort_values("return_pct", ascending=True).head(10), use_container_width=True)
    st.subheader("High Volume Stocks")
    st.dataframe(numeric.sort_values("volume", ascending=False).head(10), use_container_width=True)
    st.subheader("Gap Up Stocks")
    st.dataframe(numeric[numeric["return_pct"] > 0.005], use_container_width=True)
    st.subheader("Gap Down Stocks")
    st.dataframe(numeric[numeric["return_pct"] < -0.005], use_container_width=True)


def stock_analysis_page() -> None:
    st.header("Stock Analysis")
    ticker = st.text_input("Ticker", value="RELIANCE.NS")
    frame = load_historical_csv(ticker)
    if frame.empty:
        st.info("No historical CSV found for this ticker.")
        return
    features, _ = FeatureEngineeringService().create_feature_dataset(frame)
    sentiments = load_table("sentiment_results", 500)
    ticker_sentiment = sentiments[sentiments["ticker"] == ticker] if not sentiments.empty else pd.DataFrame()
    sentiment_score = 0.0 if ticker_sentiment.empty else round(float(ticker_sentiment["sentiment_score"].mean()), 4)
    st.metric("Sentiment Score", sentiment_score)
    st.line_chart(features.set_index("date")[["close", "ema_20", "ema_50"]])
    st.line_chart(features.set_index("date")[["rsi_14"]])
    st.line_chart(features.set_index("date")[["macd", "macd_signal"]])
    st.bar_chart(features.set_index("date")[["volume"]])
    st.line_chart(features.set_index("date")[["daily_return"]])


def ai_models_page() -> None:
    st.header("AI Models")
    registry_path = Path("models/model_registry.json")
    if not registry_path.exists():
        st.info("No trained model registry found yet.")
        return
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    if not entries:
        st.info("No trained models registered yet.")
        return
    registry = pd.DataFrame(entries)
    latest = registry.sort_values("training_date").iloc[-1]
    st.metric("Current Model", latest["model_name"])
    metrics = latest.get("metrics", {})
    st.metric("Model Accuracy", metrics.get("accuracy", "N/A"))
    st.dataframe(registry, use_container_width=True)
    st.subheader("Feature Importance")
    model_name = latest["model_name"]
    importance_path = Path("reports") / f"{model_name}_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(20)
        st.bar_chart(importance, x="feature", y="importance")
        st.dataframe(importance, use_container_width=True)
    else:
        st.info("No feature importance report found yet.")


def predictions_page() -> None:
    st.header("Predictions")
    prediction_path = Path("reports/latest_prediction.csv")
    if prediction_path.exists():
        prediction = pd.read_csv(prediction_path)
        latest = prediction.iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("Latest Prediction", latest.get("action", "N/A"))
        col2.metric("Confidence %", latest.get("confidence", "N/A"))
        st.dataframe(prediction, use_container_width=True)
    else:
        st.info("No prediction report found yet. Run the prediction command after training a model.")


# ===========================================================================
# Phase 5 — Backtesting pages
# ===========================================================================

def _run_backtest(data: pd.DataFrame, strategy_name: str, symbol: str, capital: float, risk_pct: float, brokerage: float, slippage: float) -> None:
    from backtesting.engine import BacktestConfig
    from backtesting.order_simulator import OrderConfig
    from backtesting.strategy_tester import StrategyTester, IndicatorStrategy, MACDStrategy, BollingerBandStrategy, AIPredictionStrategy, HybridStrategy
    from backtesting.results import EquityCurveGenerator
    from reports.report_generator import ReportGenerator

    order_cfg = OrderConfig(brokerage_pct=brokerage / 100, slippage_pct=slippage / 100)
    cfg = BacktestConfig(starting_capital=capital, risk_per_trade_pct=risk_pct, order_config=order_cfg)
    tester = StrategyTester(cfg)

    strategy_map = {
        "Indicator (EMA+RSI)": IndicatorStrategy(),
        "Indicator (MACD)": MACDStrategy(),
        "Indicator (Bollinger Bands)": BollingerBandStrategy(),
    }
    if "ai_action" in data.columns:
        strategy_map["AI Prediction"] = AIPredictionStrategy()
        strategy_map["Hybrid (AI+Indicator)"] = HybridStrategy(AIPredictionStrategy(), IndicatorStrategy())

    strategy = strategy_map.get(strategy_name, IndicatorStrategy())
    result = tester.test(data, strategy, strategy_name, symbol)

    # Metrics
    m = result.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{m.total_return_pct:+.2f}%")
    col2.metric("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
    col3.metric("Max Drawdown", f"{m.max_drawdown_pct:.2f}%")
    col4.metric("Win Rate", f"{m.win_rate:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Annualised Return", f"{m.annualized_return:+.2f}%")
    col6.metric("Profit Factor", f"{m.profit_factor:.3f}")
    col7.metric("Total Trades", m.total_trades)
    col8.metric("Expectancy", f"{m.expectancy:.2f}")

    # Equity curve
    if not result.equity_df.empty:
        st.subheader("Equity Curve")
        eq = result.equity_df.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        st.line_chart(eq.set_index("date")[["equity"]])

        st.subheader("Drawdown")
        equity_vals = eq["equity"].values
        import numpy as np
        peak = np.maximum.accumulate(equity_vals)
        dd = (equity_vals - peak) / np.where(peak > 0, peak, 1) * 100
        dd_df = pd.DataFrame({"drawdown_pct": dd}, index=eq["date"])
        st.area_chart(dd_df)

    # Trade log
    if not result.trades_df.empty:
        st.subheader("Trade Log")
        st.dataframe(result.trades_df, use_container_width=True)

    # Generate and save report
    gen = ReportGenerator()
    chart_gen = EquityCurveGenerator()
    chart_paths = chart_gen.generate_charts(result.equity_df, m, strategy_name)
    report_paths = gen.backtest_report(result, chart_paths, symbol)
    st.success(f"Report saved: {report_paths.get('html', '')}")


def backtesting_page() -> None:
    st.header("Backtesting")
    st.markdown("Run historical strategy backtests on any loaded ticker.")

    with st.sidebar:
        st.subheader("Backtest Configuration")
        ticker = st.text_input("Ticker", value="RELIANCE.NS", key="bt_ticker")
        capital = st.number_input("Starting Capital (₹)", value=100_000, step=10_000, key="bt_capital")
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.5, key="bt_risk")
        brokerage = st.number_input("Brokerage (%)", value=0.03, step=0.01, format="%.3f", key="bt_brokerage")
        slippage = st.number_input("Slippage (%)", value=0.05, step=0.01, format="%.3f", key="bt_slippage")

    data = load_historical_csv(ticker)
    if data.empty:
        st.warning(f"No historical data found for {ticker}. Run `python main.py --collect-history --symbol {ticker}` first.")
        return

    try:
        features, _ = FeatureEngineeringService().create_feature_dataset(data)
    except Exception as exc:
        st.error(f"Feature engineering failed: {exc}")
        return

    strategy_options = ["Indicator (EMA+RSI)", "Indicator (MACD)", "Indicator (Bollinger Bands)"]
    if "ai_action" in features.columns:
        strategy_options += ["AI Prediction", "Hybrid (AI+Indicator)"]

    strategy_name = st.selectbox("Strategy", strategy_options)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            _run_backtest(features, strategy_name, ticker, float(capital), float(risk_pct), float(brokerage), float(slippage))


def strategy_comparison_page() -> None:
    st.header("Strategy Comparison")
    st.markdown("Automatically rank all strategies on a chosen ticker.")

    ticker = st.text_input("Ticker", value="RELIANCE.NS", key="sc_ticker")
    capital = st.number_input("Starting Capital (₹)", value=100_000, step=10_000, key="sc_capital")

    if st.button("Compare All Strategies", type="primary"):
        data = load_historical_csv(ticker)
        if data.empty:
            st.warning(f"No historical data found for {ticker}.")
            return
        try:
            features, _ = FeatureEngineeringService().create_feature_dataset(data)
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            return

        from backtesting.engine import BacktestConfig
        from backtesting.strategy_tester import StrategyTester
        from backtesting.strategy_comparison import StrategyComparison
        from reports.report_generator import ReportGenerator

        with st.spinner("Running all strategies…"):
            cfg = BacktestConfig(starting_capital=float(capital))
            tester = StrategyTester(cfg)
            results = tester.test_all_strategies(features, ticker)
            comparison = StrategyComparison().compare(results)

        st.success(f"Winner: **{comparison.winner}**")

        table = comparison.comparison_table.drop(columns=["_result"], errors="ignore")
        st.dataframe(table, use_container_width=True)

        # Equity overlays
        st.subheader("Equity Curves Overlay")
        eq_data: dict[str, pd.Series] = {}
        for r in results:
            if not r.equity_df.empty:
                eq = r.equity_df.copy()
                eq["date"] = pd.to_datetime(eq["date"])
                eq_data[r.strategy_name] = eq.set_index("date")["equity"]
        if eq_data:
            overlay_df = pd.DataFrame(eq_data)
            st.line_chart(overlay_df)

        # Save report
        gen = ReportGenerator()
        paths = gen.strategy_comparison_report(table, comparison.winner)
        st.info(f"Comparison report: {paths.get('html', '')}")


def portfolio_analysis_page() -> None:
    st.header("Portfolio Analysis")
    st.markdown("Backtest a strategy across multiple NIFTY50 symbols simultaneously.")

    settings = get_settings()
    available_tickers = list(settings.nifty50_symbols)

    selected = st.multiselect(
        "Select Symbols",
        available_tickers,
        default=available_tickers[:3],
        key="pa_symbols",
    )
    capital = st.number_input("Total Portfolio Capital (₹)", value=500_000, step=50_000, key="pa_capital")

    if st.button("Run Portfolio Backtest", type="primary") and selected:
        datasets = {}
        for sym in selected:
            df = load_historical_csv(sym)
            if not df.empty:
                try:
                    features, _ = FeatureEngineeringService().create_feature_dataset(df)
                    datasets[sym] = features
                except Exception:
                    st.warning(f"Skipping {sym} — feature engineering failed.")

        if not datasets:
            st.error("No valid datasets loaded.")
            return

        from backtesting.engine import BacktestConfig
        from backtesting.portfolio_backtester import PortfolioBacktester
        from backtesting.strategy_tester import IndicatorStrategy
        from reports.report_generator import ReportGenerator

        cfg = BacktestConfig(starting_capital=float(capital))
        backtester = PortfolioBacktester(cfg, strategy_factory=lambda _s: IndicatorStrategy())

        with st.spinner("Running portfolio backtest…"):
            result = backtester.run(datasets)

        cm = result.combined_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Return", f"{cm.total_return_pct:+.2f}%")
        col2.metric("Sharpe Ratio", f"{cm.sharpe_ratio:.3f}")
        col3.metric("Max Drawdown", f"{cm.max_drawdown_pct:.2f}%")
        col4.metric("Win Rate", f"{cm.win_rate:.1f}%")

        st.subheader("Per-Symbol Summary")
        st.dataframe(result.summary_table(), use_container_width=True)

        if not result.combined_equity.empty:
            st.subheader("Portfolio Equity Curve")
            eq = result.combined_equity.copy()
            eq["date"] = pd.to_datetime(eq["date"])
            st.line_chart(eq.set_index("date")[["equity"]])

        gen = ReportGenerator()
        paths = gen.portfolio_report(result.symbol_results, cm, result.capital_allocation)
        st.info(f"Portfolio report: {paths.get('html', '')}")


# ===========================================================================
# Phase 6 — Paper Trading page
# ===========================================================================

def paper_trading_page() -> None:
    st.header("Paper Trading")
    st.caption("All activity is simulated — no real orders are placed.")

    from paper_trading.paper_broker import BrokerConfig, PaperBroker
    from paper_trading.paper_reports import PaperReportGenerator

    # Use @st.cache_resource so a single PaperBroker instance is shared across
    # all reruns and browser tabs — eliminates the session-state race condition
    # where two simultaneous widget interactions could create two broker instances.
    @st.cache_resource
    def _get_paper_broker() -> PaperBroker:
        cfg = BrokerConfig(starting_capital=100_000.0)
        return PaperBroker(cfg)

    pb: PaperBroker = _get_paper_broker()

    # Quick controls
    col_a, col_b = st.columns(2)
    if col_a.button("Enable Trading"):
        pb.enable_trading(True)
        st.success("Trading enabled")
    if col_b.button("Disable Trading"):
        pb.enable_trading(False)
        st.warning("Trading disabled")

    # Manual order panel
    st.subheader("Place Simulated Order")
    o_col1, o_col2, o_col3, o_col4, o_col5 = st.columns(5)
    sym = o_col1.text_input("Symbol", value="RELIANCE", key="pt_sym")
    qty = o_col2.number_input("Qty", min_value=1, value=10, key="pt_qty")
    price = o_col3.number_input("Price (₹)", min_value=1.0, value=2500.0, step=10.0, key="pt_price")
    side = o_col4.selectbox("Side", ["BUY", "SELL"], key="pt_side")
    if o_col5.button("Submit", type="primary"):
        pb.update_prices({sym: price})
        if side == "BUY":
            order = pb.place_buy(sym, int(qty), float(price))
        else:
            order = pb.place_sell(sym, int(qty), float(price))
        st.success(f"Order {order.status.value}: {order.order_id}")

    # Portfolio snapshot
    snap = pb.snapshot()
    bal = snap["balance"]
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Cash (₹)", f"₹{bal['cash']:,.0f}")
    s_col2.metric("Portfolio Value (₹)", f"₹{bal['total_value']:,.0f}")
    s_col3.metric("Unrealized P&L (₹)", f"₹{bal.get('unrealized_pnl', 0):,.0f}")
    s_col4.metric("Today's P&L (₹)", f"₹{bal.get('today_pnl', 0):,.0f}")

    # Open positions
    st.subheader("Open Positions")
    positions = snap.get("positions", [])
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)
    else:
        st.info("No open positions.")

    # Closed trades
    st.subheader("Closed Trades")
    closed = snap.get("closed_positions", [])
    if closed:
        st.dataframe(pd.DataFrame(closed), use_container_width=True)
    else:
        st.info("No closed trades yet.")

    # Reports
    if st.button("Generate Daily Report"):
        gen = PaperReportGenerator()
        paths = gen.daily_report(pb.portfolio)
        st.success(f"Report: {paths.get('html', '')}")


# ===========================================================================
# Phase 7 — Risk Dashboard page
# ===========================================================================

def risk_dashboard_page() -> None:
    st.header("Risk Dashboard")
    st.caption("Real-time risk monitoring — no positions are opened from this page.")

    from risk_management.risk_engine import RiskConfig, RiskEngine
    from risk_management.portfolio_risk import PortfolioRiskAnalyzer

    cfg = RiskConfig()
    engine = RiskEngine(cfg)

    st.subheader("Risk Configuration")
    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric("Max Open Positions", cfg.max_open_positions)
    r_col2.metric("Daily Loss Limit", f"{cfg.daily_loss_limit_pct:.1f}%")
    r_col3.metric("Max Portfolio Exposure", f"{cfg.max_portfolio_exposure_pct:.0f}%")

    r_col4, r_col5, r_col6 = st.columns(3)
    r_col4.metric("Max Drawdown Allowed", f"{cfg.max_drawdown_pct:.0f}%")
    r_col5.metric("Risk per Trade", f"{cfg.risk_per_trade_pct:.1f}%")
    r_col6.metric("Min AI Confidence", f"{cfg.min_ai_confidence:.0f}%")

    # Simulate position sizing
    st.subheader("Position Sizer")
    ps_col1, ps_col2, ps_col3 = st.columns(3)
    pv = ps_col1.number_input("Portfolio Value (₹)", value=100_000.0, step=10_000.0, key="rd_pv")
    ep = ps_col2.number_input("Entry Price (₹)", value=2500.0, step=10.0, key="rd_ep")
    atr = ps_col3.number_input("ATR", value=50.0, step=5.0, key="rd_atr")

    if st.button("Calculate Position Size"):
        qty = engine.size_position("SAMPLE", float(ep), capital=float(pv), atr=float(atr))
        stop = engine.stop_loss(float(ep), atr=float(atr))
        target = engine.take_profit(float(ep), atr=float(atr))
        ps_c1, ps_c2, ps_c3 = st.columns(3)
        ps_c1.metric("Suggested Quantity", qty)
        ps_c2.metric("Stop Loss (₹)", f"₹{stop:,.2f}")
        ps_c3.metric("Take Profit (₹)", f"₹{target:,.2f}")

    # Portfolio risk analysis (uses paper broker positions if available)
    st.subheader("Portfolio Risk Analysis")
    if "paper_broker" in st.session_state:
        pb = st.session_state["paper_broker"]
        positions = pb.get_positions()
        cash = pb.get_balance().get("cash", 0.0)
        total = pb.get_balance().get("total_value", cash)
        report = engine.portfolio_risk(positions, cash, total)
        rp_col1, rp_col2, rp_col3, rp_col4 = st.columns(4)
        rp_col1.metric("Risk Score", f"{report.risk_score:.0f}/100")
        rp_col2.metric("Risk Level", report.risk_level)
        rp_col3.metric("Portfolio Exposure", f"{report.portfolio_exposure_pct:.1f}%")
        rp_col4.metric("Correlation Risk", f"{report.correlation_risk:.3f}")
        st.metric("Market Beta", f"{report.market_beta:.3f}")
        if report.warnings:
            st.subheader("Warnings")
            for w in report.warnings:
                st.warning(w)
    else:
        st.info("Open the Paper Trading page first to load portfolio data.")


# ===========================================================================
# Phase 8 — Broker Status page
# ===========================================================================

def broker_status_page() -> None:
    st.header("Broker Status")
    st.caption("All connections are in SIMULATION mode by default.")

    from broker.broker_factory import BrokerFactory

    with st.sidebar:
        st.subheader("Broker Selection")
        broker_name = st.selectbox("Broker", BrokerFactory.available_brokers(), key="bs_broker")

    broker = BrokerFactory.create(broker_name)
    broker.login()

    mode_label = "SIMULATION" if getattr(broker, "SIMULATION_MODE", True) else "LIVE"
    status_color = "green" if broker.is_connected else "red"

    st.markdown(f"**Broker:** `{broker.broker_name()}`")
    st.markdown(f"**Mode:** `{mode_label}`")
    st.markdown(f"**Connection:** :{status_color}[{'Connected' if broker.is_connected else 'Disconnected'}]")

    balance = {}
    try:
        balance = broker.get_balance()
    except Exception:
        pass

    b_col1, b_col2, b_col3 = st.columns(3)
    b_col1.metric("Cash (₹)", f"₹{balance.get('cash', 0):,.0f}")
    b_col2.metric("Total Value (₹)", f"₹{balance.get('total_value', 0):,.0f}")
    b_col3.metric("Used Capital (₹)", f"₹{balance.get('used_capital', 0):,.0f}")

    st.subheader("Open Positions")
    try:
        positions = broker.get_positions()
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("No open positions.")
    except Exception as exc:
        st.warning(f"Could not fetch positions: {exc}")

    st.subheader("Orders")
    try:
        orders = broker.get_orders()
        if orders:
            st.dataframe(pd.DataFrame(orders), use_container_width=True)
        else:
            st.info("No orders today.")
    except Exception as exc:
        st.warning(f"Could not fetch orders: {exc}")


# ===========================================================================
# Phase 8 — Execution Monitor page
# ===========================================================================

def execution_monitor_page() -> None:
    st.header("Execution Monitor")
    st.caption("Track all order submissions across the current session.")

    from execution.order_tracker import OrderTracker, TrackedStatus

    # Persist tracker in session state
    if "order_tracker" not in st.session_state:
        st.session_state["order_tracker"] = OrderTracker()

    tracker: OrderTracker = st.session_state["order_tracker"]
    summary = tracker.summary()

    em_col1, em_col2, em_col3, em_col4, em_col5 = st.columns(5)
    em_col1.metric("Total", summary.get("total", 0))
    em_col2.metric("Filled", summary.get("FILLED", 0))
    em_col3.metric("Pending", summary.get("PENDING", 0))
    em_col4.metric("Cancelled", summary.get("CANCELLED", 0))
    em_col5.metric("Rejected", summary.get("REJECTED", 0))

    status_filter = st.selectbox("Filter by status", ["All"] + [s.value for s in TrackedStatus])

    if status_filter == "All":
        orders = tracker.all()
    else:
        orders = tracker.by_status(TrackedStatus(status_filter))

    if orders:
        st.dataframe(pd.DataFrame([o.to_dict() for o in orders]), use_container_width=True)
    else:
        st.info("No orders recorded yet. Use Paper Trading or run a strategy to generate orders.")


# ===========================================================================
# Router
# ===========================================================================

# ===========================================================================
# Phase 9 — Live Trading page
# ===========================================================================

def live_trading_page() -> None:
    st.header("Live Trading")
    st.caption("Simulation mode only — no real orders. Set SIMULATION_MODE=false to enable live trading.")

    from execution.trade_executor import TradeJournal
    from broker.broker_factory import BrokerFactory

    if "live_journal" not in st.session_state:
        st.session_state["live_journal"] = TradeJournal()

    journal: TradeJournal = st.session_state["live_journal"]

    # Engine status
    broker = BrokerFactory.create("paper")
    broker.login()
    bal = broker.get_balance()

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Mode", "SIMULATION")
    s_col2.metric("Cash (₹)", f"₹{bal.get('cash', 0):,.0f}")
    s_col3.metric("Open Positions", len(journal.open_trades()))
    s_col4.metric("Today P&L (₹)", f"₹{journal.today_pnl():+,.2f}")

    # Open positions
    st.subheader("Open Positions")
    open_trades = journal.open_trades()
    if open_trades:
        st.dataframe(pd.DataFrame([t.to_dict() for t in open_trades]), use_container_width=True)
    else:
        st.info("No open positions.")

    # Trade journal
    st.subheader("Trade Journal")
    all_df = journal.all_as_df()
    if not all_df.empty:
        st.dataframe(all_df, use_container_width=True)
    else:
        st.info("No trades recorded yet.")

    # Execution status
    st.subheader("Execution Status")
    if "order_tracker" in st.session_state:
        tracker = st.session_state["order_tracker"]
        s = tracker.summary()
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric("Total Orders", s.get("total", 0))
        e_col2.metric("Filled", s.get("FILLED", 0))
        e_col3.metric("Pending", s.get("PENDING", 0))
        e_col4.metric("Rejected", s.get("REJECTED", 0))


# ===========================================================================
# Phase 10 — System Health page
# ===========================================================================

def system_health_page() -> None:
    st.header("System Health")
    from monitoring.system_monitor import SystemMonitor

    monitor = SystemMonitor()
    snap = monitor.snapshot()

    status_color = "green" if snap.healthy else "red"
    st.markdown(f"**Overall Status:** :{status_color}[{'Healthy' if snap.healthy else 'Degraded'}]")

    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    h_col1.metric("CPU %", f"{snap.cpu_pct:.1f}%")
    h_col2.metric("RAM %", f"{snap.ram_pct:.1f}%")
    h_col3.metric("Disk %", f"{snap.disk_pct:.1f}%")
    h_col4.metric("DB Status", "OK" if snap.db_ok else "ERROR")

    r_col1, r_col2 = st.columns(2)
    r_col1.metric("RAM Used (GB)", f"{snap.ram_used_gb:.2f} / {snap.ram_total_gb:.2f}")
    r_col2.metric("Net Recv (MB)", f"{snap.net_recv_mb:,.1f}")

    if snap.warnings:
        st.subheader("Warnings")
        for w in snap.warnings:
            st.warning(w)

    if snap.api_statuses:
        st.subheader("API Health")
        for name, ok in snap.api_statuses.items():
            color = "green" if ok else "red"
            st.markdown(f"- {name}: :{color}[{'UP' if ok else 'DOWN'}]")

    st.caption(f"Snapshot at: {snap.timestamp}")


# ===========================================================================
# Phase 11 — AI Analytics page
# ===========================================================================

def ai_analytics_page() -> None:
    st.header("AI Analytics")

    ticker = st.text_input("Ticker", value="RELIANCE.NS", key="aa_ticker")
    data = load_historical_csv(ticker)

    if data.empty:
        st.warning(f"No historical data for {ticker}.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Regime", "Feature Importance", "Ensemble Model",
        "Drift Detection", "Optimization Results"
    ])

    with tab1:
        st.subheader("Market Regime Detection")
        from advanced_ai.market_regime import MarketRegimeDetector
        detector = MarketRegimeDetector()
        try:
            result = detector.detect(data)
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Regime", result.regime.value)
            r_col2.metric("Confidence", f"{result.confidence:.0%}")
            r_col3.metric("Volatility (Ann.)", f"{result.volatility_pct:.1f}%")
            r_col4.metric("ADX", f"{result.adx:.1f}")
            st.metric("Trend Strength", f"{result.trend_strength:+.4f}")
        except Exception as exc:
            st.error(f"Regime detection failed: {exc}")

    with tab2:
        st.subheader("Feature Importance")
        registry_path = Path("models/model_registry.json")
        if not registry_path.exists():
            st.info("Train a model first to see feature importance.")
        else:
            import json as _json
            entries = _json.loads(registry_path.read_text())
            if entries:
                latest = sorted(entries, key=lambda e: e.get("training_date", ""))[-1]
                imp_path = Path("reports") / f"{latest['model_name']}_feature_importance.csv"
                if imp_path.exists():
                    imp_df = pd.read_csv(imp_path).head(20)
                    st.bar_chart(imp_df, x="feature", y="importance")
                    st.dataframe(imp_df, use_container_width=True)
                else:
                    st.info("No feature importance file found.")

    with tab3:
        st.subheader("Ensemble Model")
        st.info("Ensemble combines RandomForest + XGBoost with weighted voting.")
        st.json({
            "rf_weight": 0.4,
            "xgb_weight": 0.6,
            "min_confidence": 0.6,
            "description": "Weighted probability average → BUY/SELL/HOLD",
        })

    with tab4:
        st.subheader("Drift Detection")
        from advanced_ai.drift_detector import DriftDetector
        try:
            features, _ = FeatureEngineeringService().create_feature_dataset(data)
            mid = len(features) // 2
            detector_drift = DriftDetector()
            detector_drift.fit_reference(features.iloc[:mid])
            report = detector_drift.detect(features.iloc[mid:])
            d_col1, d_col2 = st.columns(2)
            d_col1.metric("Data Drift (PSI)", f"{report.data_drift:.4f}")
            d_col2.metric("Critical Drift", "YES" if report.has_critical_drift else "NO")
            if report.alerts:
                for a in report.alerts:
                    st.error(a)
            if report.warnings:
                for w in report.warnings:
                    st.warning(w)
            if report.feature_drift:
                drift_df = pd.DataFrame(list(report.feature_drift.items()), columns=["feature", "psi"])
                drift_df = drift_df.sort_values("psi", ascending=False).head(20)
                st.bar_chart(drift_df, x="feature", y="psi")
        except Exception as exc:
            st.error(f"Drift detection failed: {exc}")

    with tab5:
        st.subheader("Hyperparameter Optimization")
        st.info("Run optimization from CLI: `python main.py --optimize --symbol TICKER`")
        from advanced_ai.adaptive_strategy import AdaptiveStrategy
        from advanced_ai.market_regime import Regime
        st.subheader("Adaptive Strategy — Regime Adjustments")
        adj_data = AdaptiveStrategy.all_adjustments()
        adj_df = pd.DataFrame([
            {"regime": k, **{kk: vv for kk, vv in v.items() if kk != "regime"}}
            for k, v in adj_data.items()
        ])
        st.dataframe(adj_df, use_container_width=True)


# ===========================================================================
# Router
# ===========================================================================

def autonomous_trading_page() -> None:
    st.header("Autonomous Trading — Phase 12")
    st.caption("Set capital and risk profile. The system handles everything else automatically.")

    # ── Configuration ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        capital = st.number_input(
            "Total Capital (₹)",
            min_value=10_000,
            max_value=10_000_000,
            value=50_000,
            step=5_000,
            help="How much capital to trade with (simulation mode by default)",
        )
    with col2:
        risk_profile = st.selectbox(
            "Risk Profile",
            ["conservative", "moderate", "aggressive"],
            index=1,
        )

    sim_mode = st.checkbox("Simulation Mode (no real orders)", value=True)

    # Session-state engine holder
    if "autonomous_engine" not in st.session_state:
        st.session_state.autonomous_engine = None
    if "autonomous_logs" not in st.session_state:
        st.session_state.autonomous_logs = []

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Autonomous Engine", type="primary"):
            try:
                from autonomous.autonomous_engine import AutonomousEngine, AutonomousConfig
                cfg = AutonomousConfig(
                    total_capital=float(capital),
                    risk_profile=risk_profile,
                    simulation_mode=sim_mode,
                    enforce_market_hours=False,   # dashboard: always allow ticks
                    poll_interval_seconds=300,
                )
                engine = AutonomousEngine(cfg)
                engine.start()
                st.session_state.autonomous_engine = engine
                st.success(f"Engine started — ₹{capital:,.0f} | {risk_profile} | sim={sim_mode}")
            except Exception as exc:
                st.error(f"Failed to start engine: {exc}")

    with col_stop:
        if st.button("Stop Engine"):
            if st.session_state.autonomous_engine:
                st.session_state.autonomous_engine.stop()
                st.session_state.autonomous_engine = None
                st.warning("Engine stopped.")
            else:
                st.info("No engine running.")

    # ── Status ────────────────────────────────────────────────────────
    engine = st.session_state.autonomous_engine
    if engine:
        status = engine.status()
        st.divider()
        st.subheader("Engine Status")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", "RUNNING" if status["running"] else "STOPPED")
        m2.metric("Portfolio Value", f"₹{status['total_value']:,.0f}")
        m3.metric("Daily P&L", f"₹{status['daily_pnl']:+,.0f}")
        m4.metric("Market Regime", status["current_regime"])

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Open Positions", status["open_positions"])
        m6.metric("Last Scan", f"{status['last_scan_count']} candidates")
        m7.metric("Model Accuracy", f"{status['model_accuracy']:.1f}%")
        m8.metric("Drift PSI", f"{status['model_drift_psi']:.3f}")

        if status["halted"]:
            st.error(f"ENGINE HALTED: {status['halt_reason']}")
            if st.button("Reset Safety & Resume"):
                engine.reset_safety()
                engine.start()
                st.success("Engine resumed.")

        st.caption(f"Ticks: {status['tick_count']}  |  Last: {status['last_tick_at']}  |  Uptime: {status['uptime_seconds']:.0f}s")

        # Manual single tick for testing
        if st.button("Run One Tick Now"):
            with st.spinner("Running tick..."):
                tick_result = engine.run_once()
            st.json(tick_result)
    else:
        st.info("Configure capital and risk profile above, then click Start.")

    # ── Risk Profile Info ─────────────────────────────────────────────
    with st.expander("Risk Profile Details"):
        try:
            from autonomous.risk_profile import RISK_PROFILES
            prof = RISK_PROFILES[risk_profile]
            info = {
                "max_positions": prof.max_positions,
                "risk_per_trade_pct": f"{prof.risk_per_trade_pct}%",
                "min_ai_confidence": f"{prof.min_ai_confidence}%",
                "stop_loss_atr_multiplier": prof.stop_loss_atr_multiplier,
                "take_profit_atr_multiplier": prof.take_profit_atr_multiplier,
                "daily_loss_limit_pct": f"{prof.daily_loss_limit_pct}%",
                "max_drawdown_pct": f"{prof.max_drawdown_pct}%",
                "cash_reserve_pct": f"{prof.cash_reserve_pct}%",
                "reports": f"daily={prof.report_daily} weekly={prof.report_weekly} monthly={prof.report_monthly}",
            }
            st.json(info)
        except Exception as exc:
            st.error(f"Could not load profile: {exc}")


def main() -> None:
    st.title("AI Trading System")
    auto_refresh_control()
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "News",
            "Sentiment",
            "Market Intelligence",
            "Stock Analysis",
            "AI Models",
            "Predictions",
            "─── Phase 5 ───",
            "Backtesting",
            "Strategy Comparison",
            "Portfolio Analysis",
            "─── Phase 6-8 ───",
            "Paper Trading",
            "Risk Dashboard",
            "Broker Status",
            "Execution Monitor",
            "─── Phase 9-11 ───",
            "Live Trading",
            "System Health",
            "AI Analytics",
            "─── Phase 12 ───",
            "Autonomous Trading",
        ],
    )
    if page == "Overview":
        overview_page()
    elif page == "News":
        news_page()
    elif page == "Sentiment":
        sentiment_page()
    elif page == "Market Intelligence":
        market_intelligence_page()
    elif page == "Stock Analysis":
        stock_analysis_page()
    elif page == "AI Models":
        ai_models_page()
    elif page == "Predictions":
        predictions_page()
    elif page == "Backtesting":
        backtesting_page()
    elif page == "Strategy Comparison":
        strategy_comparison_page()
    elif page == "Portfolio Analysis":
        portfolio_analysis_page()
    elif page == "Paper Trading":
        paper_trading_page()
    elif page == "Risk Dashboard":
        risk_dashboard_page()
    elif page == "Broker Status":
        broker_status_page()
    elif page == "Execution Monitor":
        execution_monitor_page()
    elif page == "Live Trading":
        live_trading_page()
    elif page == "System Health":
        system_health_page()
    elif page == "AI Analytics":
        ai_analytics_page()
    elif page == "Autonomous Trading":
        autonomous_trading_page()
    # separator entries are no-ops


if __name__ == "__main__":
    main()
