from __future__ import annotations

import argparse
from datetime import datetime
import logging

from ai_trading_system.config.settings import get_settings
from ai_trading_system.data_collection.historical import HistoricalDataCollector
from ai_trading_system.data_collection.live import LiveDataCollector
from ai_trading_system.data_collection.providers import YahooFinanceProvider
from ai_trading_system.data_collection.validators import OHLCVValidator
from ai_trading_system.database.connection import SQLiteConnectionManager
from ai_trading_system.database.repositories import MarketDataRepository
from ai_trading_system.database.schema import DatabaseInitializer
from ai_trading_system.utils.logging import configure_logging
from news.news_collector import (
    AlphaVantageNewsProvider,
    MarketauxNewsProvider,
    NewsArticle,
    NewsCollector,
    RSSNewsProvider,
    YahooFinanceNewsProvider,
)
from news.news_repository import NewsRepository
from sentiment.sentiment_engine import SentimentEngine
from sentiment.sentiment_repository import SentimentRepository


def build_services() -> tuple[HistoricalDataCollector, LiveDataCollector]:
    settings = get_settings()
    configure_logging(settings.log_level)
    connection_manager = SQLiteConnectionManager(settings.database_url)
    DatabaseInitializer(connection_manager).initialize()
    repository = MarketDataRepository(connection_manager)
    provider = YahooFinanceProvider(settings.retry_attempts, settings.retry_wait_seconds)
    return (
        HistoricalDataCollector(provider, repository, OHLCVValidator(), settings.historical_data_dir),
        LiveDataCollector(provider, repository, settings.live_data_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("--init-db", action="store_true", help="Initialize SQLite schema")
    parser.add_argument("--collect-history", action="store_true", help="Collect historical data")
    parser.add_argument("--collect-live", action="store_true", help="Collect one live quote snapshot")
    parser.add_argument("--collect-news", action="store_true", help="Collect financial news and persist it")
    parser.add_argument("--analyze-news", action="store_true", help="Analyze latest persisted news sentiment")
    parser.add_argument("--train-models", action="store_true", help="Train Random Forest and XGBoost models from a CSV")
    parser.add_argument("--predict", action="store_true", help="Generate latest prediction from a CSV using the latest registered model")
    parser.add_argument("--backtest", action="store_true", help="Run a strategy backtest on a CSV file")
    parser.add_argument("--portfolio-backtest", action="store_true", help="Run portfolio backtest across multiple symbols")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward testing on a CSV file")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation on a CSV file")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare all strategies on a CSV file")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="NSE Yahoo Finance symbol")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols for portfolio backtest")
    parser.add_argument("--years", type=int, default=1, help="Years of historical data, 1 to 5")
    parser.add_argument("--news-limit", type=int, default=25, help="Maximum news articles to collect or analyze")
    parser.add_argument("--data-file", default="", help="OHLCV CSV path for training, prediction, or backtesting")
    parser.add_argument("--strategy", default="indicator", help="Strategy for backtest: indicator | macd | bollinger | ai | hybrid")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital for backtest")
    parser.add_argument("--risk-pct", type=float, default=2.0, help="Risk per trade as %% of capital")
    parser.add_argument("--train-bars", type=int, default=252, help="Training bars for walk-forward")
    parser.add_argument("--test-bars", type=int, default=63, help="Test bars per walk-forward window")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of Monte Carlo simulations")
    # Phase 6-8 arguments
    parser.add_argument("--paper-trade", action="store_true", help="Run paper trading engine (simulation only)")
    parser.add_argument("--risk-check", action="store_true", help="Validate a trade via the risk engine")
    parser.add_argument("--broker-status", action="store_true", help="Show broker connection status and balance")
    parser.add_argument("--broker", default="paper", help="Broker name: paper | zerodha | dhan | angelone | groww")
    parser.add_argument("--side", default="BUY", help="Order side for risk-check: BUY | SELL")
    parser.add_argument("--price", type=float, default=0.0, help="Price for risk-check or paper trade order")
    parser.add_argument("--qty", type=int, default=0, help="Quantity for risk-check or paper trade order")
    # Phase 9-11 arguments
    parser.add_argument("--live-engine", action="store_true", help="Run live execution engine (simulation by default)")
    parser.add_argument("--ticks", type=int, default=1, help="Number of ticks to run (--live-engine)")
    parser.add_argument("--detect-regime", action="store_true", help="Detect market regime from historical CSV")
    parser.add_argument("--detect-drift", action="store_true", help="Detect model/feature drift")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--system-health", action="store_true", help="Print system health snapshot")
    parser.add_argument("--backup", action="store_true", help="Create backup of DB, models, logs")

    # Phase 12 — Autonomous engine
    parser.add_argument("--autonomous", action="store_true", help="Run fully autonomous trading engine")
    parser.add_argument("--risk-profile", default="moderate", choices=["conservative", "moderate", "aggressive"],
                        help="Risk profile for autonomous mode")
    parser.add_argument("--ticks-autonomous", type=int, default=1, help="Ticks to run in autonomous mode (0=infinite)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    historical_collector, live_collector = build_services()

    if args.init_db:
        logger.info("Database initialized")
    if args.collect_history:
        rows = historical_collector.collect_symbol(args.symbol, years=args.years)
        logger.info("Historical collection complete", extra={"symbol": args.symbol, "rows": rows})
    if args.collect_live:
        quote = live_collector.collect_symbol(args.symbol)
        logger.info("Live quote collection complete", extra={"quote": quote})
    if args.collect_news or args.analyze_news:
        connection_manager = SQLiteConnectionManager(settings.database_url)
        news_repository = NewsRepository(connection_manager)
        sentiment_repository = SentimentRepository(connection_manager)
        if args.collect_news:
            collector = NewsCollector(
                [
                    AlphaVantageNewsProvider(settings.alpha_vantage_api_key or ""),
                    MarketauxNewsProvider(settings.marketaux_api_key or ""),
                    YahooFinanceNewsProvider(),
                    RSSNewsProvider(list(settings.rss_feed_urls)),
                ]
            )
            articles = collector.collect([args.symbol], limit=args.news_limit)
            inserted = news_repository.save_many(articles)
            logger.info("News collection complete", extra={"articles": len(articles), "inserted": inserted})
        if args.analyze_news:
            latest_articles = news_repository.latest(args.news_limit)
            articles = [
                NewsArticle(
                    headline=row["headline"],
                    summary=row.get("summary") or "",
                    source=row.get("source") or "",
                    url=row.get("url") or "",
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    ticker=row.get("ticker") or args.symbol,
                    category=row.get("category") or "market_news",
                )
                for row in latest_articles
            ]
            results = SentimentEngine().analyze_many(articles)
            inserted = sentiment_repository.save_many(results)
            logger.info("Sentiment analysis complete", extra={"results": len(results), "inserted": inserted})
    if args.train_models:
        if not args.data_file:
            raise ValueError("--data-file is required for --train-models")
        import pandas as pd

        from training.train_pipeline import TrainingPipeline

        result = TrainingPipeline().run(pd.read_csv(args.data_file))
        logger.info("Model training complete", extra={"result": result})
        print(result)
    if args.predict:
        if not args.data_file:
            raise ValueError("--data-file is required for --predict")
        import pandas as pd

        from prediction.prediction_engine import PredictionEngine
        from training.dataset_builder import DatasetBuilder

        dataset = DatasetBuilder().build(pd.read_csv(args.data_file))
        result = PredictionEngine().predict(dataset.tail(1))
        output = pd.DataFrame([result.__dict__])
        output.to_csv("reports/latest_prediction.csv", index=False)
        output.to_html("reports/latest_prediction.html", index=False)
        print(f"{result.action}\nConfidence: {result.confidence}%\nModel: {result.model_name}")

    # ------------------------------------------------------------------
    # Phase 5 — Backtesting commands
    # ------------------------------------------------------------------
    if args.backtest or args.compare_strategies or args.walk_forward or args.monte_carlo:
        if not args.data_file:
            raise ValueError("--data-file is required for backtesting commands")
        import pandas as pd
        from features.feature_engineering import FeatureEngineeringService
        from backtesting.engine import BacktestConfig
        from backtesting.order_simulator import OrderConfig
        from backtesting.strategy_tester import (
            StrategyTester, IndicatorStrategy, MACDStrategy,
            BollingerBandStrategy, AIPredictionStrategy, HybridStrategy,
        )
        from backtesting.results import EquityCurveGenerator
        from reports.report_generator import ReportGenerator
        from pathlib import Path

        raw = pd.read_csv(args.data_file)
        features, _ = FeatureEngineeringService().create_feature_dataset(raw)
        cfg = BacktestConfig(starting_capital=args.capital, risk_per_trade_pct=args.risk_pct)
        Path("reports").mkdir(exist_ok=True)

        _STRATEGY_MAP = {
            "indicator": IndicatorStrategy(),
            "macd": MACDStrategy(),
            "bollinger": BollingerBandStrategy(),
            "ai": AIPredictionStrategy(),
            "hybrid": HybridStrategy(AIPredictionStrategy(), IndicatorStrategy()),
        }

        if args.backtest:
            strategy = _STRATEGY_MAP.get(args.strategy.lower(), IndicatorStrategy())
            tester = StrategyTester(cfg)
            result = tester.test(features, strategy, args.strategy, args.symbol)
            chart_paths = EquityCurveGenerator().generate_charts(result.equity_df, result.metrics, args.strategy)
            report_paths = ReportGenerator().backtest_report(result, chart_paths, args.symbol)
            print(result.metrics.summary_str())
            print(f"HTML report: {report_paths.get('html')}")

        if args.compare_strategies:
            from backtesting.strategy_comparison import StrategyComparison
            tester = StrategyTester(cfg)
            results = tester.test_all_strategies(features, args.symbol)
            comparison = StrategyComparison().compare(results)
            print(comparison.summary())
            table = comparison.comparison_table.drop(columns=["_result"], errors="ignore")
            report_paths = ReportGenerator().strategy_comparison_report(table, comparison.winner)
            print(f"HTML report: {report_paths.get('html')}")

        if args.walk_forward:
            from backtesting.walk_forward import WalkForwardTester
            wf = WalkForwardTester(cfg)
            strategy_fn = _STRATEGY_MAP.get(args.strategy.lower(), IndicatorStrategy())
            wf_result = wf.run(features, lambda: _STRATEGY_MAP.get(args.strategy.lower(), IndicatorStrategy()),
                               train_bars=args.train_bars, test_bars=args.test_bars, symbol=args.symbol)
            print(wf_result.aggregated_metrics.summary_str())
            summary_table = wf_result.summary_table()
            summary_table.to_csv("reports/walk_forward_summary.csv", index=False)
            print("Walk-forward summary: reports/walk_forward_summary.csv")

        if args.monte_carlo:
            from backtesting.monte_carlo import MonteCarloSimulator
            # Need an equity curve — run indicator strategy first
            tester = StrategyTester(cfg)
            bt_result = tester.test(features, IndicatorStrategy(), "Indicator", args.symbol)
            mc = MonteCarloSimulator(n_simulations=args.simulations)
            mc_result = mc.run(bt_result.equity_df, starting_capital=args.capital)
            print(mc_result.summary_str())
            mc.generate_chart(mc_result, "reports/monte_carlo.png")
            mc_summary = {
                "expected_value": mc_result.expected_value,
                "worst_case": mc_result.worst_case,
                "best_case": mc_result.best_case,
                "prob_profit": mc_result.prob_profit,
                "expected_max_drawdown": mc_result.expected_max_drawdown,
            }
            ReportGenerator().risk_report(bt_result.metrics, mc_summary)

    if args.portfolio_backtest:
        import pandas as pd
        from features.feature_engineering import FeatureEngineeringService
        from backtesting.engine import BacktestConfig
        from backtesting.portfolio_backtester import PortfolioBacktester
        from backtesting.strategy_tester import IndicatorStrategy
        from reports.report_generator import ReportGenerator
        from pathlib import Path

        symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else [args.symbol]
        datasets = {}
        for sym in symbols_list:
            csv_path = Path("data/historical") / f"{sym.replace('.', '_')}.csv"
            if csv_path.exists():
                raw = pd.read_csv(csv_path)
                try:
                    features, _ = FeatureEngineeringService().create_feature_dataset(raw)
                    datasets[sym] = features
                except Exception:
                    logger.warning(f"Skipping {sym} — feature engineering failed")
            else:
                logger.warning(f"No historical data for {sym}")

        if not datasets:
            raise RuntimeError("No datasets loaded for portfolio backtest.")

        cfg = BacktestConfig(starting_capital=args.capital, risk_per_trade_pct=args.risk_pct)
        backtester = PortfolioBacktester(cfg, strategy_factory=lambda _s: IndicatorStrategy())
        port_result = backtester.run(datasets)
        print(port_result.combined_metrics.summary_str())
        print(port_result.summary_table().to_string(index=False))
        Path("reports").mkdir(exist_ok=True)
        report_paths = ReportGenerator().portfolio_report(
            port_result.symbol_results, port_result.combined_metrics, port_result.capital_allocation
        )
        print(f"HTML report: {report_paths.get('html')}")

    # ------------------------------------------------------------------
    # Phase 6 — Paper Trading
    # ------------------------------------------------------------------
    if args.paper_trade:
        from paper_trading.paper_broker import BrokerConfig, PaperBroker
        from paper_trading.paper_reports import PaperReportGenerator

        cfg_paper = BrokerConfig(starting_capital=args.capital)
        pb = PaperBroker(cfg_paper)
        pb.enable_trading(True)

        symbol = args.symbol
        price = args.price or 2500.0
        qty = args.qty or 10
        pb.update_prices({symbol: price})

        if args.side.upper() == "BUY":
            order = pb.place_buy(symbol, qty, price)
        else:
            order = pb.place_sell(symbol, qty, price)

        logger.info("Paper order submitted", extra={"order_id": order.order_id, "status": order.status.value})
        print(f"Order ID: {order.order_id}  Status: {order.status.value}")
        snap = pb.snapshot()
        bal = snap["balance"]
        print(f"Cash: ₹{bal['cash']:,.2f}  Total Value: ₹{bal['total_value']:,.2f}")

        gen = PaperReportGenerator()
        paths = gen.daily_report(pb.portfolio)
        print(f"Paper trading report: {paths.get('html', '')}")

    # ------------------------------------------------------------------
    # Phase 7 — Risk Check
    # ------------------------------------------------------------------
    if args.risk_check:
        from risk_management.risk_engine import RiskConfig, RiskEngine

        risk_cfg = RiskConfig()
        engine = RiskEngine(risk_cfg)
        price = args.price or 2500.0
        qty = args.qty or (engine.size_position(args.symbol, price, capital=args.capital) or 10)

        result = engine.validate_trade(
            signal=args.side.upper(),
            symbol=args.symbol,
            price=price,
            quantity=int(qty),
            portfolio_cash=args.capital,
            portfolio_total_value=args.capital,
        )
        print(f"Trade Approved: {result.approved}")
        if result.reasons:
            print("Rejection reasons:")
            for r in result.reasons:
                print(f"  - {r}")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  ! {w}")

        stop = engine.stop_loss(price)
        target = engine.take_profit(price)
        print(f"Suggested stop-loss: ₹{stop:,.2f}  take-profit: ₹{target:,.2f}")

    # ------------------------------------------------------------------
    # Phase 8 — Broker Status
    # ------------------------------------------------------------------
    if args.broker_status:
        from broker.broker_factory import BrokerFactory

        broker = BrokerFactory.create(args.broker)
        connected = broker.login()
        mode = "SIMULATION" if getattr(broker, "SIMULATION_MODE", True) else "LIVE"
        print(f"Broker: {broker.broker_name()}  Mode: {mode}  Connected: {connected}")
        try:
            bal = broker.get_balance()
            print(f"Cash: ₹{bal.get('cash', 0):,.2f}  Total Value: ₹{bal.get('total_value', 0):,.2f}")
        except Exception as exc:
            print(f"Could not fetch balance: {exc}")
        try:
            positions = broker.get_positions()
            print(f"Open positions: {len(positions)}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 9 — Live Execution Engine
    # ------------------------------------------------------------------
    if args.live_engine:
        from broker.broker_factory import BrokerFactory
        from execution.live_execution_engine import LiveEngineConfig, LiveExecutionEngine

        broker = BrokerFactory.create(args.broker)
        broker.login()
        eng_cfg = LiveEngineConfig(
            simulation_mode=True,
            symbols=[s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else [args.symbol],
            enforce_market_hours=False,  # allow testing outside hours
        )
        eng = LiveExecutionEngine(broker, config=eng_cfg)
        for tick in range(args.ticks):
            print(f"Tick {tick + 1}/{args.ticks}")
            result = eng.run_once()
            print(f"  Signals: {len(result.get('signals', []))}  Exits: {len(result.get('exits', []))}  Errors: {len(result.get('errors', []))}")
        status = eng.status()
        print(f"\nStatus: {status}")

    # ------------------------------------------------------------------
    # Phase 11 — Market Regime Detection
    # ------------------------------------------------------------------
    if args.detect_regime:
        if not args.data_file:
            raise ValueError("--data-file is required for --detect-regime")
        import pandas as pd
        from advanced_ai.market_regime import MarketRegimeDetector
        raw = pd.read_csv(args.data_file)
        result = MarketRegimeDetector().detect(raw)
        print(f"Regime: {result.regime.value}  Confidence: {result.confidence:.0%}")
        print(f"Volatility: {result.volatility_pct:.1f}%  ADX: {result.adx:.1f}")
        print(f"Trend Strength: {result.trend_strength:+.4f}")

    # Phase 11 — Drift Detection
    if args.detect_drift:
        if not args.data_file:
            raise ValueError("--data-file is required for --detect-drift")
        import pandas as pd
        from features.feature_engineering import FeatureEngineeringService
        from advanced_ai.drift_detector import DriftDetector
        raw = pd.read_csv(args.data_file)
        features, _ = FeatureEngineeringService().create_feature_dataset(raw)
        mid = len(features) // 2
        detector = DriftDetector()
        detector.fit_reference(features.iloc[:mid])
        report = detector.detect(features.iloc[mid:])
        print(f"Data Drift (PSI): {report.data_drift:.4f}")
        print(f"Critical Drift: {'YES' if report.has_critical_drift else 'NO'}")
        for a in report.alerts:
            print(f"  ALERT: {a}")
        for w in report.warnings:
            print(f"  WARN:  {w}")

    # Phase 11 — Hyperparameter Optimization
    if args.optimize:
        if not args.data_file:
            raise ValueError("--data-file is required for --optimize")
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from features.feature_engineering import FeatureEngineeringService
        from features.label_generator import LabelGenerator
        from advanced_ai.hyperparameter_optimizer import HyperparameterOptimizer

        raw = pd.read_csv(args.data_file)
        features, _ = FeatureEngineeringService().create_feature_dataset(raw)
        labeled = LabelGenerator().add_labels(features).dropna(subset=["target_next_day_up"])
        feat_cols = [c for c in labeled.columns if c not in ("date", "target_next_day_up", "target_intraday_up")]
        X = labeled[feat_cols].fillna(0)
        y = labeled["target_next_day_up"].astype(int)
        param_space = {
            "n_estimators": (50, 300),
            "max_depth": (3, 12),
            "min_samples_leaf": (1, 10),
        }
        opt = HyperparameterOptimizer(
            estimator_fn=lambda **kw: RandomForestClassifier(**kw, random_state=42),
            param_space=param_space,
            method="optuna",
        )
        result = opt.optimize(X, y, n_trials=20)
        print(f"Best params: {result.best_params}")
        print(f"Best CV score: {result.best_score:.4f}  ({result.method}, {result.n_trials} trials)")

    # Phase 10 — System Health
    if args.system_health:
        from monitoring.system_monitor import SystemMonitor
        snap = SystemMonitor().snapshot()
        print(f"CPU: {snap.cpu_pct:.1f}%  RAM: {snap.ram_pct:.1f}%  Disk: {snap.disk_pct:.1f}%")
        print(f"DB OK: {snap.db_ok}  Healthy: {snap.healthy}")
        if snap.warnings:
            for w in snap.warnings:
                print(f"  WARN: {w}")

    # Phase 10 — Backup
    if args.backup:
        from monitoring.backup_manager import BackupManager
        paths = BackupManager().backup_all()
        for name, path in paths.items():
            print(f"  {name}: {path}")

    if args.autonomous:
        from autonomous.autonomous_engine import AutonomousEngine, AutonomousConfig
        auto_cfg = AutonomousConfig(
            total_capital=args.capital,
            risk_profile=args.risk_profile,
            simulation_mode=True,
            enforce_market_hours=False,
            poll_interval_seconds=60,
        )
        engine = AutonomousEngine(auto_cfg)
        n = args.ticks_autonomous
        if n == 0:
            logger.info("Autonomous engine running indefinitely (Ctrl+C to stop)…")
            engine.start()
            try:
                import time as _time
                while True:
                    _time.sleep(5)
            except KeyboardInterrupt:
                engine.stop()
        else:
            logger.info("Running %d autonomous tick(s)…", n)
            for i in range(n):
                result = engine.run_once()
                print(f"Tick {i+1}: {result}")
        print(engine.status())

    if not any(
        [
            args.init_db,
            args.collect_history,
            args.collect_live,
            args.collect_news,
            args.analyze_news,
            args.train_models,
            args.predict,
            args.backtest,
            args.portfolio_backtest,
            args.walk_forward,
            args.monte_carlo,
            args.compare_strategies,
            args.paper_trade,
            args.risk_check,
            args.broker_status,
            args.live_engine,
            args.detect_regime,
            args.detect_drift,
            args.optimize,
            args.system_health,
            args.backup,
            args.autonomous,
        ]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
