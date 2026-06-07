"""
Live Execution Engine — Phase 9 orchestrator.

Workflow per tick
-----------------
1. Fetch latest market data (price + features)
2. Generate AI prediction
3. Risk validate the signal
4. Execute trade via TradeExecutor
5. Monitor open positions via TradeMonitor (exits)
6. Log results to TradeJournal

All trading defaults to SIMULATION_MODE=True (controlled by the broker adapter).
"""
from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime, time

import pandas as pd

from broker.base_broker import BaseBroker
from execution.trade_executor import TradeExecutor, TradeJournal, TradeRecord
from execution.trade_monitor import MonitorConfig, TradeMonitor
from features.feature_engineering import FeatureEngineeringService
from prediction.prediction_engine import PredictionEngine
from risk_management.risk_engine import RiskConfig, RiskEngine


# NSE market hours (IST)
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)


@dataclass
class LiveEngineConfig:
    poll_interval_seconds: int = 60
    min_ai_confidence: float = 65.0          # % — below this → HOLD
    max_open_positions: int = 3
    simulation_mode: bool = True             # always default True
    dry_run: bool = False                    # log only, no orders
    enforce_market_hours: bool = True
    symbols: list[str] = field(default_factory=lambda: ["RELIANCE.NS"])
    journal_path: str = "reports/trade_journal.csv"
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)


class MarketHoursEngine:
    """NSE market hours guard."""

    # Known NSE holidays 2025 (add more as needed)
    NSE_HOLIDAYS_2025: frozenset[str] = frozenset({
        "2025-01-26", "2025-02-19", "2025-03-14", "2025-03-31",
        "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
        "2025-08-15", "2025-10-02", "2025-10-20", "2025-10-23",
        "2025-11-05", "2025-11-14", "2025-12-25",
    })

    @classmethod
    def is_market_open(cls) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:                                  # weekend
            return False
        if now.strftime("%Y-%m-%d") in cls.NSE_HOLIDAYS_2025:  # holiday
            return False
        t = now.time()
        return _MARKET_OPEN <= t <= _MARKET_CLOSE


class LiveExecutionEngine:
    """
    End-to-end live (or simulated) trading engine.

    Usage
    -----
    engine = LiveExecutionEngine(broker, risk_engine, prediction_engine, config)
    engine.run()            # blocking loop
    engine.run_once()       # single tick (for testing / scheduling)
    engine.stop()
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_engine: RiskEngine | None = None,
        prediction_engine: PredictionEngine | None = None,
        config: LiveEngineConfig | None = None,
    ) -> None:
        self.broker = broker
        self.config = config or LiveEngineConfig()
        self.risk = risk_engine or RiskEngine(RiskConfig())
        self.pred_engine = prediction_engine or PredictionEngine()
        self.journal = TradeJournal(self.config.journal_path)
        self.executor = TradeExecutor(broker, self.risk, self.journal)
        self.monitor = TradeMonitor(self.executor, self.config.monitor_config)
        self._running = False
        self._open_symbols: set[str] = set()   # symbols with active positions
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "LiveExecutionEngine initialised",
            extra={"simulation": self.config.simulation_mode, "symbols": self.config.symbols},
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking poll loop. Call stop() from another thread to exit."""
        self._running = True
        self.logger.info("LiveExecutionEngine loop started")
        while self._running:
            try:
                self.run_once()
            except Exception as exc:
                self.logger.error("Engine tick error: %s", exc, exc_info=True)
            time_module.sleep(self.config.poll_interval_seconds)
        self.logger.info("LiveExecutionEngine loop stopped")

    def stop(self) -> None:
        self._running = False

    def run_once(self, ohlcv_override: pd.DataFrame | None = None) -> dict:
        """
        Single execution tick. Returns summary dict.

        `ohlcv_override` bypasses live data fetch (used in backtests / tests).
        """
        if self.config.enforce_market_hours and not MarketHoursEngine.is_market_open():
            self.logger.info("Market closed — skipping tick")
            return {"status": "market_closed"}

        results = {"signals": [], "exits": [], "errors": []}

        # 1. Check exits for all open positions
        exits = self.monitor.check_all(
            price_fn=self._get_price,
            ai_signal_fn=None,   # will be wired after feature generation
        )
        results["exits"] = exits

        # 2. Iterate symbols for new signals
        for symbol in self.config.symbols:
            if len(self._open_symbols) >= self.config.max_open_positions:
                break
            if symbol in self._open_symbols:
                continue
            try:
                result = self._process_symbol(symbol, ohlcv_override)
                if result:
                    results["signals"].append(result)
            except Exception as exc:
                results["errors"].append({"symbol": symbol, "error": str(exc)})
                self.logger.warning("Symbol processing error: %s — %s", symbol, exc)

        # Sync open symbols
        self._open_symbols = {t.symbol for t in self.journal.open_trades()}
        return results

    # ------------------------------------------------------------------
    # Per-symbol processing
    # ------------------------------------------------------------------

    def _process_symbol(self, symbol: str, ohlcv_override: pd.DataFrame | None) -> dict | None:
        # Fetch / use provided OHLCV
        ohlcv = ohlcv_override if ohlcv_override is not None else self._fetch_ohlcv(symbol)
        if ohlcv is None or ohlcv.empty:
            return None

        # Feature engineering
        try:
            features, _ = FeatureEngineeringService().create_feature_dataset(ohlcv)
        except Exception as exc:
            self.logger.warning("Feature engineering failed for %s: %s", symbol, exc)
            return None

        # AI prediction
        try:
            pred = self.pred_engine.predict(features.tail(1))
        except Exception as exc:
            self.logger.warning("Prediction failed for %s: %s", symbol, exc)
            return None

        action = pred.action
        confidence = pred.confidence  # 0-100

        if confidence < self.config.min_ai_confidence:
            self.logger.info("Low confidence — HOLD", extra={"symbol": symbol, "confidence": confidence})
            return {"symbol": symbol, "action": "HOLD", "confidence": confidence, "reason": "low_confidence"}

        if action == "HOLD":
            return {"symbol": symbol, "action": "HOLD", "confidence": confidence}

        # Current price
        price = self._get_price(symbol)
        if not price:
            latest = features.iloc[-1]
            price = float(latest.get("close", 0))
        if not price:
            return None

        # ATR for sizing / stops
        atr: float | None = None
        if "atr_14" in features.columns:
            atr = float(features["atr_14"].iloc[-1])

        # Risk validation
        portfolio_value = self._portfolio_value()
        cash = self._cash()
        validation = self.risk.validate_trade(
            signal=action,
            symbol=symbol,
            price=price,
            quantity=self.risk.size_position(symbol, price, capital=portfolio_value, atr=atr),
            ai_confidence=confidence,
            portfolio_cash=cash,
            portfolio_total_value=portfolio_value,
            open_position_count=len(self._open_symbols),
        )

        if not validation.approved:
            self.logger.info("Trade blocked by risk: %s", validation.reasons)
            return {"symbol": symbol, "action": action, "blocked": True, "reasons": validation.reasons}

        if self.config.dry_run:
            self.logger.info("[DRY-RUN] Would %s %s @ %.2f (conf=%.1f%%)", action, symbol, price, confidence)
            return {"symbol": symbol, "action": action, "price": price, "dry_run": True}

        # Execute
        if action == "BUY":
            record = self.executor.execute_buy(
                symbol, price,
                ai_confidence=confidence,
                risk_score=0.0,
                reason=f"ai:{pred.model_name}",
                atr=atr,
            )
            if record:
                return {"symbol": symbol, "action": "BUY", "trade_id": record.trade_id,
                        "price": price, "quantity": record.quantity, "confidence": confidence}

        elif action == "SELL":
            # Close existing position if open
            open_for_symbol = [t for t in self.journal.open_trades() if t.symbol == symbol]
            for open_trade in open_for_symbol:
                pnl = self.executor.execute_sell(open_trade, price, reason=f"ai_sell:{pred.model_name}")
                return {"symbol": symbol, "action": "SELL", "trade_id": open_trade.trade_id,
                        "price": price, "pnl": pnl, "confidence": confidence}

        return None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_price(self, symbol: str) -> float:
        try:
            price = self.broker.get_price(symbol)
            return float(price) if price else 0.0
        except Exception:
            return 0.0

    def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        """Try to load cached historical CSV; live tick will be added later."""
        from pathlib import Path
        csv_path = Path("data/historical") / f"{symbol.replace('.', '_')}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    def _portfolio_value(self) -> float:
        try:
            return float(self.broker.get_balance().get("total_value", 100_000.0))
        except Exception:
            return 100_000.0

    def _cash(self) -> float:
        try:
            return float(self.broker.get_balance().get("cash", 100_000.0))
        except Exception:
            return 100_000.0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        open_trades = self.journal.open_trades()
        return {
            "running": self._running,
            "simulation_mode": self.config.simulation_mode,
            "open_positions": len(open_trades),
            "today_pnl": round(self.journal.today_pnl(), 2),
            "total_closed": len(self.journal.closed_trades()),
            "open_trades": [t.to_dict() for t in open_trades],
        }
