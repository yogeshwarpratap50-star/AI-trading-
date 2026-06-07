"""
Autonomous Engine — Phase 12 Main Orchestrator.

The user provides:
  * total_capital  (e.g. 50000.0)
  * risk_profile   ("conservative" | "moderate" | "aggressive")

The engine handles everything else automatically:
  1. Scan NIFTY50 for opportunities
  2. Detect market regime
  3. Allocate capital via risk profile
  4. Execute trades via LiveExecutionEngine
  5. Monitor open positions
  6. Manage models (drift / retrain)
  7. Rebalance portfolio
  8. Send periodic reports
  9. Halt on emergency conditions

SIMULATION_MODE is always True by default — no real orders ever placed
unless the user explicitly enables live mode.
"""
from __future__ import annotations

import logging
import threading
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous.capital_allocator import CapitalAllocator, AllocationPlan
from autonomous.model_manager import AutonomousModelManager
from autonomous.portfolio_rebalancer import PortfolioRebalancer
from autonomous.reporter import AutonomousReporter, ReportData
from autonomous.risk_profile import RISK_PROFILES, RiskProfile, RiskProfileConfig
from autonomous.safety_system import SafetySystem, EmergencyReason
from autonomous.stock_scanner import StockScanner
from advanced_ai.market_regime import MarketRegimeDetector, Regime
from execution.live_execution_engine import LiveEngineConfig, MarketHoursEngine


@dataclass
class AutonomousConfig:
    total_capital: float
    risk_profile: RiskProfile = "moderate"
    simulation_mode: bool = True              # always True by default
    enforce_market_hours: bool = True
    poll_interval_seconds: int = 300          # 5-minute tick by default
    data_dir: str = "data/historical"
    journal_path: str = "reports/trade_journal.csv"
    log_dir: str = "reports/autonomous"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    email_to: str = ""


@dataclass
class EngineStatus:
    running: bool = False
    halted: bool = False
    halt_reason: str = ""
    current_regime: str = "UNKNOWN"
    open_positions: int = 0
    total_value: float = 0.0
    daily_pnl: float = 0.0
    last_scan_count: int = 0
    last_tick_at: str = ""
    uptime_seconds: float = 0.0
    tick_count: int = 0
    model_accuracy: float = 0.0
    model_drift_psi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "current_regime": self.current_regime,
            "open_positions": self.open_positions,
            "total_value": round(self.total_value, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "last_scan_count": self.last_scan_count,
            "last_tick_at": self.last_tick_at,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "tick_count": self.tick_count,
            "model_accuracy": round(self.model_accuracy * 100, 1),
            "model_drift_psi": round(self.model_drift_psi, 4),
        }


class AutonomousEngine:
    """
    Top-level autonomous trading orchestrator.

    Usage
    -----
    engine = AutonomousEngine(AutonomousConfig(
        total_capital=50000.0,
        risk_profile="moderate",
    ))
    engine.start()          # non-blocking, runs in background thread
    engine.stop()           # graceful shutdown
    engine.status()         # EngineStatus dict
    engine.run_once()       # single tick (for testing)
    """

    def __init__(self, config: AutonomousConfig) -> None:
        self.config = config
        self.profile: RiskProfileConfig = RISK_PROFILES[config.risk_profile]
        self.logger = logging.getLogger(__name__)

        # Core subsystems
        self.scanner = StockScanner(data_dir=config.data_dir)
        self.allocator = CapitalAllocator(self.profile)
        self.safety = SafetySystem(self.profile, config.total_capital)
        self.regime_detector = MarketRegimeDetector()
        self.rebalancer = PortfolioRebalancer(self.profile)
        self.model_manager = AutonomousModelManager(data_dir=config.data_dir)

        # Alert channels (optional)
        telegram_alerter = self._build_telegram()
        email_alerter = self._build_email()
        self.reporter = AutonomousReporter(self.profile, telegram_alerter, email_alerter)

        # Paper broker for portfolio tracking
        self._broker = self._build_broker()

        # State
        self._status = EngineStatus(total_value=config.total_capital)
        self._running = False
        self._thread: threading.Thread | None = None
        self._started_at: datetime | None = None
        self._current_regime: Regime = Regime.SIDEWAYS

        # Log dir
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start autonomous engine in a background thread."""
        if self._running:
            self.logger.warning("Engine already running")
            return
        self._running = True
        self._started_at = datetime.now()
        self._status.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AutonomousEngine")
        self._thread.start()
        self.logger.info(
            "Autonomous engine started | capital=₹%.0f profile=%s simulation=%s",
            self.config.total_capital, self.config.risk_profile, self.config.simulation_mode,
        )

    def stop(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Stopping autonomous engine...")
        self._running = False
        self._status.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def status(self) -> dict[str, Any]:
        if self._started_at:
            self._status.uptime_seconds = (datetime.now() - self._started_at).total_seconds()
        return self._status.to_dict()

    def manual_stop(self) -> None:
        """Emergency halt requested by user."""
        self.safety.manual_stop()
        self.stop()

    def reset_safety(self) -> None:
        """Resume after manual halt or daily reset."""
        self.safety.reset()
        self._status.halted = False
        self._status.halt_reason = ""

    def run_once(self, ohlcv_override: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
        """
        Execute a single autonomous tick.
        Used for testing and manual single-step execution.
        """
        return self._tick(ohlcv_override=ohlcv_override)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                self.logger.exception("Autonomous tick error: %s", exc)
            time_module.sleep(self.config.poll_interval_seconds)

    def _tick(self, ohlcv_override: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
        """One full autonomous tick."""
        now = datetime.now()
        result: dict[str, Any] = {"tick_at": now.isoformat(), "actions": []}

        # --- Market hours guard ---
        if self.config.enforce_market_hours and not MarketHoursEngine.is_market_open():
            result["skipped"] = "market_closed"
            self._status.last_tick_at = now.isoformat()
            return result

        # --- Safety check ---
        snap = self._broker_snapshot()
        broker_ok = snap is not None
        safety_status = self.safety.check(broker_connected=broker_ok)
        if not safety_status.safe:
            self._status.halted = True
            self._status.halt_reason = safety_status.detail
            self._status.running = False
            self._running = False
            self.logger.critical("AUTONOMOUS HALT: %s", safety_status.detail)
            result["halted"] = safety_status.to_dict()
            return result

        # --- Detect market regime ---
        regime = self._detect_regime(ohlcv_override)
        self._current_regime = regime
        self._status.current_regime = regime.value

        # --- Scan for opportunities ---
        min_conf = self.profile.min_ai_confidence
        candidates = self.scanner.scan(min_score=30.0, top_n=self.profile.max_positions * 2)
        self._status.last_scan_count = len(candidates)
        result["scan_count"] = len(candidates)

        # --- Current portfolio state ---
        if snap:
            cash = float(snap.get("balance", {}).get("cash", self.config.total_capital))
            total_val = float(snap.get("balance", {}).get("total_value", cash))
            open_positions = snap.get("positions", [])
            self._status.open_positions = len(open_positions)
            self._status.total_value = total_val
        else:
            cash = self.config.total_capital
            open_positions = []
            total_val = cash

        # --- Capital allocation ---
        available_slots = self.profile.max_positions - len(open_positions)
        if available_slots > 0 and candidates:
            plan: AllocationPlan = self.allocator.allocate(cash, candidates[:available_slots])
            result["allocations"] = [a.to_dict() for a in plan.allocations]

            # Execute allocations
            for alloc in plan.allocations:
                if alloc.suggested_qty > 0:
                    trade_result = self._execute_buy(alloc, regime)
                    result["actions"].append(trade_result)
        else:
            result["allocations"] = []

        # --- Rebalancing ---
        if open_positions and snap:
            rebalance_actions = self.rebalancer.evaluate(
                open_positions, total_val, regime
            )
            for action in rebalance_actions:
                rb_result = self._execute_rebalance(action)
                result["actions"].append(rb_result)

        # --- Model management ---
        features_sample = self._load_features_sample()
        if features_sample is not None:
            model_cycle = self.model_manager.cycle(
                symbol="MARKET",
                features=features_sample,
            )
            self._status.model_accuracy = model_cycle.get("accuracy", 0.0)
            self._status.model_drift_psi = model_cycle.get("drift_psi", 0.0)
            result["model"] = model_cycle

        # --- Reporting ---
        if snap:
            report_data = AutonomousReporter.build_report(
                period="daily",
                portfolio_snapshot=snap,
                trade_journal=None,
                safety_system=self.safety,
                model_manager=self.model_manager,
                regime_str=regime.value,
                starting_capital=self.config.total_capital,
            )
            sent = self.reporter.maybe_send(report_data)
            if sent:
                result["reports_sent"] = sent

        # Update status
        self._status.tick_count += 1
        self._status.last_tick_at = now.isoformat()
        if snap:
            self._status.daily_pnl = float(snap.get("balance", {}).get("today_pnl", 0))

        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _detect_regime(self, ohlcv_override: dict[str, pd.DataFrame] | None) -> Regime:
        """Detect regime from NIFTY50 data (first available symbol)."""
        try:
            # Use override if provided (for testing)
            if ohlcv_override:
                ohlcv = next(iter(ohlcv_override.values()))
                result = self.regime_detector.detect(ohlcv)
                return result.regime

            # Load from data dir
            data_dir = Path(self.config.data_dir)
            for csv_path in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = [c.lower() for c in df.columns]
                    if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                        result = self.regime_detector.detect(df.tail(100))
                        return result.regime
                except Exception:
                    continue
        except Exception as exc:
            self.logger.debug("Regime detection failed: %s", exc)
        return Regime.SIDEWAYS

    def _broker_snapshot(self) -> dict[str, Any] | None:
        try:
            return self._broker.snapshot()
        except Exception as exc:
            self.logger.warning("Broker snapshot failed: %s", exc)
            return None

    def _execute_buy(self, alloc, regime: Regime) -> dict[str, Any]:
        """Execute a buy order via the paper broker."""
        try:
            if self.config.simulation_mode:
                # Simulate the buy — record in broker
                result = self._broker.place_order(
                    symbol=alloc.symbol,
                    side="BUY",
                    quantity=alloc.suggested_qty,
                    price=alloc.entry_price,
                    order_type="MARKET",
                )
                self.logger.info(
                    "[SIM] BUY %s x%d @ ₹%.2f | stop=₹%.2f target=₹%.2f",
                    alloc.symbol, alloc.suggested_qty,
                    alloc.entry_price, alloc.stop_loss, alloc.take_profit,
                )
                return {
                    "type": "buy",
                    "symbol": alloc.symbol,
                    "qty": alloc.suggested_qty,
                    "price": alloc.entry_price,
                    "stop_loss": alloc.stop_loss,
                    "take_profit": alloc.take_profit,
                    "simulation": True,
                }
        except Exception as exc:
            self.logger.warning("Buy execution failed for %s: %s", alloc.symbol, exc)
        return {"type": "buy_failed", "symbol": alloc.symbol}

    def _execute_rebalance(self, action) -> dict[str, Any]:
        """Execute a rebalance sell order."""
        try:
            if self.config.simulation_mode and action.quantity_delta > 0:
                self.logger.info(
                    "[SIM] REBALANCE %s %s x%d — %s",
                    action.action, action.symbol, action.quantity_delta, action.reason,
                )
                return {
                    "type": "rebalance",
                    "action": action.action,
                    "symbol": action.symbol,
                    "qty": action.quantity_delta,
                    "reason": action.reason,
                    "simulation": True,
                }
        except Exception as exc:
            self.logger.warning("Rebalance failed for %s: %s", action.symbol, exc)
        return {"type": "rebalance_skipped", "symbol": action.symbol}

    def _load_features_sample(self) -> pd.DataFrame | None:
        """Load a sample of recent features for model management."""
        data_dir = Path(self.config.data_dir)
        for csv_path in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.lower() for c in df.columns]
                if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                    from features.feature_engineering import FeatureEngineeringService
                    features, _ = FeatureEngineeringService().create_feature_dataset(df)
                    return features.tail(50)
            except Exception:
                pass
        return None

    def _build_broker(self):
        """Build a paper broker configured with starting capital."""
        try:
            from paper_trading.paper_broker import PaperBroker
            from paper_trading.paper_portfolio import PaperPortfolio
            from paper_trading.paper_order_manager import PaperOrderManager
            portfolio = PaperPortfolio(initial_capital=self.config.total_capital)
            order_manager = PaperOrderManager()
            broker = PaperBroker(portfolio=portfolio, order_manager=order_manager)
            return broker
        except Exception as exc:
            self.logger.warning("Could not build PaperBroker: %s — using stub", exc)
            return _StubBroker(self.config.total_capital)

    def _build_telegram(self):
        if self.config.telegram_token and self.config.telegram_chat_id:
            try:
                from alerts.telegram_alerts import TelegramAlerter
                return TelegramAlerter(
                    token=self.config.telegram_token,
                    chat_id=self.config.telegram_chat_id,
                )
            except Exception:
                pass
        return None

    def _build_email(self):
        if self.config.email_to:
            try:
                from alerts.email_alerts import EmailAlerter
                return EmailAlerter()
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Stub broker — used when PaperBroker import fails
# ---------------------------------------------------------------------------

class _StubBroker:
    def __init__(self, capital: float) -> None:
        self._capital = capital

    def place_order(self, **kwargs) -> dict:
        return {"status": "simulated", **kwargs}

    def snapshot(self) -> dict[str, Any]:
        return {
            "balance": {
                "cash": self._capital,
                "total_value": self._capital,
                "unrealized_pnl": 0.0,
                "today_pnl": 0.0,
            },
            "positions": [],
            "closed_positions": [],
            "trading_enabled": True,
        }
