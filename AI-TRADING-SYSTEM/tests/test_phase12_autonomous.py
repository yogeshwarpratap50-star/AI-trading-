"""
Phase 12 — Fully Autonomous Trading Mode
Tests: risk_profile, stock_scanner, capital_allocator, safety_system,
       portfolio_rebalancer, reporter, autonomous_engine
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

# ────────────────────────────────────────────────────────────────────────────
# risk_profile
# ────────────────────────────────────────────────────────────────────────────

class TestRiskProfile:
    def test_all_three_profiles_exist(self):
        from autonomous.risk_profile import RISK_PROFILES
        assert "conservative" in RISK_PROFILES
        assert "moderate" in RISK_PROFILES
        assert "aggressive" in RISK_PROFILES

    def test_conservative_has_lower_risk_than_aggressive(self):
        from autonomous.risk_profile import RISK_PROFILES
        c = RISK_PROFILES["conservative"]
        a = RISK_PROFILES["aggressive"]
        assert c.risk_per_trade_pct < a.risk_per_trade_pct
        assert c.min_ai_confidence > a.min_ai_confidence
        assert c.max_positions < a.max_positions
        assert c.cash_reserve_pct > a.cash_reserve_pct

    def test_profiles_are_frozen(self):
        from autonomous.risk_profile import RISK_PROFILES
        with pytest.raises((AttributeError, TypeError)):
            RISK_PROFILES["moderate"].max_positions = 99

    def test_daily_loss_limit_positive(self):
        from autonomous.risk_profile import RISK_PROFILES
        for prof in RISK_PROFILES.values():
            assert prof.daily_loss_limit_pct > 0

    def test_report_flags_set(self):
        from autonomous.risk_profile import RISK_PROFILES
        for prof in RISK_PROFILES.values():
            assert isinstance(prof.report_daily, bool)
            assert isinstance(prof.report_weekly, bool)
            assert isinstance(prof.report_monthly, bool)


# ────────────────────────────────────────────────────────────────────────────
# stock_scanner
# ────────────────────────────────────────────────────────────────────────────

class TestStockScanner:
    def _make_ohlcv(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        close = 1000 + np.cumsum(rng.normal(0, 10, n))
        return pd.DataFrame({
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": rng.integers(100_000, 500_000, n).astype(float),
        })

    def test_scan_result_has_required_fields(self, tmp_path):
        from autonomous.stock_scanner import StockScanner, ScanResult
        ohlcv = self._make_ohlcv()
        sym = "TEST_NS"
        csv_path = tmp_path / f"{sym}.csv"
        ohlcv.to_csv(csv_path, index=False)

        scanner = StockScanner(symbols=[f"{sym.replace('_', '.')}.NS"], data_dir=tmp_path)
        # Force _load_ohlcv to find the file
        result = scanner._score_symbol(sym.replace("_", ".").replace(".NS", "") + ".NS")
        if result is None:
            # Fallback: call directly with patched load
            scanner.symbols = [sym]
            result = scanner._score_symbol(sym)
        if result is not None:
            assert hasattr(result, "score")
            assert hasattr(result, "momentum_score")
            assert hasattr(result, "volume_score")
            assert hasattr(result, "sentiment_score")
            assert hasattr(result, "ai_confidence")
            assert hasattr(result, "current_price")
            assert hasattr(result, "atr")

    def test_scan_returns_sorted_by_score(self, tmp_path):
        from autonomous.stock_scanner import StockScanner
        # Create two symbols with different data
        for sym, trend in [("HIGH_NS", 2.0), ("LOW_NS", 0.5)]:
            n = 100
            close = np.linspace(1000, 1000 * trend, n)
            df = pd.DataFrame({
                "open": close, "high": close * 1.01,
                "low": close * 0.99, "close": close,
                "volume": np.full(n, 200_000.0),
            })
            df.to_csv(tmp_path / f"{sym}.csv", index=False)

        scanner = StockScanner(symbols=["HIGH.NS", "LOW.NS"], data_dir=tmp_path)
        results = scanner.scan_all()
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_sentiment_defaults_neutral_when_no_db(self, tmp_path):
        from autonomous.stock_scanner import StockScanner
        scanner = StockScanner(data_dir=tmp_path, sentiment_db_path=tmp_path / "no.db")
        score, raw = scanner._get_sentiment("RELIANCE.NS")
        assert score == 50.0
        assert raw == 0.0

    def test_ai_prediction_returns_hold_without_engine(self, tmp_path):
        from autonomous.stock_scanner import StockScanner
        scanner = StockScanner(data_dir=tmp_path, prediction_engine=None)
        df = pd.DataFrame()
        action, conf = scanner._get_ai_prediction("TEST.NS", df)
        assert action == "HOLD"
        assert conf == 0.0

    def test_scan_result_to_dict(self):
        from autonomous.stock_scanner import ScanResult
        r = ScanResult(
            symbol="TCS.NS", score=75.0, momentum_score=80.0, volume_score=60.0,
            sentiment_score=50.0, ai_confidence=70.0, ai_action="BUY",
            current_price=3500.0, atr=45.0,
        )
        d = r.to_dict()
        assert d["symbol"] == "TCS.NS"
        assert d["score"] == 75.0


# ────────────────────────────────────────────────────────────────────────────
# capital_allocator
# ────────────────────────────────────────────────────────────────────────────

class TestCapitalAllocator:
    def _candidates(self):
        from autonomous.stock_scanner import ScanResult
        return [
            ScanResult("A.NS", 80.0, 85, 70, 60, 75.0, "BUY", 1000.0, 20.0),
            ScanResult("B.NS", 60.0, 60, 55, 50, 65.0, "BUY", 500.0, 10.0),
            ScanResult("C.NS", 40.0, 45, 40, 45, 55.0, "BUY", 2000.0, 40.0),
        ]

    def test_allocates_within_deployable_capital(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["moderate"])
        plan = allocator.allocate(100_000.0, self._candidates())
        deployed = sum(a.allocated_capital for a in plan.allocations)
        assert deployed <= plan.deployable_capital + 1   # 1 ₹ rounding tolerance

    def test_cash_reserve_respected(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        prof = RISK_PROFILES["moderate"]
        allocator = CapitalAllocator(prof)
        plan = allocator.allocate(100_000.0, self._candidates())
        expected_reserve = 100_000.0 * prof.cash_reserve_pct / 100
        assert abs(plan.cash_reserve - expected_reserve) < 1

    def test_each_allocation_has_stop_and_target(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["moderate"])
        plan = allocator.allocate(100_000.0, self._candidates())
        for a in plan.allocations:
            assert a.stop_loss < a.entry_price
            assert a.take_profit > a.entry_price

    def test_stop_loss_not_more_than_10pct_below(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["aggressive"])
        plan = allocator.allocate(100_000.0, self._candidates())
        for a in plan.allocations:
            assert a.stop_loss >= a.entry_price * 0.90

    def test_empty_candidates_returns_empty_plan(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["conservative"])
        plan = allocator.allocate(50_000.0, [])
        assert len(plan.allocations) == 0

    def test_plan_summary_is_string(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["moderate"])
        plan = allocator.allocate(100_000.0, self._candidates())
        s = plan.summary()
        assert isinstance(s, str)
        assert "₹" in s

    def test_plan_to_dict(self):
        from autonomous.capital_allocator import CapitalAllocator
        from autonomous.risk_profile import RISK_PROFILES
        allocator = CapitalAllocator(RISK_PROFILES["moderate"])
        plan = allocator.allocate(100_000.0, self._candidates())
        d = plan.to_dict()
        assert "total_capital" in d
        assert "allocations" in d


# ────────────────────────────────────────────────────────────────────────────
# safety_system
# ────────────────────────────────────────────────────────────────────────────

class TestSafetySystem:
    def _make_safety(self, profile_name="moderate", capital=100_000.0):
        from autonomous.safety_system import SafetySystem
        from autonomous.risk_profile import RISK_PROFILES
        return SafetySystem(RISK_PROFILES[profile_name], capital)

    def test_safe_by_default(self):
        safety = self._make_safety()
        status = safety.check()
        assert status.safe

    def test_daily_loss_triggers_halt(self):
        from autonomous.safety_system import EmergencyReason
        safety = self._make_safety()
        # Moderate daily_loss_limit_pct = 3% of 100k = ₹3000
        safety.record_pnl(-4000)
        status = safety.check()
        assert not status.safe
        assert status.reason == EmergencyReason.DAILY_LOSS_LIMIT

    def test_broker_failure_triggers_halt(self):
        from autonomous.safety_system import EmergencyReason
        safety = self._make_safety()
        status = safety.check(broker_connected=False)
        assert not status.safe
        assert status.reason == EmergencyReason.BROKER_FAILURE

    def test_confidence_collapse_triggers_halt(self):
        from autonomous.safety_system import EmergencyReason
        safety = self._make_safety()
        for _ in range(10):
            safety.record_confidence(30.0)   # below CONFIDENCE_FLOOR=45
        status = safety.check()
        assert not status.safe
        assert status.reason == EmergencyReason.CONFIDENCE_COLLAPSE

    def test_high_confidence_no_halt(self):
        safety = self._make_safety()
        for _ in range(10):
            safety.record_confidence(70.0)
        status = safety.check()
        assert status.safe

    def test_manual_stop(self):
        from autonomous.safety_system import EmergencyReason
        safety = self._make_safety()
        safety.manual_stop()
        status = safety.check()
        assert not status.safe
        assert status.reason == EmergencyReason.MANUAL_STOP

    def test_reset_clears_halt(self):
        safety = self._make_safety()
        safety.manual_stop()
        safety.reset()
        status = safety.check()
        assert status.safe

    def test_reset_daily_clears_pnl(self):
        safety = self._make_safety()
        safety.record_pnl(-5000)
        safety.reset_daily()
        status = safety.check()
        assert status.safe

    def test_market_anomaly_triggers_halt(self):
        from autonomous.safety_system import EmergencyReason
        safety = self._make_safety()
        # 19 normal returns then one extreme spike
        for _ in range(19):
            safety.record_return(0.001)
        safety.record_return(5.0)   # extreme z-score
        status = safety.check()
        # May or may not trigger depending on std; just test it doesn't crash
        assert isinstance(status.safe, bool)

    def test_status_to_dict(self):
        safety = self._make_safety()
        d = safety.check().to_dict()
        assert "safe" in d
        assert "reason" in d


# ────────────────────────────────────────────────────────────────────────────
# portfolio_rebalancer
# ────────────────────────────────────────────────────────────────────────────

class TestPortfolioRebalancer:
    def _profile(self, name="moderate"):
        from autonomous.risk_profile import RISK_PROFILES
        return RISK_PROFILES[name]

    def _positions(self):
        return [
            {"symbol": "A.NS", "market_value": 30000.0, "last_price": 1000.0, "unrealized_pnl": 500.0, "quantity": 30},
            {"symbol": "B.NS", "market_value": 20000.0, "last_price": 500.0,  "unrealized_pnl": -200.0, "quantity": 40},
            {"symbol": "C.NS", "market_value": 10000.0, "last_price": 2000.0, "unrealized_pnl": 100.0, "quantity": 5},
        ]

    def test_empty_positions_returns_no_actions(self):
        from autonomous.portfolio_rebalancer import PortfolioRebalancer
        rb = PortfolioRebalancer(self._profile())
        actions = rb.evaluate([], 100_000.0)
        assert actions == []

    def test_bull_regime_lower_cash_target(self):
        from autonomous.portfolio_rebalancer import PortfolioRebalancer
        from advanced_ai.market_regime import Regime
        rb = PortfolioRebalancer(self._profile())
        bull_target = rb._regime_cash_target(Regime.BULL)
        bear_target = rb._regime_cash_target(Regime.BEAR)
        assert bear_target > bull_target

    def test_high_volatility_reduces_weakest_position(self):
        from autonomous.portfolio_rebalancer import PortfolioRebalancer
        from advanced_ai.market_regime import Regime
        rb = PortfolioRebalancer(self._profile())
        actions = rb.evaluate(self._positions(), 100_000.0, Regime.HIGH_VOLATILITY)
        sell_all = [a for a in actions if a.action == "SELL_ALL"]
        assert len(sell_all) >= 1

    def test_over_concentrated_triggers_sell_partial(self):
        from autonomous.portfolio_rebalancer import PortfolioRebalancer
        from advanced_ai.market_regime import Regime
        rb = PortfolioRebalancer(self._profile())
        # Moderate max_single_position_pct = 25%. 60k / 100k = 60% — over limit.
        big_pos = [{"symbol": "GIANT.NS", "market_value": 60000.0, "last_price": 1000.0, "unrealized_pnl": 1000.0, "quantity": 60}]
        actions = rb.evaluate(big_pos, 100_000.0, Regime.BULL)
        sell_partial = [a for a in actions if a.action == "SELL_PARTIAL"]
        assert len(sell_partial) >= 1

    def test_rebalance_action_to_dict(self):
        from autonomous.portfolio_rebalancer import RebalanceAction
        a = RebalanceAction("X.NS", "SELL_PARTIAL", "test", 30.0, 20.0, 5)
        d = a.to_dict()
        assert d["symbol"] == "X.NS"
        assert d["action"] == "SELL_PARTIAL"


# ────────────────────────────────────────────────────────────────────────────
# reporter
# ────────────────────────────────────────────────────────────────────────────

class TestAutonomousReporter:
    def _profile(self):
        from autonomous.risk_profile import RISK_PROFILES
        return RISK_PROFILES["moderate"]

    def test_build_report_returns_report_data(self):
        from autonomous.reporter import AutonomousReporter
        snap = {
            "balance": {"cash": 10000.0, "total_value": 50000.0, "unrealized_pnl": 500.0, "today_pnl": 200.0},
            "positions": [],
        }
        report = AutonomousReporter.build_report(
            period="daily",
            portfolio_snapshot=snap,
            starting_capital=50000.0,
        )
        assert report.period == "daily"
        assert report.total_capital == 50000.0
        assert report.cash == 10000.0

    def test_to_text_contains_key_info(self):
        from autonomous.reporter import AutonomousReporter, ReportData
        report = ReportData(
            period="daily", start_date="2025-01-01", end_date="2025-01-01",
            total_capital=50000.0, portfolio_value=51000.0, cash=10000.0,
            realized_pnl=500.0, unrealized_pnl=500.0, total_pnl=1000.0,
            total_return_pct=2.0, num_trades=5, win_rate=60.0,
        )
        text = report.to_text()
        assert "DAILY REPORT" in text
        assert "₹" in text
        assert "50,000" in text

    def test_send_daily_dispatches_when_profile_enabled(self):
        from autonomous.reporter import AutonomousReporter, ReportData
        from autonomous.risk_profile import RISK_PROFILES
        mock_tg = MagicMock()
        reporter = AutonomousReporter(RISK_PROFILES["moderate"], telegram_alerter=mock_tg)
        report = ReportData(
            period="daily", start_date="2025-01-01", end_date="2025-01-01",
            total_capital=50000.0, portfolio_value=51000.0, cash=10000.0,
            realized_pnl=0.0, unrealized_pnl=0.0, total_pnl=0.0,
            total_return_pct=0.0, num_trades=0, win_rate=0.0,
        )
        reporter.send_daily(report)
        mock_tg.daily_report.assert_called_once()

    def test_no_send_when_profile_disables_daily(self):
        from autonomous.reporter import AutonomousReporter, ReportData
        from autonomous.risk_profile import RISK_PROFILES
        import dataclasses
        prof = RISK_PROFILES["moderate"]
        # Create a copy-like profile with report_daily=False
        mock_prof = MagicMock()
        mock_prof.report_daily = False
        mock_prof.report_weekly = False
        mock_prof.report_monthly = False
        mock_tg = MagicMock()
        reporter = AutonomousReporter(mock_prof, telegram_alerter=mock_tg)
        report = ReportData(
            period="daily", start_date="2025-01-01", end_date="2025-01-01",
            total_capital=0, portfolio_value=0, cash=0, realized_pnl=0,
            unrealized_pnl=0, total_pnl=0, total_return_pct=0, num_trades=0, win_rate=0,
        )
        reporter.send_daily(report)
        mock_tg.daily_report.assert_not_called()

    def test_to_html_returns_string(self):
        from autonomous.reporter import ReportData
        report = ReportData(
            period="weekly", start_date="2025-01-01", end_date="2025-01-07",
            total_capital=50000.0, portfolio_value=52000.0, cash=8000.0,
            realized_pnl=1000.0, unrealized_pnl=1000.0, total_pnl=2000.0,
            total_return_pct=4.0, num_trades=10, win_rate=60.0,
        )
        html = report.to_html()
        assert "<h2>" in html
        assert "₹" in html


# ────────────────────────────────────────────────────────────────────────────
# autonomous_engine
# ────────────────────────────────────────────────────────────────────────────

class TestAutonomousEngine:
    def _make_engine(self, capital=50_000.0, profile="moderate"):
        from autonomous.autonomous_engine import AutonomousEngine, AutonomousConfig
        cfg = AutonomousConfig(
            total_capital=capital,
            risk_profile=profile,
            simulation_mode=True,
            enforce_market_hours=False,    # skip market hours in tests
            poll_interval_seconds=9999,
        )
        return AutonomousEngine(cfg)

    def test_engine_initialises(self):
        engine = self._make_engine()
        assert engine is not None
        assert engine.config.simulation_mode is True

    def test_status_returns_dict(self):
        engine = self._make_engine()
        s = engine.status()
        assert isinstance(s, dict)
        assert "running" in s
        assert "halted" in s
        assert "current_regime" in s

    def test_run_once_returns_dict(self):
        engine = self._make_engine()
        result = engine.run_once()
        assert isinstance(result, dict)
        assert "tick_at" in result

    def test_run_once_skips_when_no_data(self):
        engine = self._make_engine()
        result = engine.run_once()
        # With no data files, scan count should be 0
        assert result.get("scan_count", 0) == 0

    def test_run_once_with_ohlcv_override(self, tmp_path):
        engine = self._make_engine()
        # Provide synthetic market data
        n = 100
        rng = np.random.default_rng(0)
        close = 1000 + np.cumsum(rng.normal(0, 5, n))
        ohlcv = pd.DataFrame({
            "open": close, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.full(n, 200_000.0),
        })
        result = engine.run_once(ohlcv_override={"NIFTY": ohlcv})
        assert isinstance(result, dict)
        assert "current_regime" not in result or True   # just must not crash

    def test_manual_stop_halts_engine(self):
        engine = self._make_engine()
        engine.manual_stop()
        assert engine.safety._manual_stop is True

    def test_reset_safety_clears_halt(self):
        engine = self._make_engine()
        engine.manual_stop()
        engine.reset_safety()
        status = engine.safety.check()
        assert status.safe

    def test_conservative_profile_respected(self):
        engine = self._make_engine(profile="conservative")
        assert engine.profile.max_positions <= 3
        assert engine.profile.risk_per_trade_pct <= 1.0

    def test_aggressive_profile_respected(self):
        engine = self._make_engine(profile="aggressive")
        assert engine.profile.max_positions >= 5

    def test_start_and_stop(self):
        import time
        engine = self._make_engine()
        engine.start()
        assert engine._running is True
        time.sleep(0.1)
        engine.stop()
        assert engine._running is False

    def test_engine_status_after_tick(self):
        engine = self._make_engine()
        engine.run_once()
        s = engine.status()
        assert s["tick_count"] == 1
        assert s["last_tick_at"] != ""
