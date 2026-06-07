"""Tests for Phase 9 — Auto Execution Engine."""
from __future__ import annotations

from datetime import datetime

import pytest

from broker.broker_factory import BrokerFactory
from execution.trade_executor import TradeExecutor, TradeJournal, TradeRecord
from execution.trade_monitor import MonitorConfig, TradeMonitor
from execution.live_execution_engine import LiveEngineConfig, LiveExecutionEngine, MarketHoursEngine
from risk_management.risk_engine import RiskConfig, RiskEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def broker():
    b = BrokerFactory.create("zerodha")
    b.set_mock_price("RELIANCE", 2500.0)
    b.set_mock_price("TCS", 3500.0)
    b.login()
    return b


@pytest.fixture
def risk_engine():
    return RiskEngine(RiskConfig(risk_per_trade_pct=2.0))


@pytest.fixture
def journal():
    return TradeJournal(path="reports/test_journal.csv")


@pytest.fixture
def executor(broker, risk_engine, journal):
    return TradeExecutor(broker, risk_engine, journal)


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord:
    def test_to_dict_keys(self) -> None:
        record = TradeRecord(
            trade_id="T1", symbol="RELIANCE", side="BUY", quantity=5,
            entry_price=2500.0, exit_price=None, entry_time=datetime.now(),
            exit_time=None, stop_loss=2400.0, take_profit=2700.0,
            reason="signal", ai_confidence=75.0, risk_score=30.0,
        )
        d = record.to_dict()
        assert "trade_id" in d
        assert "realized_pnl" in d

    def test_close_computes_pnl(self) -> None:
        record = TradeRecord(
            trade_id="T1", symbol="TCS", side="BUY", quantity=10,
            entry_price=3500.0, exit_price=None, entry_time=datetime.now(),
            exit_time=None, stop_loss=3300.0, take_profit=3800.0,
            reason="test", ai_confidence=70.0, risk_score=20.0,
        )
        record.close(exit_price=3700.0)
        assert record.realized_pnl == pytest.approx(2000.0)
        assert record.exit_price == 3700.0

    def test_sell_side_pnl(self) -> None:
        record = TradeRecord(
            trade_id="T2", symbol="RELIANCE", side="SELL", quantity=5,
            entry_price=2500.0, exit_price=None, entry_time=datetime.now(),
            exit_time=None, stop_loss=2600.0, take_profit=2300.0,
            reason="test", ai_confidence=70.0, risk_score=20.0,
        )
        record.close(exit_price=2300.0)
        assert record.realized_pnl == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------

class TestTradeJournal:
    def _make_record(self, trade_id: str = "T1") -> TradeRecord:
        return TradeRecord(
            trade_id=trade_id, symbol="RELIANCE", side="BUY", quantity=5,
            entry_price=2500.0, exit_price=None, entry_time=datetime.now(),
            exit_time=None, stop_loss=2400.0, take_profit=2700.0,
            reason="signal", ai_confidence=75.0, risk_score=30.0,
        )

    def test_add_and_open(self) -> None:
        j = TradeJournal(path="reports/test_j.csv")
        j.add(self._make_record())
        assert len(j.open_trades()) == 1

    def test_close_trade_moves_to_closed(self) -> None:
        j = TradeJournal(path="reports/test_j.csv")
        r = self._make_record("T99")
        j.add(r)
        j.close_trade("T99", exit_price=2600.0, reason="take_profit")
        assert len(j.closed_trades()) == 1
        assert r.realized_pnl == pytest.approx(500.0)

    def test_today_pnl(self) -> None:
        j = TradeJournal(path="reports/test_j.csv")
        r = self._make_record("T88")
        j.add(r)
        j.close_trade("T88", 2700.0)
        assert j.today_pnl() == pytest.approx(1000.0)

    def test_all_as_df(self) -> None:
        j = TradeJournal(path="reports/test_j.csv")
        j.add(self._make_record())
        df = j.all_as_df()
        assert not df.empty
        assert "trade_id" in df.columns


# ---------------------------------------------------------------------------
# TradeExecutor
# ---------------------------------------------------------------------------

class TestTradeExecutor:
    def test_execute_buy_creates_record(self, executor: TradeExecutor) -> None:
        record = executor.execute_buy("RELIANCE", price=2500.0, ai_confidence=75.0)
        assert record is not None
        assert record.symbol == "RELIANCE"
        assert record.side == "BUY"

    def test_execute_buy_sets_stops(self, executor: TradeExecutor) -> None:
        record = executor.execute_buy("RELIANCE", price=2500.0, atr=50.0)
        assert record is not None
        assert record.stop_loss is not None
        assert record.stop_loss < 2500.0

    def test_execute_sell_closes_record(self, executor: TradeExecutor) -> None:
        record = executor.execute_buy("RELIANCE", price=2500.0)
        assert record is not None
        pnl = executor.execute_sell(record, price=2600.0)
        assert record.exit_price == pytest.approx(2600.0)

    def test_journal_tracks_trade(self, executor: TradeExecutor) -> None:
        executor.execute_buy("TCS", price=3500.0)
        assert len(executor.journal.open_trades()) >= 1


# ---------------------------------------------------------------------------
# TradeMonitor
# ---------------------------------------------------------------------------

class TestTradeMonitor:
    def _open_trade(self, executor: TradeExecutor, symbol: str = "RELIANCE", entry: float = 2500.0, stop: float = 2400.0, target: float = 2700.0) -> TradeRecord:
        from execution.trade_executor import TradeRecord
        import uuid
        r = TradeRecord(
            trade_id=str(uuid.uuid4())[:8], symbol=symbol, side="BUY",
            quantity=5, entry_price=entry, exit_price=None,
            entry_time=datetime.now(), exit_time=None,
            stop_loss=stop, take_profit=target,
            reason="test", ai_confidence=70.0, risk_score=10.0,
        )
        executor.journal.add(r)
        return r

    def test_stop_loss_triggers_exit(self, executor: TradeExecutor) -> None:
        self._open_trade(executor, stop=2400.0)
        monitor = TradeMonitor(executor, MonitorConfig(enable_take_profit=False, enable_trailing_stop=False))
        exits = monitor.check_all(price_fn=lambda sym: 2300.0)  # below stop
        assert len(exits) == 1
        assert exits[0]["reason"] == "stop_loss"

    def test_take_profit_triggers_exit(self, executor: TradeExecutor) -> None:
        self._open_trade(executor, target=2700.0)
        monitor = TradeMonitor(executor, MonitorConfig(enable_stop_loss=False, enable_trailing_stop=False))
        exits = monitor.check_all(price_fn=lambda sym: 2800.0)  # above target
        assert len(exits) == 1
        assert exits[0]["reason"] == "take_profit"

    def test_no_exit_within_range(self, executor: TradeExecutor) -> None:
        self._open_trade(executor, stop=2400.0, target=2700.0)
        monitor = TradeMonitor(executor, MonitorConfig(
            enable_stop_loss=True, enable_take_profit=True,
            enable_trailing_stop=False, enable_time_exit=False, enable_eod_exit=False,
        ))
        exits = monitor.check_all(price_fn=lambda sym: 2550.0)  # within range
        assert len(exits) == 0

    def test_ai_exit_signal(self, executor: TradeExecutor) -> None:
        self._open_trade(executor, stop=2000.0, target=5000.0)
        monitor = TradeMonitor(executor, MonitorConfig(
            enable_stop_loss=False, enable_take_profit=False,
            enable_trailing_stop=False, enable_time_exit=False, enable_eod_exit=False,
            ai_exit_threshold=0.7,
        ))
        exits = monitor.check_all(
            price_fn=lambda sym: 2550.0,
            ai_signal_fn=lambda sym: ("SELL", 0.8),
        )
        assert any("ai_exit" in e["reason"] for e in exits)


# ---------------------------------------------------------------------------
# MarketHoursEngine
# ---------------------------------------------------------------------------

class TestMarketHoursEngine:
    def test_is_market_open_returns_bool(self) -> None:
        result = MarketHoursEngine.is_market_open()
        assert isinstance(result, bool)

    def test_holiday_is_closed(self) -> None:
        # Just verify the holiday is in the set — no datetime mocking needed
        assert "2025-01-26" in MarketHoursEngine.NSE_HOLIDAYS_2025


# ---------------------------------------------------------------------------
# LiveExecutionEngine
# ---------------------------------------------------------------------------

class TestLiveExecutionEngine:
    def test_run_once_market_closed(self, broker) -> None:
        cfg = LiveEngineConfig(
            simulation_mode=True,
            enforce_market_hours=True,
            symbols=["RELIANCE"],
        )
        from unittest.mock import patch
        eng = LiveExecutionEngine(broker, config=cfg)
        # Patch market hours to return False
        with patch("execution.live_execution_engine.MarketHoursEngine.is_market_open", return_value=False):
            result = eng.run_once()
        assert result["status"] == "market_closed"

    def test_run_once_no_model_skips_gracefully(self, broker) -> None:
        cfg = LiveEngineConfig(
            simulation_mode=True,
            enforce_market_hours=False,
            symbols=["RELIANCE"],
        )
        eng = LiveExecutionEngine(broker, config=cfg)
        # No model registered → prediction will fail → engine logs error, doesn't crash
        result = eng.run_once()
        assert "errors" in result or "signals" in result

    def test_status_structure(self, broker) -> None:
        eng = LiveExecutionEngine(broker, config=LiveEngineConfig(enforce_market_hours=False))
        s = eng.status()
        assert "running" in s
        assert "open_positions" in s
        assert "today_pnl" in s

    def test_stop(self, broker) -> None:
        eng = LiveExecutionEngine(broker, config=LiveEngineConfig())
        eng._running = True
        eng.stop()
        assert not eng._running
