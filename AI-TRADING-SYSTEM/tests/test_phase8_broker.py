"""Tests for Phase 8 — Broker Integration & Execution Layer."""
from __future__ import annotations

import pytest

from broker.broker_factory import BrokerFactory
from broker.zerodha.zerodha_adapter import ZerodhaAdapter
from broker.dhan.dhan_adapter import DhanAdapter
from broker.angelone.angel_adapter import AngelOneAdapter
from broker.groww.groww_adapter import GrowwAdapter
from broker.base_broker import OrderResult
from execution.order_tracker import OrderTracker, TrackedStatus
from execution.order_manager import OrderManager
from execution.execution_engine import ExecutionConfig, ExecutionEngine


# ---------------------------------------------------------------------------
# BrokerFactory
# ---------------------------------------------------------------------------

class TestBrokerFactory:
    def test_paper_broker_is_default(self) -> None:
        broker = BrokerFactory.create("paper")
        assert broker is not None
        assert broker.SIMULATION_MODE is True

    def test_zerodha_simulation(self) -> None:
        broker = BrokerFactory.create("zerodha")
        assert broker.SIMULATION_MODE is True

    def test_dhan_simulation(self) -> None:
        broker = BrokerFactory.create("dhan")
        assert broker.SIMULATION_MODE is True

    def test_angelone_simulation(self) -> None:
        broker = BrokerFactory.create("angelone")
        assert broker.SIMULATION_MODE is True

    def test_groww_simulation(self) -> None:
        broker = BrokerFactory.create("groww")
        assert broker.SIMULATION_MODE is True

    def test_available_brokers_list(self) -> None:
        brokers = BrokerFactory.available_brokers()
        assert "paper" in brokers
        assert "zerodha" in brokers
        assert "groww" in brokers


# ---------------------------------------------------------------------------
# Adapter tests (simulation mode)
# ---------------------------------------------------------------------------

class TestZerodhaAdapter:
    @pytest.fixture
    def adapter(self) -> ZerodhaAdapter:
        a = ZerodhaAdapter(api_key="test", access_token="test", simulation_mode=True)
        a.set_mock_price("RELIANCE", 2500.0)
        a.login()
        return a

    def test_login(self, adapter: ZerodhaAdapter) -> None:
        assert adapter.is_connected

    def test_place_buy(self, adapter: ZerodhaAdapter) -> None:
        result = adapter.place_buy("RELIANCE", 5, 2500.0)
        assert result.status in ("SIMULATED", "FILLED", "PLACED")
        assert result.symbol == "RELIANCE"

    def test_place_sell(self, adapter: ZerodhaAdapter) -> None:
        result = adapter.place_sell("RELIANCE", 5, 2500.0)
        assert result.side == "SELL"

    def test_get_balance_keys(self, adapter: ZerodhaAdapter) -> None:
        bal = adapter.get_balance()
        assert "cash" in bal

    def test_broker_name(self, adapter: ZerodhaAdapter) -> None:
        assert "Zerodha" in adapter.broker_name()


class TestDhanAdapter:
    @pytest.fixture
    def adapter(self) -> DhanAdapter:
        a = DhanAdapter(client_id="test", access_token="test", simulation_mode=True)
        a.login()
        return a

    def test_place_buy(self, adapter: DhanAdapter) -> None:
        result = adapter.place_buy("TCS", 2, 3500.0)
        assert isinstance(result, OrderResult)

    def test_get_positions(self, adapter: DhanAdapter) -> None:
        positions = adapter.get_positions()
        assert isinstance(positions, list)


class TestAngelOneAdapter:
    @pytest.fixture
    def adapter(self) -> AngelOneAdapter:
        a = AngelOneAdapter(api_key="test", client_code="test", password="test",
                            totp_secret="test", simulation_mode=True)
        a.login()
        return a

    def test_place_buy(self, adapter: AngelOneAdapter) -> None:
        result = adapter.place_buy("INFY", 3, 1500.0)
        assert isinstance(result, OrderResult)

    def test_logout(self, adapter: AngelOneAdapter) -> None:
        adapter.logout()
        assert not adapter.is_connected


class TestGrowwAdapter:
    def test_simulation_only(self) -> None:
        adapter = GrowwAdapter(simulation_mode=True)
        assert adapter.SIMULATION_MODE is True

    def test_live_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            GrowwAdapter(simulation_mode=False)

    def test_place_buy(self) -> None:
        adapter = GrowwAdapter()
        adapter.login()
        result = adapter.place_buy("WIPRO", 5, 400.0)
        assert result.status == "SIMULATED"


# ---------------------------------------------------------------------------
# OrderTracker
# ---------------------------------------------------------------------------

class TestOrderTracker:
    def test_track_and_retrieve(self) -> None:
        tracker = OrderTracker()
        order = tracker.track(
            order_id="O1", broker_order_id="B1", symbol="TCS", side="BUY",
            quantity=5, requested_price=3500.0, filled_price=3501.0,
            status=TrackedStatus.FILLED, broker_name="Paper",
        )
        assert tracker.get("O1") is order

    def test_by_status_filters(self) -> None:
        tracker = OrderTracker()
        tracker.track("O1", "B1", "TCS", "BUY", 5, 100.0, 100.0, TrackedStatus.FILLED, "Paper")
        tracker.track("O2", "B2", "TCS", "SELL", 5, 100.0, None, TrackedStatus.PENDING, "Paper")
        assert len(tracker.by_status(TrackedStatus.FILLED)) == 1
        assert len(tracker.by_status(TrackedStatus.PENDING)) == 1

    def test_update_status(self) -> None:
        tracker = OrderTracker()
        tracker.track("O1", "B1", "INFY", "BUY", 3, 1500.0, None, TrackedStatus.PENDING, "Paper")
        tracker.update("O1", TrackedStatus.FILLED, filled_price=1502.0)
        assert tracker.get("O1").status == TrackedStatus.FILLED
        assert tracker.get("O1").filled_price == pytest.approx(1502.0)

    def test_summary_counts(self) -> None:
        tracker = OrderTracker()
        tracker.track("O1", "B1", "TCS", "BUY", 1, 100.0, 100.0, TrackedStatus.FILLED, "Paper")
        tracker.track("O2", "B2", "TCS", "BUY", 1, 100.0, None, TrackedStatus.REJECTED, "Paper")
        s = tracker.summary()
        assert s["FILLED"] == 1
        assert s["REJECTED"] == 1
        assert s["total"] == 2

    def test_by_symbol(self) -> None:
        tracker = OrderTracker()
        tracker.track("O1", "B1", "TCS", "BUY", 1, 100.0, 100.0, TrackedStatus.FILLED, "Paper")
        tracker.track("O2", "B2", "INFY", "BUY", 1, 100.0, 100.0, TrackedStatus.FILLED, "Paper")
        assert len(tracker.by_symbol("TCS")) == 1


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class TestOrderManager:
    @pytest.fixture
    def manager(self) -> OrderManager:
        broker = BrokerFactory.create("zerodha")
        broker.set_mock_price("RELIANCE", 2500.0)
        broker.login()
        tracker = OrderTracker()
        return OrderManager(broker, tracker)

    def test_buy_records_in_tracker(self, manager: OrderManager) -> None:
        manager.buy("RELIANCE", 5, 2500.0)
        assert manager.summary()["total"] >= 1

    def test_sell_records_in_tracker(self, manager: OrderManager) -> None:
        manager.sell("RELIANCE", 5, 2500.0)
        assert manager.summary()["total"] >= 1

    def test_zero_quantity_rejected(self, manager: OrderManager) -> None:
        result = manager.buy("RELIANCE", 0, 2500.0)
        assert result is None

    def test_duplicate_order_blocked(self, manager: OrderManager) -> None:
        # Manually add a pending key to simulate mid-flight order
        manager._recent_orders.add("TCS:BUY")
        result = manager.buy("TCS", 1, 3500.0)  # should be blocked as duplicate
        assert result is None


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    @pytest.fixture
    def engine(self) -> ExecutionEngine:
        broker = BrokerFactory.create("zerodha")
        broker.set_mock_price("RELIANCE", 2500.0)
        broker.login()
        return ExecutionEngine(broker, config=ExecutionConfig(dry_run=False))

    def test_hold_signal_skipped(self, engine: ExecutionEngine) -> None:
        result = engine.execute({"action": "HOLD", "symbol": "RELIANCE", "quantity": 5}, price=2500.0)
        assert result["status"] == "SKIPPED"

    def test_buy_signal_submitted(self, engine: ExecutionEngine) -> None:
        result = engine.execute({"action": "BUY", "symbol": "RELIANCE", "quantity": 5}, price=2500.0)
        assert result["status"] == "SUBMITTED"

    def test_sell_signal_submitted(self, engine: ExecutionEngine) -> None:
        result = engine.execute({"action": "SELL", "symbol": "RELIANCE", "quantity": 5}, price=2500.0)
        assert result["status"] == "SUBMITTED"

    def test_dry_run_mode(self) -> None:
        broker = BrokerFactory.create("zerodha")
        broker.set_mock_price("RELIANCE", 2500.0)
        broker.login()
        eng = ExecutionEngine(broker, config=ExecutionConfig(dry_run=True))
        result = eng.execute({"action": "BUY", "symbol": "RELIANCE", "quantity": 5}, price=2500.0)
        assert result["status"] == "DRY_RUN"
        assert eng.tracker.summary()["total"] == 0

    def test_get_summary_keys(self, engine: ExecutionEngine) -> None:
        s = engine.get_summary()
        assert "simulation_mode" in s
        assert "broker_connected" in s
        assert "order_summary" in s
