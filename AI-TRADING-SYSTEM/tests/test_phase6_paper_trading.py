"""Tests for Phase 6 — Paper Trading Engine."""
from __future__ import annotations

import pytest

from paper_trading.paper_broker import BrokerConfig, PaperBroker
from paper_trading.paper_order_manager import OrderSide, OrderStatus, OrderType, PaperOrderManager
from paper_trading.paper_portfolio import PaperPortfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def broker() -> PaperBroker:
    cfg = BrokerConfig(starting_capital=100_000.0, brokerage_pct=0.0003, tax_pct=0.001, slippage_pct=0.0)
    pb = PaperBroker(cfg)
    pb.update_prices({"RELIANCE": 2500.0, "TCS": 3500.0})
    return pb


@pytest.fixture
def portfolio() -> PaperPortfolio:
    return PaperPortfolio(starting_capital=50_000.0)


@pytest.fixture
def order_manager() -> PaperOrderManager:
    return PaperOrderManager()


# ---------------------------------------------------------------------------
# PaperPortfolio
# ---------------------------------------------------------------------------

class TestPaperPortfolio:
    def _now(self):
        from datetime import datetime
        return datetime.now()

    def test_initial_cash(self, portfolio: PaperPortfolio) -> None:
        assert portfolio.cash == pytest.approx(50_000.0)

    def test_open_position_reduces_cash(self, portfolio: PaperPortfolio) -> None:
        portfolio.open_position("TCS", quantity=10, price=3000.0, fees=50.0, timestamp=self._now())
        assert portfolio.cash == pytest.approx(50_000.0 - 10 * 3000.0 - 50.0)

    def test_close_position_returns_cash(self, portfolio: PaperPortfolio) -> None:
        portfolio.open_position("TCS", quantity=10, price=3000.0, fees=50.0, timestamp=self._now())
        portfolio.close_position("TCS", quantity=10, price=3200.0, fees=50.0, timestamp=self._now())
        assert portfolio.cash > 50_000.0  # profit realized

    def test_unrealized_pnl(self, portfolio: PaperPortfolio) -> None:
        portfolio.open_position("TCS", quantity=10, price=3000.0, fees=0.0, timestamp=self._now())
        portfolio.update_prices({"TCS": 3500.0})
        assert portfolio.unrealized_pnl() == pytest.approx(5000.0)

    def test_total_value(self, portfolio: PaperPortfolio) -> None:
        portfolio.open_position("TCS", quantity=10, price=3000.0, fees=0.0, timestamp=self._now())
        portfolio.update_prices({"TCS": 3000.0})
        assert portfolio.total_value() == pytest.approx(50_000.0)

    def test_snapshot_keys(self, portfolio: PaperPortfolio) -> None:
        snap = portfolio.snapshot()
        assert "cash" in snap
        assert "open_positions" in snap
        assert "total_value" in snap


# ---------------------------------------------------------------------------
# PaperOrderManager
# ---------------------------------------------------------------------------

class TestPaperOrderManager:
    def test_create_order(self, order_manager: PaperOrderManager) -> None:
        order = order_manager.create_order("RELIANCE", OrderSide.BUY, 5, OrderType.MARKET)
        assert order.status == OrderStatus.PENDING

    def test_fill_order(self, order_manager: PaperOrderManager) -> None:
        order = order_manager.create_order("RELIANCE", OrderSide.BUY, 5, OrderType.MARKET)
        order_manager.fill_order(order.order_id, price=2500.0, fees=10.0)
        assert order_manager.get_order(order.order_id).status == OrderStatus.FILLED

    def test_cancel_order(self, order_manager: PaperOrderManager) -> None:
        order = order_manager.create_order("RELIANCE", OrderSide.SELL, 5, OrderType.MARKET)
        order_manager.cancel_order(order.order_id)
        assert order_manager.get_order(order.order_id).status == OrderStatus.CANCELLED

    def test_pending_orders(self, order_manager: PaperOrderManager) -> None:
        order_manager.create_order("TCS", OrderSide.BUY, 2, OrderType.MARKET)
        order_manager.create_order("TCS", OrderSide.BUY, 3, OrderType.MARKET)
        assert len(order_manager.pending_orders()) == 2

    def test_filled_orders_list(self, order_manager: PaperOrderManager) -> None:
        order = order_manager.create_order("TCS", OrderSide.BUY, 2, OrderType.MARKET)
        order_manager.fill_order(order.order_id, price=3500.0, fees=5.0)
        assert len(order_manager.filled_orders()) == 1

    def test_cancel_all_pending(self, order_manager: PaperOrderManager) -> None:
        for _ in range(3):
            order_manager.create_order("TCS", OrderSide.BUY, 1, OrderType.MARKET)
        order_manager.cancel_all_pending()
        assert order_manager.pending_orders() == []


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------

class TestPaperBroker:
    def test_buy_reduces_cash(self, broker: PaperBroker) -> None:
        initial = broker.portfolio.cash
        order = broker.place_buy("RELIANCE", 10, 2500.0)
        assert order.status == OrderStatus.FILLED
        assert broker.portfolio.cash < initial

    def test_sell_without_position_rejected(self, broker: PaperBroker) -> None:
        order = broker.place_sell("TCS", 5, 3500.0)
        assert order.status == OrderStatus.REJECTED

    def test_sell_after_buy_fills(self, broker: PaperBroker) -> None:
        broker.place_buy("TCS", 5, 3500.0)
        order = broker.place_sell("TCS", 5, 3600.0)
        assert order.status == OrderStatus.FILLED

    def test_insufficient_cash_rejected(self, broker: PaperBroker) -> None:
        order = broker.place_buy("RELIANCE", 10_000, 2500.0)  # way too many
        assert order.status == OrderStatus.REJECTED

    def test_trading_disabled_rejects(self, broker: PaperBroker) -> None:
        broker.enable_trading(False)
        order = broker.place_buy("RELIANCE", 1, 2500.0)
        assert order.status == OrderStatus.REJECTED

    def test_get_balance_keys(self, broker: PaperBroker) -> None:
        bal = broker.get_balance()
        assert "cash" in bal
        assert "total_value" in bal

    def test_snapshot_structure(self, broker: PaperBroker) -> None:
        broker.place_buy("RELIANCE", 2, 2500.0)
        snap = broker.snapshot()
        assert "positions" in snap
        assert "balance" in snap

    def test_daily_loss_limit(self, broker: PaperBroker) -> None:
        from datetime import date
        # Inject a large daily loss directly into the portfolio's daily_pnl record
        today = date.today().isoformat()
        broker.portfolio.daily_pnl[today] = -5_000.0   # exceeds 3% of 100k (=3000)
        still_ok = broker.check_daily_loss_limit()
        assert still_ok is False
        assert not broker.is_trading_enabled()
