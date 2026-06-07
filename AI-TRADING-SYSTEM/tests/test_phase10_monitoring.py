"""Tests for Phase 10 — Cloud Deployment & Monitoring."""
from __future__ import annotations

import pytest

from monitoring.system_monitor import SystemMonitor, SystemSnapshot
from monitoring.backup_manager import BackupManager
from monitoring.log_manager import configure_log_manager, get_trade_logger
from alerts.telegram_alerts import TelegramAlerter
from alerts.email_alerts import EmailAlerter
from alerts.discord_alerts import DiscordAlerter


# ---------------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------------

class TestSystemMonitor:
    def test_snapshot_returns_instance(self) -> None:
        monitor = SystemMonitor()
        snap = monitor.snapshot()
        assert isinstance(snap, SystemSnapshot)

    def test_snapshot_has_required_fields(self) -> None:
        snap = SystemMonitor().snapshot()
        assert hasattr(snap, "cpu_pct")
        assert hasattr(snap, "ram_pct")
        assert hasattr(snap, "disk_pct")
        assert hasattr(snap, "db_ok")
        assert hasattr(snap, "timestamp")

    def test_healthy_property_with_good_metrics(self) -> None:
        snap = SystemSnapshot(
            timestamp="2025-01-01T00:00:00",
            cpu_pct=30.0, ram_pct=40.0,
            ram_used_gb=4.0, ram_total_gb=16.0,
            disk_pct=50.0, disk_used_gb=50.0, disk_total_gb=100.0,
            net_sent_mb=100.0, net_recv_mb=200.0,
            db_ok=True,
        )
        assert snap.healthy is True

    def test_healthy_property_with_high_cpu(self) -> None:
        snap = SystemSnapshot(
            timestamp="2025-01-01T00:00:00",
            cpu_pct=95.0, ram_pct=40.0,
            ram_used_gb=4.0, ram_total_gb=16.0,
            disk_pct=50.0, disk_used_gb=50.0, disk_total_gb=100.0,
            net_sent_mb=0.0, net_recv_mb=0.0,
            db_ok=True,
        )
        assert snap.healthy is False

    def test_to_dict_serialisable(self) -> None:
        snap = SystemMonitor().snapshot()
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "cpu_pct" in d

    def test_history_accumulates(self) -> None:
        monitor = SystemMonitor(history_limit=5)
        for _ in range(3):
            monitor.snapshot()
        assert len(monitor.history()) == 3

    def test_history_limit_enforced(self) -> None:
        monitor = SystemMonitor(history_limit=2)
        for _ in range(5):
            monitor.snapshot()
        assert len(monitor.history()) == 2

    def test_db_check_bad_path_returns_false(self) -> None:
        # A path with invalid chars forces a connection error on all platforms
        monitor = SystemMonitor(db_path="/\x00invalid/path.db")
        assert monitor._check_db() is False


# ---------------------------------------------------------------------------
# BackupManager
# ---------------------------------------------------------------------------

class TestBackupManager:
    def test_list_backups_returns_list(self, tmp_path) -> None:
        bm = BackupManager(backup_dir=tmp_path / "backups")
        assert isinstance(bm.list_backups(), list)

    def test_backup_missing_db_returns_none(self, tmp_path) -> None:
        bm = BackupManager(backup_dir=tmp_path / "backups")
        result = bm.backup_database(tmp_path / "no.db")
        assert result is None

    def test_backup_db_creates_zip(self, tmp_path) -> None:
        db_file = tmp_path / "test.db"
        db_file.write_text("fake db")
        bm = BackupManager(backup_dir=tmp_path / "backups")
        result = bm.backup_database(db_file)
        assert result is not None
        assert result.exists()

    def test_backup_all_returns_dict(self, tmp_path) -> None:
        bm = BackupManager(backup_dir=tmp_path / "backups")
        result = bm.backup_all()
        assert isinstance(result, dict)
        assert "database" in result
        assert "models" in result


# ---------------------------------------------------------------------------
# LogManager
# ---------------------------------------------------------------------------

class TestLogManager:
    def test_configure_does_not_raise(self, tmp_path) -> None:
        configure_log_manager(log_dir=tmp_path / "logs")

    def test_trade_logger_available(self, tmp_path) -> None:
        configure_log_manager(log_dir=tmp_path / "logs")
        logger = get_trade_logger()
        assert logger is not None
        logger.info("Test trade log entry")


# ---------------------------------------------------------------------------
# Alerters (no-credential mode → logs only)
# ---------------------------------------------------------------------------

class TestTelegramAlerter:
    def test_send_without_credentials_returns_true(self) -> None:
        alerter = TelegramAlerter(token="", chat_id="")
        assert alerter.send("test message") is True

    def test_trade_executed_does_not_raise(self) -> None:
        alerter = TelegramAlerter()
        alerter.trade_executed("RELIANCE", "BUY", 10, 2500.0, pnl=500.0)

    def test_risk_breach_does_not_raise(self) -> None:
        alerter = TelegramAlerter()
        alerter.risk_breach("Max drawdown exceeded", "15.2%")

    def test_daily_report_does_not_raise(self) -> None:
        alerter = TelegramAlerter()
        alerter.daily_report(pnl=5000.0, trades=12, win_rate=66.7)


class TestEmailAlerter:
    def test_send_without_credentials_returns_true(self) -> None:
        alerter = EmailAlerter(host="", user="", recipient="")
        assert alerter.send("Test Subject", "Test body") is True

    def test_trade_executed_does_not_raise(self) -> None:
        alerter = EmailAlerter()
        alerter.trade_executed("TCS", "SELL", 5, 3500.0)


class TestDiscordAlerter:
    def test_send_without_webhook_returns_true(self) -> None:
        alerter = DiscordAlerter(webhook_url="")
        assert alerter.send("test") is True

    def test_daily_report_does_not_raise(self) -> None:
        alerter = DiscordAlerter()
        alerter.daily_report(pnl=-200.0, trades=5, win_rate=40.0)

    def test_system_failure_does_not_raise(self) -> None:
        alerter = DiscordAlerter()
        alerter.system_failure("PredictionEngine", "Model file not found")
