"""
System Monitor — Phase 10D.

Tracks CPU, RAM, Disk, Network, Database connectivity, and API health.
Uses psutil (optional; gracefully degrades if not installed).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _try_import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


@dataclass
class SystemSnapshot:
    timestamp: str
    cpu_pct: float
    ram_pct: float
    ram_used_gb: float
    ram_total_gb: float
    disk_pct: float
    disk_used_gb: float
    disk_total_gb: float
    net_sent_mb: float
    net_recv_mb: float
    db_ok: bool
    api_statuses: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def healthy(self) -> bool:
        return (
            self.cpu_pct < 90
            and self.ram_pct < 90
            and self.disk_pct < 90
            and self.db_ok
        )


class SystemMonitor:
    """
    Collects system resource metrics.

    Usage
    -----
    monitor = SystemMonitor(db_path="trading.db")
    snap = monitor.snapshot()
    history = monitor.history()   # list of recent snapshots
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        api_endpoints: dict[str, str] | None = None,
        history_limit: int = 100,
    ) -> None:
        self._db_path = Path(db_path) if db_path else None
        self._api_endpoints: dict[str, str] = api_endpoints or {}
        self._history: list[SystemSnapshot] = []
        self._history_limit = history_limit
        self._psutil = _try_import_psutil()
        self.logger = logging.getLogger(__name__)

    def snapshot(self) -> SystemSnapshot:
        cpu = self._cpu()
        ram = self._ram()
        disk = self._disk()
        net = self._net()
        db_ok = self._check_db()
        api = self._check_apis()
        warnings: list[str] = []
        if cpu > 80:
            warnings.append(f"High CPU: {cpu:.0f}%")
        if ram["pct"] > 80:
            warnings.append(f"High RAM: {ram['pct']:.0f}%")
        if disk["pct"] > 85:
            warnings.append(f"Low disk space: {disk['pct']:.0f}% used")
        if not db_ok:
            warnings.append("Database unreachable")
        for name, ok in api.items():
            if not ok:
                warnings.append(f"API down: {name}")

        snap = SystemSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_pct=cpu,
            ram_pct=ram["pct"],
            ram_used_gb=ram["used_gb"],
            ram_total_gb=ram["total_gb"],
            disk_pct=disk["pct"],
            disk_used_gb=disk["used_gb"],
            disk_total_gb=disk["total_gb"],
            net_sent_mb=net["sent_mb"],
            net_recv_mb=net["recv_mb"],
            db_ok=db_ok,
            api_statuses=api,
            warnings=warnings,
        )
        self._history.append(snap)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        return snap

    def history(self) -> list[SystemSnapshot]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Metric collectors
    # ------------------------------------------------------------------

    def _cpu(self) -> float:
        if self._psutil:
            return float(self._psutil.cpu_percent(interval=0.2))
        return 0.0

    def _ram(self) -> dict[str, float]:
        if self._psutil:
            mem = self._psutil.virtual_memory()
            return {
                "pct": mem.percent,
                "used_gb": mem.used / 1e9,
                "total_gb": mem.total / 1e9,
            }
        return {"pct": 0.0, "used_gb": 0.0, "total_gb": 0.0}

    def _disk(self, path: str = "/") -> dict[str, float]:
        if self._psutil:
            try:
                d = self._psutil.disk_usage(path)
                return {"pct": d.percent, "used_gb": d.used / 1e9, "total_gb": d.total / 1e9}
            except Exception:
                pass
        return {"pct": 0.0, "used_gb": 0.0, "total_gb": 0.0}

    def _net(self) -> dict[str, float]:
        if self._psutil:
            try:
                n = self._psutil.net_io_counters()
                return {"sent_mb": n.bytes_sent / 1e6, "recv_mb": n.bytes_recv / 1e6}
            except Exception:
                pass
        return {"sent_mb": 0.0, "recv_mb": 0.0}

    def _check_db(self) -> bool:
        if self._db_path is None:
            return True
        try:
            with sqlite3.connect(str(self._db_path), timeout=2) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _check_apis(self) -> dict[str, bool]:
        if not self._api_endpoints:
            return {}
        results = {}
        for name, url in self._api_endpoints.items():
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=3) as resp:
                    results[name] = resp.status < 500
            except Exception:
                results[name] = False
        return results
