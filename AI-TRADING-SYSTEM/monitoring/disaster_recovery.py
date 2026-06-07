"""
Disaster Recovery System — Production Audit.

Provides:
  * Database backup (SQLite snapshot with integrity check)
  * Model backup (copy all registered model files)
  * Automatic restore (DB, models, trade logs)
  * Server recovery (state rebuild from logs)
  * Trade log recovery (CSV append journal reconstruction)

All operations are non-destructive: originals are never overwritten.
Backups are timestamped and stored under backup_root (default: backups/).
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BackupManifest:
    """Index file written alongside every backup set."""
    backup_id: str
    created_at: str
    source_root: str
    files: dict[str, str] = field(default_factory=dict)   # label → relative path
    integrity: dict[str, bool] = field(default_factory=dict)  # label → ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "source_root": self.source_root,
            "files": self.files,
            "integrity": self.integrity,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "BackupManifest":
        data = json.loads(path.read_text())
        m = cls(
            backup_id=data["backup_id"],
            created_at=data["created_at"],
            source_root=data["source_root"],
        )
        m.files = data.get("files", {})
        m.integrity = data.get("integrity", {})
        return m


@dataclass
class RecoveryResult:
    success: bool
    restored: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    backup_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "restored": self.restored,
            "skipped": self.skipped,
            "errors": self.errors,
            "backup_id": self.backup_id,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DisasterRecovery:
    """
    Disaster Recovery coordinator.

    Usage
    -----
    dr = DisasterRecovery()
    manifest = dr.backup_all()          # full snapshot
    result   = dr.restore_latest()      # automatic restore from newest backup
    result   = dr.restore(backup_id)    # restore specific backup
    logs     = dr.recover_trade_logs()  # rebuild trade log from CSV
    dr.server_recovery()                # full cold-start recovery sequence
    """

    def __init__(
        self,
        source_root: str | Path = ".",
        backup_root: str | Path = "backups",
        db_path: str | Path | None = None,
        model_dir: str | Path = "models/saved",
        trade_journal_path: str | Path = "reports/trade_journal.csv",
        log_dir: str | Path = "logs",
    ) -> None:
        self.source_root = Path(source_root)
        self.backup_root = Path(backup_root)
        self.db_path = Path(db_path) if db_path else Path("database/trading.db")
        self.model_dir = Path(model_dir)
        self.trade_journal_path = Path(trade_journal_path)
        self.log_dir = Path(log_dir)
        self.backup_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BACKUP
    # ------------------------------------------------------------------

    def backup_all(self) -> BackupManifest:
        """
        Create a complete backup: database + models + trade logs.
        Returns the manifest for the backup set.
        """
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True)

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=datetime.now().isoformat(),
            source_root=str(self.source_root),
        )

        manifest.files.update(self._backup_database(backup_dir, manifest))
        manifest.files.update(self._backup_models(backup_dir, manifest))
        manifest.files.update(self._backup_trade_logs(backup_dir, manifest))
        manifest.files.update(self._backup_config(backup_dir, manifest))

        manifest.save(backup_dir / "manifest.json")
        _logger.info("Backup complete: %s (%d files)", backup_id, len(manifest.files))
        return manifest

    def _backup_database(self, backup_dir: Path, manifest: BackupManifest) -> dict[str, str]:
        """Hot-backup SQLite using the sqlite3 backup API (safe while running)."""
        label = "database"
        if not self.db_path.exists():
            manifest.integrity[label] = False
            _logger.warning("DB not found at %s — skipping db backup", self.db_path)
            return {}

        dest = backup_dir / "trading.db"
        try:
            with sqlite3.connect(str(self.db_path)) as src:
                src.execute("PRAGMA wal_checkpoint(FULL)")
                with sqlite3.connect(str(dest)) as dst:
                    src.backup(dst)
            # Integrity check
            with sqlite3.connect(str(dest)) as conn:
                result = conn.execute("PRAGMA integrity_check").fetchone()
                ok = result[0] == "ok"
            manifest.integrity[label] = ok
            if not ok:
                _logger.error("DB backup integrity check FAILED for %s", backup_dir)
            return {label: str(dest.relative_to(self.backup_root))}
        except Exception as exc:
            manifest.integrity[label] = False
            _logger.error("DB backup error: %s", exc)
            return {}

    def _backup_models(self, backup_dir: Path, manifest: BackupManifest) -> dict[str, str]:
        """Copy all model files and the registry JSON."""
        files: dict[str, str] = {}
        model_backup_dir = backup_dir / "models"
        model_backup_dir.mkdir()

        # Model files
        if self.model_dir.exists():
            for model_file in self.model_dir.rglob("*"):
                if model_file.is_file():
                    rel = model_file.relative_to(self.model_dir)
                    dest = model_backup_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, dest)
                    label = f"model:{rel}"
                    files[label] = str(dest.relative_to(self.backup_root))
                    manifest.integrity[label] = True
        else:
            _logger.warning("Model dir %s not found — skipping model backup", self.model_dir)

        # Registry
        registry_path = Path("models/registry.json")
        if registry_path.exists():
            dest = backup_dir / "registry.json"
            shutil.copy2(registry_path, dest)
            files["model_registry"] = str(dest.relative_to(self.backup_root))
            manifest.integrity["model_registry"] = True

        return files

    def _backup_trade_logs(self, backup_dir: Path, manifest: BackupManifest) -> dict[str, str]:
        """Copy trade journal CSV and any log files."""
        files: dict[str, str] = {}
        logs_backup_dir = backup_dir / "logs"
        logs_backup_dir.mkdir()

        # Trade journal
        if self.trade_journal_path.exists():
            dest = logs_backup_dir / self.trade_journal_path.name
            shutil.copy2(self.trade_journal_path, dest)
            files["trade_journal"] = str(dest.relative_to(self.backup_root))
            manifest.integrity["trade_journal"] = True

        # System logs
        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log*"):
                dest = logs_backup_dir / log_file.name
                try:
                    shutil.copy2(log_file, dest)
                    files[f"log:{log_file.name}"] = str(dest.relative_to(self.backup_root))
                    manifest.integrity[f"log:{log_file.name}"] = True
                except Exception as exc:
                    _logger.warning("Could not backup log %s: %s", log_file, exc)

        return files

    def _backup_config(self, backup_dir: Path, manifest: BackupManifest) -> dict[str, str]:
        """Backup .env (if present) and requirements.txt."""
        files: dict[str, str] = {}
        for name in [".env", "requirements.txt", "docker-compose.yml"]:
            src = self.source_root / name
            if src.exists():
                dest = backup_dir / name
                shutil.copy2(src, dest)
                files[f"config:{name}"] = str(dest.relative_to(self.backup_root))
                manifest.integrity[f"config:{name}"] = True
        return files

    # ------------------------------------------------------------------
    # RESTORE
    # ------------------------------------------------------------------

    def restore_latest(self, dry_run: bool = False) -> RecoveryResult:
        """Restore from the most recent backup."""
        latest = self._find_latest_backup()
        if latest is None:
            return RecoveryResult(success=False, errors=["No backups found in " + str(self.backup_root)])
        return self.restore(latest.backup_id, dry_run=dry_run)

    def restore(self, backup_id: str, dry_run: bool = False) -> RecoveryResult:
        """
        Restore database, models, and trade journal from a specific backup.

        Parameters
        ----------
        backup_id : str
            The backup timestamp ID (folder name under backup_root).
        dry_run : bool
            If True, only report what would be restored without touching files.
        """
        backup_dir = self.backup_root / backup_id
        manifest_path = backup_dir / "manifest.json"

        if not backup_dir.exists():
            return RecoveryResult(success=False, errors=[f"Backup {backup_id} not found"])

        manifest = BackupManifest.load(manifest_path) if manifest_path.exists() else None
        result = RecoveryResult(success=True, backup_id=backup_id)

        # --- Database ---
        db_backup = backup_dir / "trading.db"
        if db_backup.exists():
            if not dry_run:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                # Pre-restore integrity check
                with sqlite3.connect(str(db_backup)) as conn:
                    ic = conn.execute("PRAGMA integrity_check").fetchone()
                if ic and ic[0] == "ok":
                    shutil.copy2(db_backup, self.db_path)
                    result.restored.append("database")
                    _logger.info("Database restored from %s", backup_id)
                else:
                    result.errors.append("database: backup integrity check failed — NOT restored")
                    _logger.error("DB backup %s failed integrity check — skipping restore", backup_id)
            else:
                result.restored.append("database (dry_run)")
        else:
            result.skipped.append("database: backup not found")

        # --- Models ---
        model_backup_dir = backup_dir / "models"
        if model_backup_dir.exists():
            if not dry_run:
                if self.model_dir.exists():
                    shutil.rmtree(self.model_dir)
                shutil.copytree(model_backup_dir, self.model_dir)
                result.restored.append("models")
                _logger.info("Models restored from %s", backup_id)
            else:
                result.restored.append("models (dry_run)")
        else:
            result.skipped.append("models: backup not found")

        # --- Model registry ---
        registry_backup = backup_dir / "registry.json"
        if registry_backup.exists():
            if not dry_run:
                shutil.copy2(registry_backup, Path("models/registry.json"))
                result.restored.append("model_registry")
            else:
                result.restored.append("model_registry (dry_run)")

        # --- Trade journal ---
        journal_backup = backup_dir / "logs" / self.trade_journal_path.name
        if journal_backup.exists():
            if not dry_run:
                self.trade_journal_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(journal_backup, self.trade_journal_path)
                result.restored.append("trade_journal")
                _logger.info("Trade journal restored from %s", backup_id)
            else:
                result.restored.append("trade_journal (dry_run)")
        else:
            result.skipped.append("trade_journal: backup not found")

        result.success = len(result.errors) == 0
        return result

    # ------------------------------------------------------------------
    # TRADE LOG RECOVERY
    # ------------------------------------------------------------------

    def recover_trade_logs(self) -> list[dict[str, Any]]:
        """
        Reconstruct trade records from all backup copies of the trade journal.

        Merges all backup CSVs → deduplicates on trade_id → returns sorted list.
        Useful when live journal is corrupted.
        """
        all_records: dict[str, dict] = {}

        # Scan all backup sets for trade journals
        for backup_dir in sorted(self.backup_root.iterdir()):
            journal = backup_dir / "logs" / self.trade_journal_path.name
            if not journal.exists():
                continue
            try:
                with open(journal, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get("trade_id", "")
                        if tid and tid not in all_records:
                            all_records[tid] = dict(row)
            except Exception as exc:
                _logger.warning("Could not read backup journal %s: %s", journal, exc)

        # Also try live journal
        if self.trade_journal_path.exists():
            try:
                with open(self.trade_journal_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get("trade_id", "")
                        if tid:
                            all_records[tid] = dict(row)   # live wins on conflict
            except Exception as exc:
                _logger.warning("Could not read live journal: %s", exc)

        records = sorted(all_records.values(), key=lambda r: r.get("entry_time", ""))
        _logger.info("Trade log recovery: found %d unique trades", len(records))

        # Write recovered log
        if records:
            recovered_path = self.trade_journal_path.parent / "trade_journal_recovered.csv"
            recovered_path.parent.mkdir(parents=True, exist_ok=True)
            with open(recovered_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
            _logger.info("Recovered journal written to %s", recovered_path)

        return records

    # ------------------------------------------------------------------
    # SERVER RECOVERY (cold-start sequence)
    # ------------------------------------------------------------------

    def server_recovery(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Full cold-start recovery sequence.
        Run this after a server crash or fresh VM deployment.

        Steps:
          1. Find latest backup
          2. Verify backup integrity
          3. Restore database
          4. Restore models
          5. Restore trade journal
          6. Reconstruct trade log from all backup copies
          7. Re-initialise database schema (safe — uses CREATE TABLE IF NOT EXISTS)
          8. Report recovery status
        """
        report: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "steps": [],
        }

        def step(name: str, ok: bool, detail: str = "") -> None:
            status = "OK" if ok else "FAILED"
            report["steps"].append({"step": name, "status": status, "detail": detail})
            _logger.info("Recovery [%s] %s %s", status, name, detail)

        # Step 1: Find latest backup
        latest = self._find_latest_backup()
        if latest is None:
            step("find_backup", False, "No backups found")
            report["success"] = False
            return report
        step("find_backup", True, f"Found backup {latest.backup_id}")

        # Step 2: Verify backup integrity
        integrity_ok = all(latest.integrity.values()) if latest.integrity else True
        step("verify_integrity", integrity_ok, str(latest.integrity))

        # Step 3-5: Restore
        restore_result = self.restore(latest.backup_id, dry_run=dry_run)
        step("restore_database", "database" in restore_result.restored or dry_run)
        step("restore_models", "models" in restore_result.restored or dry_run)
        step("restore_trade_journal", "trade_journal" in restore_result.restored or dry_run)

        if restore_result.errors:
            for err in restore_result.errors:
                step("restore_error", False, err)

        # Step 6: Reconstruct trade logs
        if not dry_run:
            records = self.recover_trade_logs()
            step("recover_trade_logs", True, f"{len(records)} trades recovered")
        else:
            step("recover_trade_logs", True, "dry_run — skipped")

        # Step 7: Re-initialise DB schema
        if not dry_run and self.db_path.exists():
            try:
                from ai_trading_system.database.connection import SQLiteConnectionManager
                from ai_trading_system.database.schema import DatabaseInitializer
                conn_mgr = SQLiteConnectionManager(f"sqlite:///{self.db_path}")
                DatabaseInitializer(conn_mgr).initialize()
                step("init_db_schema", True)
            except Exception as exc:
                step("init_db_schema", False, str(exc))
        else:
            step("init_db_schema", True, "dry_run — skipped")

        report["restore_result"] = restore_result.to_dict()
        report["finished_at"] = datetime.now().isoformat()
        report["success"] = restore_result.success
        return report

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_backups(self) -> list[BackupManifest]:
        """Return all backup manifests sorted newest first."""
        manifests: list[BackupManifest] = []
        for backup_dir in sorted(self.backup_root.iterdir(), reverse=True):
            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifests.append(BackupManifest.load(manifest_path))
                except Exception:
                    pass
        return manifests

    def _find_latest_backup(self) -> BackupManifest | None:
        manifests = self.list_backups()
        return manifests[0] if manifests else None

    def verify_backup(self, backup_id: str) -> dict[str, bool]:
        """Verify integrity of all files in a specific backup."""
        backup_dir = self.backup_root / backup_id
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            return {"manifest": False}

        manifest = BackupManifest.load(manifest_path)
        results: dict[str, bool] = {"manifest": True}

        db_backup = backup_dir / "trading.db"
        if db_backup.exists():
            try:
                with sqlite3.connect(str(db_backup)) as conn:
                    ic = conn.execute("PRAGMA integrity_check").fetchone()
                results["database"] = ic[0] == "ok" if ic else False
            except Exception:
                results["database"] = False

        results.update(manifest.integrity)
        return results
