"""Automated backup — database, models, config, logs."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path


class BackupManager:
    """
    Creates timestamped zip backups of critical directories.

    Usage
    -----
    bm = BackupManager(backup_dir="backups")
    bm.backup_all()
    """

    def __init__(self, backup_dir: str | Path = "backups") -> None:
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def backup_database(self, db_path: str | Path = "trading.db") -> Path | None:
        return self._zip_file(Path(db_path), "db")

    def backup_models(self, models_dir: str | Path = "models") -> Path | None:
        return self._zip_dir(Path(models_dir), "models")

    def backup_config(self, config_dir: str | Path = ".") -> Path | None:
        files = list(Path(config_dir).glob("*.env")) + list(Path(config_dir).glob("*.cfg"))
        if not files:
            return None
        tag = self._timestamp_tag()
        dest = self._backup_dir / f"config_{tag}.zip"
        with __import__("zipfile").ZipFile(dest, "w") as zf:
            for f in files:
                zf.write(f, f.name)
        self.logger.info("Config backup: %s", dest)
        return dest

    def backup_logs(self, logs_dir: str | Path = "logs") -> Path | None:
        return self._zip_dir(Path(logs_dir), "logs")

    def backup_all(self) -> dict[str, Path | None]:
        return {
            "database": self.backup_database(),
            "models": self.backup_models(),
            "config": self.backup_config(),
            "logs": self.backup_logs(),
        }

    def list_backups(self) -> list[str]:
        return sorted(p.name for p in self._backup_dir.glob("*.zip"))

    # ------------------------------------------------------------------

    def _zip_file(self, src: Path, tag_prefix: str) -> Path | None:
        if not src.exists():
            return None
        tag = self._timestamp_tag()
        dest = self._backup_dir / f"{tag_prefix}_{tag}.zip"
        with __import__("zipfile").ZipFile(dest, "w") as zf:
            zf.write(src, src.name)
        self.logger.info("Backup created: %s", dest)
        return dest

    def _zip_dir(self, src: Path, tag_prefix: str) -> Path | None:
        if not src.exists():
            return None
        tag = self._timestamp_tag()
        dest = self._backup_dir / f"{tag_prefix}_{tag}"
        shutil.make_archive(str(dest), "zip", str(src.parent), src.name)
        self.logger.info("Backup created: %s.zip", dest)
        return Path(str(dest) + ".zip")

    @staticmethod
    def _timestamp_tag() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
