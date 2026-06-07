from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_level: str = "INFO", log_dir: str | Path = "logs") -> None:
    """Configure console and file logging for all Phase 1 modules."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "application.log", encoding="utf-8"),
        ],
        force=True,
    )
