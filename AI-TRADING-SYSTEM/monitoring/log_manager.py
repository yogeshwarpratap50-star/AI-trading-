"""Centralised log management — rotating file handlers for trade / prediction / system logs."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


_LOGGERS_CONFIGURED: set[str] = set()


def configure_log_manager(log_dir: str | Path = "logs", max_bytes: int = 10_485_760, backup_count: int = 5) -> None:
    """
    Set up rotating file handlers for key log channels.

    Call once at startup (idempotent — subsequent calls are no-ops).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    channels = {
        "trade": log_dir / "trade.log",
        "prediction": log_dir / "prediction.log",
        "system": log_dir / "system.log",
        "error": log_dir / "error.log",
    }

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")

    for channel, path in channels.items():
        if channel in _LOGGERS_CONFIGURED:
            continue
        logger = logging.getLogger(channel)
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        _LOGGERS_CONFIGURED.add(channel)

    # Error logger also gets a dedicated handler on the root
    err_handler = logging.handlers.RotatingFileHandler(
        channels["error"], maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(fmt)
    logging.getLogger().addHandler(err_handler)


def get_trade_logger() -> logging.Logger:
    return logging.getLogger("trade")


def get_prediction_logger() -> logging.Logger:
    return logging.getLogger("prediction")


def get_system_logger() -> logging.Logger:
    return logging.getLogger("system")
