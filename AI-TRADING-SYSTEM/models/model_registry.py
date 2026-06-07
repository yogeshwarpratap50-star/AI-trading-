from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """File-backed model registry for versioned model metadata."""

    def __init__(self, registry_path: Path = Path("models/model_registry.json")) -> None:
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write([])

    def register(
        self,
        model_name: str,
        version: str,
        model_path: str | Path,
        metrics: dict[str, float],
        feature_set: list[str],
        is_best: bool = False,
    ) -> dict[str, Any]:
        entry = {
            "model_name": model_name,
            "version": version,
            "training_date": datetime.now(UTC).isoformat(),
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_set": feature_set,
            "is_best": is_best,
        }
        entries = self.entries()
        entries.append(entry)
        self._write(entries)
        return entry

    def entries(self) -> list[dict[str, Any]]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def latest(self, model_name: str | None = None) -> dict[str, Any] | None:
        entries = self.entries()
        if model_name:
            entries = [entry for entry in entries if entry["model_name"] == model_name]
        elif any(entry.get("is_best") for entry in entries):
            entries = [entry for entry in entries if entry.get("is_best")]
        if not entries:
            return None
        return sorted(entries, key=lambda entry: entry["training_date"])[-1]

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.registry_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
