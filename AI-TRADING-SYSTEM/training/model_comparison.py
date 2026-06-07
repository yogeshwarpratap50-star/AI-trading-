from __future__ import annotations

from pathlib import Path

import pandas as pd


class ModelComparison:
    """Compares model metrics and selects the best model."""

    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare(self, results: dict[str, dict[str, float]], primary_metric: str = "f1") -> tuple[str, pd.DataFrame]:
        frame = pd.DataFrame.from_dict(results, orient="index").reset_index(names="model_name")
        frame = frame.sort_values(primary_metric, ascending=False).reset_index(drop=True)
        frame.to_csv(self.output_dir / "model_performance_report.csv", index=False)
        frame.to_html(self.output_dir / "model_performance_report.html", index=False)
        return str(frame.loc[0, "model_name"]), frame
