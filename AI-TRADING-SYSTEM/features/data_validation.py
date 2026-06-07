from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    null_counts: dict[str, int]
    duplicate_rows: int
    outlier_counts: dict[str, int]
    missing_candles: int
    is_valid: bool


class DataQualityValidator:
    """Creates validation reports for OHLCV feature inputs."""

    def __init__(self, expected_frequency: str = "D", outlier_zscore: float = 4.0) -> None:
        self.expected_frequency = expected_frequency
        self.outlier_zscore = outlier_zscore
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        if frame.empty:
            return ValidationReport(0, {}, 0, {}, 0, False)

        null_counts = {column: int(count) for column, count in frame.isna().sum().items() if int(count) > 0}
        duplicate_rows = int(frame.duplicated().sum())
        numeric = frame.select_dtypes(include="number")
        outlier_counts: dict[str, int] = {}
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty or series.std(ddof=0) == 0:
                outlier_counts[column] = 0
                continue
            zscores = ((series - series.mean()) / series.std(ddof=0)).abs()
            outlier_counts[column] = int((zscores > self.outlier_zscore).sum())

        missing_candles = self._missing_candle_count(frame)
        is_valid = not null_counts and duplicate_rows == 0 and missing_candles == 0
        self.logger.info("Data validation report generated", extra={"is_valid": is_valid})
        return ValidationReport(
            row_count=len(frame),
            null_counts=null_counts,
            duplicate_rows=duplicate_rows,
            outlier_counts=outlier_counts,
            missing_candles=missing_candles,
            is_valid=is_valid,
        )

    def _missing_candle_count(self, frame: pd.DataFrame) -> int:
        if "date" not in frame.columns:
            return 0
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
        if len(dates) < 2:
            return 0
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=self.expected_frequency)
        return max(len(expected.difference(pd.DatetimeIndex(dates))) - self._weekend_count(expected), 0)

    @staticmethod
    def _weekend_count(index: pd.DatetimeIndex) -> int:
        return int((index.dayofweek >= 5).sum())
