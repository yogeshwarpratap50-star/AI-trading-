from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import pandas as pd


class IndicatorInputError(ValueError):
    """Raised when indicator input data is invalid."""


class Indicator(ABC):
    """Base class for technical indicators."""

    required_columns: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of frame with indicator columns added."""

    def _validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise IndicatorInputError("Input DataFrame cannot be empty.")

        missing = set(self.required_columns).difference(frame.columns)
        if missing:
            raise IndicatorInputError(f"Missing required columns: {sorted(missing)}")

        clean = frame.copy()
        for column in self.required_columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        return clean.sort_index()
