"""
Model Drift Detector — Phase 11G.

Detects feature drift, data drift, and prediction drift using
Population Stability Index (PSI) and distribution comparison.
No scipy dependency — uses numpy only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


_PSI_THRESHOLD_WARN = 0.1    # slight drift
_PSI_THRESHOLD_ALERT = 0.25  # significant drift → retrain


@dataclass
class DriftReport:
    feature_drift: dict[str, float]       # feature → PSI score
    data_drift: float                     # overall mean PSI
    prediction_drift: float               # shift in predicted distribution
    alerts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_critical_drift(self) -> bool:
        return bool(self.alerts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_drift": {k: round(v, 4) for k, v in self.feature_drift.items()},
            "data_drift": round(self.data_drift, 4),
            "prediction_drift": round(self.prediction_drift, 4),
            "alerts": self.alerts,
            "warnings": self.warnings,
            "has_critical_drift": self.has_critical_drift,
        }


class DriftDetector:
    """
    Compares a reference dataset (training distribution) to a current dataset.

    Usage
    -----
    detector = DriftDetector()
    detector.fit_reference(train_df)
    report = detector.detect(current_df, current_predictions, reference_predictions)
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._reference: pd.DataFrame | None = None
        self._ref_pred_dist: np.ndarray | None = None
        self.logger = logging.getLogger(__name__)

    def fit_reference(self, reference_df: pd.DataFrame, predictions: np.ndarray | None = None) -> None:
        self._reference = reference_df.copy()
        if predictions is not None:
            self._ref_pred_dist = np.array(predictions, dtype=float)

    def detect(
        self,
        current_df: pd.DataFrame,
        current_predictions: np.ndarray | None = None,
        features: list[str] | None = None,
    ) -> DriftReport:
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before detect()")

        cols = features or [c for c in self._reference.columns if c in current_df.columns]
        feature_psi: dict[str, float] = {}
        alerts: list[str] = []
        warnings: list[str] = []

        for col in cols:
            try:
                psi = self._psi(
                    self._reference[col].dropna().values,
                    current_df[col].dropna().values,
                )
                feature_psi[col] = psi
                if psi > _PSI_THRESHOLD_ALERT:
                    alerts.append(f"CRITICAL drift in '{col}': PSI={psi:.3f}")
                elif psi > _PSI_THRESHOLD_WARN:
                    warnings.append(f"Drift warning in '{col}': PSI={psi:.3f}")
            except Exception:
                pass

        data_drift = float(np.mean(list(feature_psi.values()))) if feature_psi else 0.0

        # Prediction drift
        pred_drift = 0.0
        if current_predictions is not None and self._ref_pred_dist is not None:
            pred_drift = self._psi(self._ref_pred_dist, np.array(current_predictions))
            if pred_drift > _PSI_THRESHOLD_ALERT:
                alerts.append(f"CRITICAL prediction drift: PSI={pred_drift:.3f}")
            elif pred_drift > _PSI_THRESHOLD_WARN:
                warnings.append(f"Prediction drift warning: PSI={pred_drift:.3f}")

        return DriftReport(
            feature_drift=feature_psi,
            data_drift=data_drift,
            prediction_drift=pred_drift,
            alerts=alerts,
            warnings=warnings,
        )

    def _psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index between two arrays."""
        bins = np.percentile(reference, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_pct = ref_counts / (ref_counts.sum() or 1) + 1e-8
        cur_pct = cur_counts / (cur_counts.sum() or 1) + 1e-8

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(abs(psi), 6)
