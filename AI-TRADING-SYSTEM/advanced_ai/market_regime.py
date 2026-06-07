"""
Market Regime Detection — Phase 11D.

Classifies current market into one of five regimes using price/volatility signals.
No external dependencies beyond numpy/pandas.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float          # 0–1
    trend_strength: float      # positive = bullish, negative = bearish
    volatility_pct: float      # annualised volatility
    adx: float                 # trend strength (ADX-like)
    details: dict

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "trend_strength": round(self.trend_strength, 4),
            "volatility_pct": round(self.volatility_pct, 2),
            "adx": round(self.adx, 2),
            **self.details,
        }


class MarketRegimeDetector:
    """
    Detects market regime from OHLCV DataFrame.

    Parameters
    ----------
    fast_period : EMA period for short trend
    slow_period : EMA period for long trend
    vol_lookback : rolling window for volatility
    adx_period   : ADX calculation window
    vol_high_pct : annualised vol threshold above which → HIGH_VOLATILITY
    vol_low_pct  : annualised vol threshold below which → LOW_VOLATILITY
    trend_thresh : minimum |trend_strength| to classify as BULL/BEAR vs SIDEWAYS
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        vol_lookback: int = 20,
        adx_period: int = 14,
        vol_high_pct: float = 30.0,
        vol_low_pct: float = 10.0,
        trend_thresh: float = 0.005,
    ) -> None:
        self.fast = fast_period
        self.slow = slow_period
        self.vol_lookback = vol_lookback
        self.adx_period = adx_period
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self.trend_thresh = trend_thresh

    def detect(self, ohlcv: pd.DataFrame) -> RegimeResult:
        """
        Detect regime from a DataFrame with columns: open, high, low, close, volume.
        Returns the regime for the most recent data point.
        """
        df = ohlcv.copy()
        df.columns = [c.lower() for c in df.columns]
        close = df["close"].astype(float)

        # Trend via EMA crossover
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        trend_strength = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1])

        # Volatility (annualised)
        returns = close.pct_change().dropna()
        vol_daily = float(returns.tail(self.vol_lookback).std())
        vol_annual = vol_daily * np.sqrt(252) * 100  # %

        # ADX-like metric (simplified directional movement)
        adx = self._adx(df)

        # Classification logic
        regime, confidence = self._classify(trend_strength, vol_annual, adx)

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_pct=vol_annual,
            adx=adx,
            details={
                "ema_fast": round(float(ema_fast.iloc[-1]), 2),
                "ema_slow": round(float(ema_slow.iloc[-1]), 2),
                "vol_daily_pct": round(vol_daily * 100, 3),
            },
        )

    def _classify(self, trend: float, vol: float, adx: float) -> tuple[Regime, float]:
        # Volatility overrides trend when extreme
        if vol > self.vol_high_pct:
            conf = min(1.0, (vol - self.vol_high_pct) / self.vol_high_pct + 0.5)
            return Regime.HIGH_VOLATILITY, round(conf, 3)
        if vol < self.vol_low_pct:
            conf = min(1.0, (self.vol_low_pct - vol) / self.vol_low_pct + 0.5)
            return Regime.LOW_VOLATILITY, round(conf, 3)

        # Trend classification
        if trend > self.trend_thresh and adx > 20:
            conf = min(1.0, abs(trend) / self.trend_thresh * 0.5 + 0.4)
            return Regime.BULL, round(conf, 3)
        if trend < -self.trend_thresh and adx > 20:
            conf = min(1.0, abs(trend) / self.trend_thresh * 0.5 + 0.4)
            return Regime.BEAR, round(conf, 3)
        return Regime.SIDEWAYS, round(0.5 + (1 - adx / 100) * 0.3, 3)

    def _adx(self, df: pd.DataFrame) -> float:
        """Simplified ADX approximation using True Range ratio."""
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.adx_period).mean()
            dm_plus = (high.diff()).clip(lower=0)
            dm_minus = (-low.diff()).clip(lower=0)
            di_plus = (dm_plus.rolling(self.adx_period).mean() / atr.replace(0, np.nan)) * 100
            di_minus = (dm_minus.rolling(self.adx_period).mean() / atr.replace(0, np.nan)) * 100
            dx = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)) * 100
            adx = dx.rolling(self.adx_period).mean().iloc[-1]
            return float(adx) if not np.isnan(adx) else 0.0
        except Exception:
            return 0.0
