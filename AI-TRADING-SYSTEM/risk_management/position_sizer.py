from __future__ import annotations

import logging
import math
from enum import Enum


class SizingMethod(str, Enum):
    FIXED_PCT = "fixed_pct"         # fixed % of portfolio capital
    ATR_BASED = "atr_based"         # risk a fixed ₹ amount per ATR unit
    VOLATILITY = "volatility"       # inverse-volatility weighting
    KELLY = "kelly"                 # fractional Kelly criterion


class PositionSizer:
    """
    Calculates share quantities using four professional sizing methods.

    Parameters
    ----------
    capital:         Total available capital (₹).
    risk_per_trade:  Maximum % of capital to risk on one trade (default 1%).
    """

    def __init__(self, capital: float, risk_per_trade_pct: float = 1.0) -> None:
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.logger = logging.getLogger(__name__)

    def size(
        self,
        method: SizingMethod,
        price: float,
        *,
        atr: float | None = None,
        atr_multiplier: float = 2.0,
        volatility: float | None = None,   # annualised σ as a decimal, e.g. 0.25
        win_rate: float | None = None,     # for Kelly: historical win rate 0–1
        avg_win_loss_ratio: float | None = None,  # for Kelly
        stop_pct: float | None = None,     # for fixed-% stop
    ) -> int:
        """
        Returns integer share quantity (≥ 1 when capital permits, else 0).
        """
        if price <= 0 or self.capital <= 0:
            return 0

        risk_amount = self.capital * (self.risk_per_trade_pct / 100.0)

        if method == SizingMethod.FIXED_PCT:
            # Simple: allocate risk_per_trade_pct of capital to the position
            return max(1, int(risk_amount / price))

        if method == SizingMethod.ATR_BASED:
            if not atr or atr <= 0:
                self.logger.warning("ATR not provided for ATR-based sizing — falling back to fixed pct")
                return max(1, int(risk_amount / price))
            # Risk amount / (ATR * multiplier) = max shares before stop is hit
            stop_distance = atr * atr_multiplier
            return max(1, int(risk_amount / stop_distance))

        if method == SizingMethod.VOLATILITY:
            if not volatility or volatility <= 0:
                self.logger.warning("Volatility not provided — falling back to fixed pct")
                return max(1, int(risk_amount / price))
            # Scale position inversely with volatility; target 1% portfolio vol contribution
            target_vol_contribution = self.capital * 0.01
            shares = int(target_vol_contribution / (price * volatility))
            return max(1, shares)

        if method == SizingMethod.KELLY:
            if win_rate is None or avg_win_loss_ratio is None:
                self.logger.warning("Win rate / ratio not provided for Kelly — falling back to fixed pct")
                return max(1, int(risk_amount / price))
            # f* = (p * b - q) / b  where b = avg_win/avg_loss ratio, p = win rate, q = 1-p
            p = max(0.0, min(1.0, win_rate))
            q = 1.0 - p
            b = max(0.01, avg_win_loss_ratio)
            kelly_fraction = (p * b - q) / b
            # Use half-Kelly to reduce variance
            half_kelly = max(0.0, kelly_fraction / 2.0)
            allocation = self.capital * half_kelly
            return max(1, int(allocation / price))

        return max(1, int(risk_amount / price))

    # ------------------------------------------------------------------
    # Stop-loss / take-profit levels
    # ------------------------------------------------------------------

    @staticmethod
    def atr_stop(entry_price: float, atr: float, multiplier: float = 2.0, side: str = "BUY") -> float:
        """ATR-based stop-loss price."""
        if side == "BUY":
            return max(0.01, entry_price - atr * multiplier)
        return entry_price + atr * multiplier

    @staticmethod
    def trailing_stop(entry_price: float, highest_price: float, trail_pct: float = 2.0, side: str = "BUY") -> float:
        """Trailing stop that rises with the best price reached."""
        if side == "BUY":
            return highest_price * (1 - trail_pct / 100.0)
        return highest_price * (1 + trail_pct / 100.0)

    @staticmethod
    def pct_stop(entry_price: float, stop_pct: float = 2.0, side: str = "BUY") -> float:
        """Fixed percentage stop."""
        if side == "BUY":
            return entry_price * (1 - stop_pct / 100.0)
        return entry_price * (1 + stop_pct / 100.0)

    @staticmethod
    def volatility_stop(entry_price: float, daily_vol: float, days: int = 5, z: float = 2.0, side: str = "BUY") -> float:
        """Volatility-based stop: entry ± z * σ * √days."""
        vol_move = entry_price * daily_vol * math.sqrt(days) * z
        if side == "BUY":
            return max(0.01, entry_price - vol_move)
        return entry_price + vol_move

    # ------------------------------------------------------------------
    # Take-profit levels
    # ------------------------------------------------------------------

    @staticmethod
    def fixed_target(entry_price: float, target_pct: float = 3.0, side: str = "BUY") -> float:
        if side == "BUY":
            return entry_price * (1 + target_pct / 100.0)
        return entry_price * (1 - target_pct / 100.0)

    @staticmethod
    def atr_target(entry_price: float, atr: float, risk_reward: float = 2.0, multiplier: float = 2.0, side: str = "BUY") -> float:
        """Risk:Reward based take profit."""
        stop_distance = atr * multiplier
        if side == "BUY":
            return entry_price + stop_distance * risk_reward
        return entry_price - stop_distance * risk_reward
