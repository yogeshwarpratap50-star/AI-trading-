"""
Portfolio Rebalancer — Phase 12.

Automatically:
  * Detects drift from target allocation
  * Increases cash during HIGH_VOLATILITY or BEAR regimes
  * Reduces over-concentrated positions
  * Generates rebalance orders without user input
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from autonomous.risk_profile import RiskProfileConfig
from advanced_ai.market_regime import Regime


@dataclass
class RebalanceAction:
    symbol: str
    action: str        # "SELL_PARTIAL" | "SELL_ALL" | "REDUCE_CASH" | "HOLD"
    reason: str
    current_pct: float
    target_pct: float
    quantity_delta: int   # shares to sell (positive = reduce)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "reason": self.reason,
            "current_pct": round(self.current_pct, 2),
            "target_pct": round(self.target_pct, 2),
            "quantity_delta": self.quantity_delta,
        }


class PortfolioRebalancer:
    """
    Evaluates current portfolio against target allocations and returns
    a list of actions to restore balance.

    Usage
    -----
    rebalancer = PortfolioRebalancer(profile)
    actions = rebalancer.evaluate(positions, total_value, regime)
    """

    def __init__(self, profile: RiskProfileConfig) -> None:
        self.profile = profile
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        positions: list[dict[str, Any]],
        total_value: float,
        regime: Regime | None = None,
    ) -> list[RebalanceAction]:
        if not positions or total_value <= 0:
            return []

        actions: list[RebalanceAction] = []
        regime = regime or Regime.SIDEWAYS

        # Determine target cash % based on regime
        target_cash_pct = self._regime_cash_target(regime)
        current_equity_value = sum(float(p.get("market_value", 0)) for p in positions)
        current_cash_pct = max(0.0, (total_value - current_equity_value) / total_value * 100)

        # Cash is too low and regime is risky → increase cash
        if current_cash_pct < target_cash_pct - 2:   # 2% tolerance
            self.logger.info(
                "Rebalance: cash %.1f%% < target %.1f%% — reducing exposure",
                current_cash_pct, target_cash_pct,
            )
            # Find over-sized positions to trim first
            for pos in sorted(positions, key=lambda p: float(p.get("market_value", 0)), reverse=True):
                mv = float(pos.get("market_value", 0))
                current_pct = mv / total_value * 100
                max_allowed = self.profile.max_single_position_pct

                if current_pct > max_allowed:
                    excess_pct = current_pct - max_allowed
                    target_pct = max_allowed
                    price = float(pos.get("last_price", 1))
                    qty_to_sell = max(1, int((mv * excess_pct / 100) / price))
                    actions.append(RebalanceAction(
                        symbol=pos["symbol"],
                        action="SELL_PARTIAL",
                        reason=f"Over-concentrated ({current_pct:.1f}% > {max_allowed}%)",
                        current_pct=current_pct,
                        target_pct=target_pct,
                        quantity_delta=qty_to_sell,
                    ))

        # High volatility regime — sell smallest/weakest positions
        if regime == Regime.HIGH_VOLATILITY and len(positions) > 2:
            worst = sorted(positions, key=lambda p: float(p.get("unrealized_pnl", 0)))[0]
            mv = float(worst.get("market_value", 0))
            if mv > 0:
                pct = mv / total_value * 100
                price = float(worst.get("last_price", 1))
                qty = max(1, int(mv / price))
                actions.append(RebalanceAction(
                    symbol=worst["symbol"],
                    action="SELL_ALL",
                    reason="HIGH_VOLATILITY regime — reducing positions",
                    current_pct=pct,
                    target_pct=0.0,
                    quantity_delta=qty,
                ))

        # Bear regime — tighten exposure across all positions
        if regime == Regime.BEAR:
            bear_cap = self.profile.max_single_position_pct * 0.6   # cut max size by 40%
            for pos in positions:
                mv = float(pos.get("market_value", 0))
                current_pct = mv / total_value * 100
                if current_pct > bear_cap:
                    price = float(pos.get("last_price", 1))
                    excess = mv * (current_pct - bear_cap) / 100 / current_pct * mv
                    qty_to_sell = max(1, int(excess / price))
                    actions.append(RebalanceAction(
                        symbol=pos["symbol"],
                        action="SELL_PARTIAL",
                        reason=f"BEAR regime cap ({bear_cap:.1f}%)",
                        current_pct=current_pct,
                        target_pct=bear_cap,
                        quantity_delta=qty_to_sell,
                    ))

        # Unconditional concentration cap — regardless of cash level
        max_allowed = self.profile.max_single_position_pct
        for pos in positions:
            mv = float(pos.get("market_value", 0))
            current_pct = mv / total_value * 100
            if current_pct > max_allowed:
                existing = [a for a in actions if a.symbol == pos["symbol"]]
                if not existing:
                    price = float(pos.get("last_price", 1))
                    excess_pct = current_pct - max_allowed
                    qty_to_sell = max(1, int((mv * excess_pct / 100) / price))
                    actions.append(RebalanceAction(
                        symbol=pos["symbol"],
                        action="SELL_PARTIAL",
                        reason=f"Over-concentrated ({current_pct:.1f}% > {max_allowed}%)",
                        current_pct=current_pct,
                        target_pct=max_allowed,
                        quantity_delta=qty_to_sell,
                    ))

        # Drift threshold check — rebalance if any position drifted too far
        threshold = self.profile.rebalance_threshold_pct
        equal_target = 100.0 / len(positions) if positions else 100.0
        for pos in positions:
            mv = float(pos.get("market_value", 0))
            current_pct = mv / total_value * 100
            if abs(current_pct - equal_target) > threshold:
                if current_pct > equal_target:
                    price = float(pos.get("last_price", 1))
                    excess = (current_pct - equal_target) / 100 * total_value
                    qty = max(1, int(excess / price))
                    existing = [a for a in actions if a.symbol == pos["symbol"]]
                    if not existing:
                        actions.append(RebalanceAction(
                            symbol=pos["symbol"],
                            action="SELL_PARTIAL",
                            reason=f"Drift {current_pct:.1f}% vs target {equal_target:.1f}%",
                            current_pct=current_pct,
                            target_pct=equal_target,
                            quantity_delta=qty,
                        ))

        return actions

    def _regime_cash_target(self, regime: Regime) -> float:
        """How much cash to keep (%) based on market regime."""
        base = self.profile.cash_reserve_pct
        adjustments = {
            Regime.BULL: 0,
            Regime.SIDEWAYS: 5,
            Regime.LOW_VOLATILITY: 0,
            Regime.BEAR: 20,
            Regime.HIGH_VOLATILITY: 30,
        }
        return base + adjustments.get(regime, 0)
