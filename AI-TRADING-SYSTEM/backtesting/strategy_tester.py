from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.portfolio import Portfolio
from backtesting.results import BacktestMetrics


# ===========================================================================
# Built-in strategy implementations
# ===========================================================================

class IndicatorStrategy:
    """
    EMA crossover strategy with RSI confirmation.

    BUY  when fast EMA > slow EMA and RSI < rsi_oversold
    SELL when fast EMA < slow EMA or RSI > rsi_overbought
    """

    def __init__(
        self,
        fast_ema: str = "ema_20",
        slow_ema: str = "ema_50",
        rsi_col: str = "rsi_14",
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 65.0,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_col = rsi_col
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        fast = row.get(self.fast_ema)
        slow = row.get(self.slow_ema)
        rsi = float(row.get(self.rsi_col, 50))
        if fast is None or slow is None or pd.isna(fast) or pd.isna(slow):
            return "HOLD"
        if float(fast) > float(slow) and rsi < self.rsi_oversold:
            return "BUY"
        if float(fast) < float(slow) or rsi > self.rsi_overbought:
            return "SELL"
        return "HOLD"


class MACDStrategy:
    """MACD crossover strategy (histogram sign change)."""

    def __init__(self, macd_col: str = "macd", signal_col: str = "macd_signal") -> None:
        self.macd_col = macd_col
        self.signal_col = signal_col
        self._prev_diff: float | None = None

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        macd = row.get(self.macd_col)
        signal = row.get(self.signal_col)
        if macd is None or signal is None or pd.isna(macd) or pd.isna(signal):
            return "HOLD"
        diff = float(macd) - float(signal)
        result = "HOLD"
        if self._prev_diff is not None:
            if self._prev_diff <= 0 < diff:
                result = "BUY"
            elif self._prev_diff >= 0 > diff:
                result = "SELL"
        self._prev_diff = diff
        return result


class BollingerBandStrategy:
    """
    Mean-reversion strategy on Bollinger Bands.

    BUY  when close crosses below lower band
    SELL when close crosses above upper band
    """

    def __init__(
        self,
        lower_col: str = "bb_lower",
        upper_col: str = "bb_upper",
        close_col: str = "close",
    ) -> None:
        self.lower_col = lower_col
        self.upper_col = upper_col
        self.close_col = close_col

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        close = row.get(self.close_col)
        lower = row.get(self.lower_col)
        upper = row.get(self.upper_col)
        if any(x is None or pd.isna(x) for x in (close, lower, upper)):
            return "HOLD"
        if float(close) < float(lower):
            return "BUY"
        if float(close) > float(upper):
            return "SELL"
        return "HOLD"


class AIPredictionStrategy:
    """
    Uses pre-computed AI prediction columns in the DataFrame.

    Expected columns: 'ai_action' ('BUY'/'SELL'/'HOLD') and 'ai_confidence' (0–100).
    """

    def __init__(
        self,
        action_col: str = "ai_action",
        confidence_col: str = "ai_confidence",
        confidence_threshold: float = 60.0,
    ) -> None:
        self.action_col = action_col
        self.confidence_col = confidence_col
        self.confidence_threshold = confidence_threshold

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        action = str(row.get(self.action_col, "HOLD")).upper().strip()
        confidence = float(row.get(self.confidence_col, 0.0))
        if action in ("BUY", "SELL") and confidence >= self.confidence_threshold:
            return action
        return "HOLD"


class HybridStrategy:
    """
    Combines AI prediction with technical indicator confirmation.

    Modes:
      require_both=False (default): AI leads, indicator must not contradict.
      require_both=True           : Both must independently agree.
    """

    def __init__(
        self,
        ai_strategy: AIPredictionStrategy,
        indicator_strategy: IndicatorStrategy | None = None,
        require_both: bool = False,
    ) -> None:
        self.ai = ai_strategy
        self.indicator = indicator_strategy or IndicatorStrategy()
        self.require_both = require_both

    def generate_signal(self, row: pd.Series, portfolio: Portfolio) -> str:
        ai_sig = self.ai.generate_signal(row, portfolio)
        ind_sig = self.indicator.generate_signal(row, portfolio)

        if self.require_both:
            if ai_sig == ind_sig and ai_sig != "HOLD":
                return ai_sig
            return "HOLD"

        # AI leads; only filter if indicator explicitly contradicts
        if ai_sig == "BUY" and ind_sig != "SELL":
            return "BUY"
        if ai_sig == "SELL" and ind_sig != "BUY":
            return "SELL"
        return "HOLD"


# ===========================================================================
# Strategy Tester
# ===========================================================================

@dataclass
class StrategyResult:
    strategy_name: str
    metrics: BacktestMetrics
    trades_df: pd.DataFrame
    equity_df: pd.DataFrame

    def summary(self) -> str:
        return f"{self.strategy_name}\n{self.metrics.summary_str()}"


class StrategyTester:
    """Runs one or multiple strategies against historical data."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.engine = BacktestEngine(config)
        self.logger = logging.getLogger(__name__)

    def test(
        self,
        data: pd.DataFrame,
        strategy: object,
        strategy_name: str,
        symbol: str = "STOCK",
    ) -> StrategyResult:
        portfolio, metrics = self.engine.run(data, strategy, symbol)  # type: ignore[arg-type]
        trades_df = pd.DataFrame([t.to_dict() for t in portfolio.closed_trades()])
        equity_df = portfolio.equity_series()
        self.logger.info(
            "Strategy test complete",
            extra={"strategy": strategy_name, "return_pct": metrics.total_return_pct},
        )
        return StrategyResult(strategy_name, metrics, trades_df, equity_df)

    def test_indicator_strategy(self, data: pd.DataFrame, symbol: str = "STOCK") -> StrategyResult:
        return self.test(data, IndicatorStrategy(), "Indicator (EMA+RSI)", symbol)

    def test_macd_strategy(self, data: pd.DataFrame, symbol: str = "STOCK") -> StrategyResult:
        return self.test(data, MACDStrategy(), "Indicator (MACD)", symbol)

    def test_bollinger_strategy(self, data: pd.DataFrame, symbol: str = "STOCK") -> StrategyResult:
        return self.test(data, BollingerBandStrategy(), "Indicator (Bollinger Bands)", symbol)

    def test_ai_strategy(
        self,
        data: pd.DataFrame,
        symbol: str = "STOCK",
        confidence_threshold: float = 60.0,
    ) -> StrategyResult:
        strategy = AIPredictionStrategy(confidence_threshold=confidence_threshold)
        return self.test(data, strategy, "AI Prediction", symbol)

    def test_hybrid_strategy(
        self,
        data: pd.DataFrame,
        symbol: str = "STOCK",
        require_both: bool = False,
    ) -> StrategyResult:
        strategy = HybridStrategy(AIPredictionStrategy(), IndicatorStrategy(), require_both=require_both)
        name = "Hybrid (AI+Indicator, strict)" if require_both else "Hybrid (AI+Indicator)"
        return self.test(data, strategy, name, symbol)

    def test_all_indicator_strategies(
        self, data: pd.DataFrame, symbol: str = "STOCK"
    ) -> list[StrategyResult]:
        return [
            self.test_indicator_strategy(data, symbol),
            self.test_macd_strategy(data, symbol),
            self.test_bollinger_strategy(data, symbol),
        ]

    def test_all_strategies(
        self, data: pd.DataFrame, symbol: str = "STOCK"
    ) -> list[StrategyResult]:
        results = self.test_all_indicator_strategies(data, symbol)
        if "ai_action" in data.columns:
            results.append(self.test_ai_strategy(data, symbol))
            results.append(self.test_hybrid_strategy(data, symbol))
            results.append(self.test_hybrid_strategy(data, symbol, require_both=True))
        return results
