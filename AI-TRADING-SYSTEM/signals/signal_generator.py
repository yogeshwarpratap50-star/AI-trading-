from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingSignal:
    action: str
    confidence: float
    reason: str


class SignalGenerator:
    """Applies AI, trend, and sentiment rules to produce trading signals."""

    def __init__(self, confidence_threshold: float = 80.0) -> None:
        self.confidence_threshold = confidence_threshold

    def generate(
        self,
        ai_action: str,
        confidence: float,
        ema_20: float,
        ema_50: float,
        sentiment: str,
    ) -> TradingSignal:
        sentiment_normalized = sentiment.lower()
        if confidence > self.confidence_threshold and ai_action == "BUY" and ema_20 > ema_50 and sentiment_normalized == "positive":
            return TradingSignal("BUY", confidence, "AI confidence, EMA trend, and sentiment are bullish.")
        if confidence > self.confidence_threshold and ai_action == "SELL" and ema_20 < ema_50 and sentiment_normalized == "negative":
            return TradingSignal("SELL", confidence, "AI confidence, EMA trend, and sentiment are bearish.")
        return TradingSignal("HOLD", confidence, "Rules did not confirm a high-confidence directional trade.")
