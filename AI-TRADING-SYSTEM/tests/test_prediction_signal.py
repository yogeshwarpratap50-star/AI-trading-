import pandas as pd

from prediction.prediction_engine import PredictionEngine
from signals.signal_generator import SignalGenerator


class FakeModel:
    model_name = "fake_model"
    feature_columns = ["feature"]

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"0": [0.13], "1": [0.87]})


def test_prediction_engine_outputs_buy_with_confidence() -> None:
    result = PredictionEngine(model=FakeModel()).predict(pd.DataFrame({"feature": [1.0]}))

    assert result.action == "BUY"
    assert result.confidence == 87.0


def test_signal_generator_applies_buy_sell_hold_rules() -> None:
    generator = SignalGenerator()

    buy = generator.generate("BUY", 87, ema_20=105, ema_50=100, sentiment="Positive")
    hold = generator.generate("BUY", 70, ema_20=105, ema_50=100, sentiment="Positive")
    sell = generator.generate("SELL", 91, ema_20=95, ema_50=100, sentiment="Negative")

    assert buy.action == "BUY"
    assert hold.action == "HOLD"
    assert sell.action == "SELL"
