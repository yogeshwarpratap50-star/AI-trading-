import pandas as pd
import pytest

from ai_trading_system.data_collection.exceptions import DataValidationError
from ai_trading_system.data_collection.validators import OHLCVValidator


def test_ohlcv_validator_cleans_and_sorts_rows() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2025-01-02", "open": "10", "high": "12", "low": "9", "close": "11", "volume": "100"},
            {"date": "2025-01-01", "open": "9", "high": "10", "low": "8", "close": "9.5", "volume": None},
        ]
    )

    result = OHLCVValidator().validate(frame)

    assert list(result["close"]) == [9.5, 11.0]
    assert result.loc[0, "volume"] == 0


def test_ohlcv_validator_rejects_empty_data() -> None:
    with pytest.raises(DataValidationError):
        OHLCVValidator().validate(pd.DataFrame())
