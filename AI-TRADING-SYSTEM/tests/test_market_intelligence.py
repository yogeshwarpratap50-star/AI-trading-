import pandas as pd

from market_intelligence.market_intelligence import MarketIntelligenceService


def test_market_intelligence_report_contains_rankings_and_score() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["^NSEI", "RELIANCE.NS", "INFY.NS", "TCS.NS"],
            "open": [22000, 2500, 1500, 3600],
            "close": [22150, 2550, 1480, 3620],
            "previous_close": [22000, 2480, 1510, 3600],
            "volume": [100000, 500000, 300000, 350000],
        }
    )

    report = MarketIntelligenceService().generate(frame, {"RELIANCE.NS": "Energy", "INFY.NS": "IT", "TCS.NS": "IT"})

    assert report.nifty_trend == "Bullish"
    assert "RELIANCE.NS" in report.top_gainers
    assert 0 <= report.market_health_score <= 100
    assert report.sector_trend["IT"] in {"Bullish", "Bearish", "Neutral"}
