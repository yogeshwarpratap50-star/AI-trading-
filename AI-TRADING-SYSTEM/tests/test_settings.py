from ai_trading_system.config.settings import Settings


def test_default_settings_are_valid() -> None:
    settings = Settings()

    assert settings.app_name == "AI-TRADING-SYSTEM"
    assert settings.default_history_years == 1
    assert settings.database_url.startswith("sqlite:///")
