from ai_trading_system.database.connection import SQLiteConnectionManager
from ai_trading_system.database.schema import DatabaseInitializer


def test_database_initializer_creates_tables(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    manager = SQLiteConnectionManager(db_url)

    DatabaseInitializer(manager).initialize()

    with manager.connect() as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    table_names = {row["name"] for row in rows}
    assert "historical_prices" in table_names
    assert "live_quotes" in table_names
    assert "data_collection_runs" in table_names
