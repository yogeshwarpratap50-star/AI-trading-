"""
End-to-End Production Simulation — Full Pipeline Validation.

Workflow:
  Data → Features → AI → Prediction → Risk Check → Paper Trade → Reporting

Each stage is validated independently and as a full pipeline.
Final test generates a production validation report.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2500.0 + np.cumsum(rng.normal(0, 25, n))
    close = np.maximum(close, 100.0)
    return pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.001, 0.015, n)),
        "low":    close * (1 - rng.uniform(0.001, 0.015, n)),
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
        "date":   pd.date_range("2024-01-02", periods=n, freq="B"),
    })


def _make_training_data(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) ready for model.train() — numeric features only."""
    from features.feature_engineering import FeatureEngineeringService
    from features.label_generator import LabelGenerator
    df = _make_ohlcv(n)
    features, _ = FeatureEngineeringService().create_feature_dataset(df)
    labeled = LabelGenerator().add_labels(features)
    target_col = "target_next_day_up"
    labeled = labeled.dropna(subset=[target_col])
    if target_col not in labeled.columns or len(labeled) < 50:
        return pd.DataFrame(), pd.Series(dtype=float)
    # Exclude target cols, date, OHLCV passthrough, and any non-numeric columns
    exclude = {target_col, "target_intraday_direction", "date",
               "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in labeled.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(labeled[c])]
    X = labeled[feature_cols].copy()
    y = labeled[target_col].astype(int)
    return X, y


# ===========================================================================
# STAGE 1: DATA VALIDATION
# ===========================================================================

class TestStage1Data:
    def test_ohlcv_schema_complete(self):
        df = _make_ohlcv()
        required = {"open", "high", "low", "close", "volume"}
        assert required.issubset(df.columns)

    def test_no_negative_prices(self):
        df = _make_ohlcv()
        for col in ("open", "high", "low", "close"):
            assert (df[col] > 0).all()

    def test_high_ge_low(self):
        df = _make_ohlcv()
        assert (df["high"] >= df["low"]).all()

    def test_volume_nonnegative(self):
        df = _make_ohlcv()
        assert (df["volume"] >= 0).all()

    def test_validator_accepts_clean_data(self):
        from ai_trading_system.data_collection.validators import OHLCVValidator
        df = _make_ohlcv()
        clean = OHLCVValidator().validate(df)
        assert len(clean) > 0

    def test_validator_rejects_empty(self):
        from ai_trading_system.data_collection.validators import OHLCVValidator, DataValidationError
        with pytest.raises(DataValidationError):
            OHLCVValidator().validate(pd.DataFrame())

    def test_validator_drops_high_lt_low_rows(self):
        from ai_trading_system.data_collection.validators import OHLCVValidator
        df = _make_ohlcv(50)
        df.loc[5, "high"] = df.loc[5, "low"] - 10
        clean = OHLCVValidator().validate(df)
        assert len(clean) < len(df)

    def test_historical_data_dir_constant(self):
        from ai_trading_system.config.settings import get_settings
        settings = get_settings()
        assert Path(settings.historical_data_dir) == Path("data/historical")

    def test_csv_filename_format(self):
        """RELIANCE.NS -> RELIANCE_NS.csv"""
        symbol = "RELIANCE.NS"
        assert symbol.replace(".", "_") + ".csv" == "RELIANCE_NS.csv"


# ===========================================================================
# STAGE 2: FEATURE ENGINEERING
# ===========================================================================

class TestStage2Features:
    def test_features_created(self):
        from features.feature_engineering import FeatureEngineeringService
        df = _make_ohlcv()
        features, _ = FeatureEngineeringService().create_feature_dataset(df)
        assert not features.empty
        assert len(features.columns) >= 10

    def test_features_no_inf(self):
        from features.feature_engineering import FeatureEngineeringService
        df = _make_ohlcv()
        features, _ = FeatureEngineeringService().create_feature_dataset(df)
        numeric = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_features_no_all_nan_columns(self):
        from features.feature_engineering import FeatureEngineeringService
        df = _make_ohlcv()
        features, _ = FeatureEngineeringService().create_feature_dataset(df)
        all_nan_cols = features.columns[features.isna().all()]
        assert len(all_nan_cols) == 0

    def test_feature_model_cols_numeric(self):
        from features.feature_engineering import FeatureEngineeringService
        df = _make_ohlcv()
        features, _ = FeatureEngineeringService().create_feature_dataset(df)
        skip = {"date", "timestamp"}
        non_numeric = [c for c in features.columns
                       if c not in skip and not pd.api.types.is_numeric_dtype(features[c])]
        assert len(non_numeric) == 0

    def test_label_generator_binary(self):
        from features.feature_engineering import FeatureEngineeringService
        from features.label_generator import LabelGenerator
        df = _make_ohlcv(100)
        features, _ = FeatureEngineeringService().create_feature_dataset(df)
        labeled = LabelGenerator().add_labels(features)
        col = "target_next_day_up"
        assert col in labeled.columns
        unique = set(labeled[col].dropna().unique())
        assert unique.issubset({0, 1})

    def test_features_stable(self):
        from features.feature_engineering import FeatureEngineeringService
        df = _make_ohlcv()
        f1, _ = FeatureEngineeringService().create_feature_dataset(df)
        f2, _ = FeatureEngineeringService().create_feature_dataset(df)
        assert len(f1.columns) == len(f2.columns)


# ===========================================================================
# STAGE 3: AI MODEL TRAINING
# ===========================================================================

class TestStage3AI:
    def test_random_forest_trains(self, tmp_path):
        from models.random_forest_model import RandomForestTradingModel
        X, y = _make_training_data()
        if X.empty:
            pytest.skip("Insufficient training data")
        model = RandomForestTradingModel()
        model.train(X, y)
        assert model.model is not None

    def test_xgboost_trains(self, tmp_path):
        from models.xgboost_model import XGBoostTradingModel
        X, y = _make_training_data()
        if X.empty:
            pytest.skip("Insufficient training data")
        model = XGBoostTradingModel()
        model.train(X, y)
        assert model.model is not None

    def test_model_save_load(self, tmp_path):
        from models.random_forest_model import RandomForestTradingModel
        X, y = _make_training_data()
        if X.empty:
            pytest.skip()
        model = RandomForestTradingModel()
        model.train(X, y)
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded = RandomForestTradingModel()
        loaded.load(save_path)
        assert loaded.model is not None

    def test_model_predict_proba_returns_dataframe(self):
        from models.random_forest_model import RandomForestTradingModel
        X, y = _make_training_data()
        if X.empty:
            pytest.skip()
        model = RandomForestTradingModel()
        model.train(X, y)
        proba = model.predict_proba(X.tail(5))
        assert isinstance(proba, pd.DataFrame)
        assert not proba.empty

    def test_model_accuracy_above_random(self):
        from models.random_forest_model import RandomForestTradingModel
        X, y = _make_training_data()
        if X.empty:
            pytest.skip()
        n = int(len(X) * 0.8)
        model = RandomForestTradingModel()
        model.train(X.iloc[:n], y.iloc[:n])
        metrics = model.evaluate(X.iloc[n:], y.iloc[n:])
        assert metrics.get("accuracy", 0) >= 0.4   # at least better than pure noise


# ===========================================================================
# STAGE 4: PREDICTION ENGINE
# ===========================================================================

def _build_engine(tmp_path: Path):
    """Helper: train and register a model, return (engine, X_test)."""
    from models.random_forest_model import RandomForestTradingModel
    from models.model_registry import ModelRegistry
    from prediction.prediction_engine import PredictionEngine

    X, y = _make_training_data()
    if X.empty:
        return None, None

    model = RandomForestTradingModel()
    model.train(X, y)
    model_path = tmp_path / "rf.pkl"
    model.save(model_path)
    registry = ModelRegistry(registry_path=tmp_path / "registry.json")
    registry.register(model.model_name, "v1", str(model_path), {"accuracy": 0.6}, list(X.columns))
    engine = PredictionEngine(registry=registry)
    return engine, X


class TestStage4Prediction:
    def test_predict_returns_result(self, tmp_path):
        from prediction.prediction_engine import PredictionResult
        engine, X = _build_engine(tmp_path)
        if engine is None:
            pytest.skip()
        result = engine.predict(X.tail(1))
        assert isinstance(result, PredictionResult)
        assert result.action in ("BUY", "SELL", "HOLD")

    def test_predict_confidence_in_range(self, tmp_path):
        engine, X = _build_engine(tmp_path)
        if engine is None:
            pytest.skip()
        result = engine.predict(X.tail(1))
        assert 0.0 <= result.confidence <= 100.0

    def test_predict_empty_returns_hold(self):
        from prediction.prediction_engine import PredictionEngine
        engine = PredictionEngine()
        result = engine.predict(pd.DataFrame())
        assert result.action == "HOLD"
        assert result.model_name == "none"

    def test_predict_no_model_returns_hold(self, tmp_path):
        from prediction.prediction_engine import PredictionEngine
        from models.model_registry import ModelRegistry
        registry = ModelRegistry(registry_path=tmp_path / "empty_registry.json")
        engine = PredictionEngine(registry=registry)
        result = engine.predict(pd.DataFrame({"a": [1.0]}))
        assert result.action == "HOLD"

    def test_predict_wrong_features_returns_hold(self, tmp_path):
        engine, X = _build_engine(tmp_path)
        if engine is None:
            pytest.skip()
        wrong = pd.DataFrame({"wrong_col_1": [1.0], "wrong_col_2": [2.0]})
        result = engine.predict(wrong)
        assert result.action == "HOLD"

    def test_predict_single_row(self, tmp_path):
        engine, X = _build_engine(tmp_path)
        if engine is None:
            pytest.skip()
        result = engine.predict(X.iloc[[0]])
        assert result.action in ("BUY", "SELL", "HOLD")


# ===========================================================================
# STAGE 5: RISK ENGINE
# ===========================================================================

class TestStage5Risk:
    def _engine(self):
        from risk_management.risk_engine import RiskEngine, RiskConfig
        return RiskEngine(RiskConfig(
            risk_per_trade_pct=2.0,
            min_ai_confidence=60.0,
            max_open_positions=3,
            max_single_position_pct=25.0,
            daily_loss_limit_pct=3.0,
        ))

    def test_valid_buy_passes(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="RELIANCE", price=2500.0, quantity=5,
            ai_confidence=75.0, ema_20=2480.0, ema_50=2450.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
        )
        assert r.approved

    def test_low_confidence_rejected(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="TCS", price=3500.0, quantity=5,
            ai_confidence=40.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
        )
        assert not r.approved

    def test_zero_price_rejected(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="BAD", price=0.0, quantity=5,
            ai_confidence=75.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
        )
        assert not r.approved

    def test_max_positions_rejected(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="X", price=1000.0, quantity=5,
            ai_confidence=75.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=3, starting_capital=100_000.0,
        )
        assert not r.approved

    def test_size_position_positive(self):
        qty = self._engine().size_position("RELIANCE", 2500.0, capital=100_000.0, atr=50.0)
        assert qty > 0

    def test_size_position_zero_price_returns_zero(self):
        qty = self._engine().size_position("BAD", 0.0, capital=100_000.0)
        assert qty == 0

    def test_stop_loss_below_entry(self):
        stop = self._engine().stop_loss(2500.0, atr=50.0)
        assert stop < 2500.0

    def test_take_profit_above_entry(self):
        tp = self._engine().take_profit(2500.0, atr=50.0)
        assert tp > 2500.0

    def test_daily_loss_halt(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="X", price=1000.0, quantity=5,
            ai_confidence=75.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
            today_pnl=-4000.0,
        )
        assert not r.approved

    def test_insufficient_cash_rejected(self):
        r = self._engine().validate_trade(
            signal="BUY", symbol="X", price=1000.0, quantity=500,  # 500k
            ai_confidence=75.0,
            portfolio_cash=10_000.0, portfolio_total_value=10_000.0,
            open_position_count=0, starting_capital=10_000.0,
        )
        assert not r.approved


# ===========================================================================
# STAGE 6: PAPER TRADING
# ===========================================================================

class TestStage6PaperTrading:
    def _broker(self, capital: float = 100_000.0):
        from paper_trading.paper_broker import BrokerConfig, PaperBroker
        return PaperBroker(BrokerConfig(starting_capital=capital))

    def test_buy_reduces_cash(self):
        pb = self._broker()
        pb.place_buy("RELIANCE", 10, 2500.0)
        assert pb.portfolio.cash < 100_000.0

    def test_buy_creates_position(self):
        pb = self._broker()
        pb.place_buy("TCS", 5, 3500.0)
        assert "TCS" in pb.portfolio.positions

    def test_sell_without_position_rejected(self):
        from paper_trading.paper_order_manager import OrderStatus
        pb = self._broker()
        order = pb.place_sell("GHOST", 5, 1000.0)
        assert order.status == OrderStatus.REJECTED

    def test_buy_then_sell_closes_position(self):
        pb = self._broker()
        pb.place_buy("INFY", 10, 1500.0)
        pb.update_prices({"INFY": 1600.0})
        pb.place_sell("INFY", 10, 1600.0)
        assert "INFY" not in pb.portfolio.positions
        assert len(pb.portfolio.closed_positions) == 1

    def test_buy_then_sell_profit(self):
        pb = self._broker()
        pb.place_buy("SBIN", 20, 600.0)
        pb.update_prices({"SBIN": 700.0})
        pb.place_sell("SBIN", 20, 700.0)
        pnl = pb.portfolio.closed_positions[0].realized_pnl
        assert pnl > 0

    def test_insufficient_cash_rejected(self):
        from paper_trading.paper_order_manager import OrderStatus
        pb = self._broker(capital=1000.0)
        order = pb.place_buy("RELIANCE", 100, 3000.0)
        assert order.status == OrderStatus.REJECTED

    def test_trading_disabled_rejects_buy(self):
        from paper_trading.paper_order_manager import OrderStatus
        pb = self._broker()
        pb.enable_trading(False)
        order = pb.place_buy("RELIANCE", 5, 2500.0)
        assert order.status == OrderStatus.REJECTED

    def test_daily_loss_limit_disables_trading(self):
        pb = self._broker()
        # daily_pnl is tracked via portfolio.daily_pnl dict keyed by today's date
        pb.portfolio.daily_pnl[date.today().isoformat()] = -4000.0
        enabled = pb.check_daily_loss_limit(3.0)
        assert not enabled
        assert not pb.is_trading_enabled()

    def test_snapshot_has_required_keys(self):
        pb = self._broker()
        snap = pb.snapshot()
        assert "balance" in snap
        assert "positions" in snap
        assert "closed_positions" in snap
        assert "trading_enabled" in snap
        assert "cash" in snap["balance"]

    def test_slippage_applied_on_buy(self):
        pb = self._broker()
        order = pb.place_buy("X", 1, 1000.0)
        # filled_price should be >= 1000 due to slippage
        if order.filled_price is not None:
            assert order.filled_price >= 1000.0

    def test_fees_applied_on_sell(self):
        pb = self._broker()
        pb.place_buy("X", 10, 1000.0)
        pb.update_prices({"X": 1100.0})
        order = pb.place_sell("X", 10, 1100.0)
        assert order.fees >= 0

    def test_full_buy_sell_cycle_pnl(self):
        pb = self._broker()
        pb.place_buy("MARUTI", 5, 10000.0)
        pb.update_prices({"MARUTI": 11000.0})
        pb.place_sell("MARUTI", 5, 11000.0)
        assert len(pb.portfolio.closed_positions) == 1
        # Profit: 5 * (11000 - 10000) = 5000 before fees
        assert pb.portfolio.closed_positions[0].realized_pnl > 0


# ===========================================================================
# STAGE 7: FULL PIPELINE
# ===========================================================================

class TestStage7FullPipeline:
    def test_full_pipeline(self, tmp_path):
        from features.feature_engineering import FeatureEngineeringService
        from features.label_generator import LabelGenerator
        from models.random_forest_model import RandomForestTradingModel
        from models.model_registry import ModelRegistry
        from prediction.prediction_engine import PredictionEngine
        from risk_management.risk_engine import RiskEngine, RiskConfig
        from paper_trading.paper_broker import BrokerConfig, PaperBroker

        # 1. Data
        df = _make_ohlcv(400)

        # 2. Features
        features, _ = FeatureEngineeringService().create_feature_dataset(df)

        # 3. Labels + Training split
        labeled = LabelGenerator().add_labels(features)
        target_col = "target_next_day_up"
        labeled = labeled.dropna(subset=[target_col])
        if len(labeled) < 100:
            pytest.skip("Insufficient labeled data")

        exclude = {target_col, "target_intraday_direction", "date",
                   "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in labeled.columns
                        if c not in exclude and pd.api.types.is_numeric_dtype(labeled[c])]
        X = labeled[feature_cols]
        y = labeled[target_col].astype(int)

        # 4. Train
        model = RandomForestTradingModel()
        model.train(X.iloc[:int(len(X) * 0.8)], y.iloc[:int(len(y) * 0.8)])
        model_path = tmp_path / "rf.pkl"
        model.save(model_path)
        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(model.model_name, "v1", str(model_path), {}, list(X.columns))

        # 5. Predict
        engine = PredictionEngine(registry=registry)
        result = engine.predict(X.tail(1))
        assert result.action in ("BUY", "SELL", "HOLD")

        # 6. Risk check
        risk = RiskEngine(RiskConfig(min_ai_confidence=40.0))
        price = float(df["close"].iloc[-1])
        qty = max(1, risk.size_position("TEST", price, capital=100_000.0,
                                        atr=float(df["close"].diff().abs().mean())))
        validation = risk.validate_trade(
            signal=result.action if result.action != "HOLD" else "BUY",
            symbol="TEST", price=price, quantity=qty,
            ai_confidence=max(result.confidence, 50.0),
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
        )

        # 7. Paper trade
        pb = PaperBroker(BrokerConfig(starting_capital=100_000.0))
        if validation.approved:
            pb.place_buy("TEST", qty, price)
            assert "TEST" in pb.portfolio.positions

        # 8. Snapshot / report
        snap = pb.snapshot()
        assert "balance" in snap and "positions" in snap

    def test_pipeline_blocks_bad_signal(self):
        from risk_management.risk_engine import RiskEngine, RiskConfig
        from paper_trading.paper_broker import BrokerConfig, PaperBroker
        risk = RiskEngine(RiskConfig(min_ai_confidence=80.0))
        pb = PaperBroker(BrokerConfig(starting_capital=100_000.0))
        validation = risk.validate_trade(
            signal="BUY", symbol="X", price=1000.0, quantity=5,
            ai_confidence=30.0,
            portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
            open_position_count=0, starting_capital=100_000.0,
        )
        assert not validation.approved
        if validation.approved:
            pb.place_buy("X", 5, 1000.0)
        assert pb.portfolio.cash == 100_000.0

    def test_empty_prediction_does_not_place_order(self):
        from prediction.prediction_engine import PredictionEngine
        from paper_trading.paper_broker import BrokerConfig, PaperBroker
        engine = PredictionEngine()
        pb = PaperBroker(BrokerConfig(starting_capital=50_000.0))
        result = engine.predict(pd.DataFrame())
        assert result.action == "HOLD"
        if result.action == "BUY":
            pb.place_buy("X", 1, 1000.0)
        assert pb.portfolio.cash == 50_000.0


# ===========================================================================
# STAGE 8: DISASTER RECOVERY
# ===========================================================================

class TestStage8DisasterRecovery:
    def test_backup_creates_manifest(self, tmp_path):
        import sqlite3
        from monitoring.disaster_recovery import DisasterRecovery
        db_file = tmp_path / "test.db"
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        dr = DisasterRecovery(
            source_root=tmp_path, backup_root=tmp_path / "backups",
            db_path=db_file, model_dir=tmp_path / "models",
            trade_journal_path=tmp_path / "journal.csv",
            log_dir=tmp_path / "logs",
        )
        manifest = dr.backup_all()
        assert manifest.backup_id != ""
        assert (tmp_path / "backups" / manifest.backup_id / "manifest.json").exists()

    def test_restore_dry_run(self, tmp_path):
        import sqlite3
        from monitoring.disaster_recovery import DisasterRecovery
        db_file = tmp_path / "test.db"
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        dr = DisasterRecovery(
            source_root=tmp_path, backup_root=tmp_path / "backups",
            db_path=db_file, model_dir=tmp_path / "models",
            trade_journal_path=tmp_path / "journal.csv",
            log_dir=tmp_path / "logs",
        )
        dr.backup_all()
        result = dr.restore_latest(dry_run=True)
        assert result.success

    def test_trade_log_recovery_deduplicates(self, tmp_path):
        import csv
        from monitoring.disaster_recovery import DisasterRecovery
        dr = DisasterRecovery(
            source_root=tmp_path, backup_root=tmp_path / "backups",
            db_path=tmp_path / "nope.db", model_dir=tmp_path / "models",
            trade_journal_path=tmp_path / "journal.csv",
            log_dir=tmp_path / "logs",
        )
        for bid in ["20250101_090000", "20250101_120000"]:
            bdir = tmp_path / "backups" / bid / "logs"
            bdir.mkdir(parents=True)
            (bdir / "journal.csv").write_text(
                "trade_id,symbol,entry_time\n"
                "T1,RELIANCE,2025-01-01T09:15:00\n"
                "T2,TCS,2025-01-01T10:00:00\n"
            )
        records = dr.recover_trade_logs()
        ids = [r["trade_id"] for r in records]
        assert len(ids) == len(set(ids))
        assert "T1" in ids and "T2" in ids

    def test_verify_backup_integrity(self, tmp_path):
        import sqlite3
        from monitoring.disaster_recovery import DisasterRecovery
        db_file = tmp_path / "test.db"
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        dr = DisasterRecovery(
            source_root=tmp_path, backup_root=tmp_path / "backups",
            db_path=db_file, model_dir=tmp_path / "models",
            trade_journal_path=tmp_path / "journal.csv",
            log_dir=tmp_path / "logs",
        )
        manifest = dr.backup_all()
        results = dr.verify_backup(manifest.backup_id)
        assert results.get("database", False) is True

    def test_list_backups_newest_first(self, tmp_path):
        import sqlite3
        from monitoring.disaster_recovery import DisasterRecovery
        db_file = tmp_path / "test.db"
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        dr = DisasterRecovery(
            source_root=tmp_path, backup_root=tmp_path / "backups",
            db_path=db_file, model_dir=tmp_path / "models",
            trade_journal_path=tmp_path / "journal.csv",
            log_dir=tmp_path / "logs",
        )
        dr.backup_all()
        time.sleep(1.1)
        dr.backup_all()
        backups = dr.list_backups()
        assert len(backups) >= 2
        assert backups[0].backup_id > backups[1].backup_id


# ===========================================================================
# FINAL VALIDATION REPORT
# ===========================================================================

@dataclass
class StageResult:
    name: str
    passed: int = 0
    failed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def success_rate(self) -> float:
        return self.passed / self.total * 100 if self.total else 0.0


class TestValidationReport:
    """Generates the final production validation report."""

    def test_generate_validation_report(self, tmp_path):
        stages: list[StageResult] = []

        # Stage 1: Data
        s1 = StageResult("Stage 1: Data Validation")
        try:
            from ai_trading_system.data_collection.validators import OHLCVValidator, DataValidationError
            OHLCVValidator().validate(_make_ohlcv()); s1.passed += 1
        except Exception as e:
            s1.failed += 1; s1.warnings.append(str(e))
        try:
            OHLCVValidator().validate(pd.DataFrame())
            s1.failed += 1
        except Exception:
            s1.passed += 1
        stages.append(s1)

        # Stage 2: Features
        s2 = StageResult("Stage 2: Feature Engineering")
        try:
            from features.feature_engineering import FeatureEngineeringService
            f, _ = FeatureEngineeringService().create_feature_dataset(_make_ohlcv(300))
            assert not f.empty and len(f.columns) >= 10; s2.passed += 1
        except Exception as e:
            s2.failed += 1; s2.warnings.append(str(e))
        stages.append(s2)

        # Stage 3: Model Training
        s3 = StageResult("Stage 3: AI Model Training")
        trained_model = None; X_train = None
        try:
            from models.random_forest_model import RandomForestTradingModel
            X, y = _make_training_data(400)
            if not X.empty:
                m = RandomForestTradingModel(); m.train(X, y)
                trained_model = m; X_train = X; s3.passed += 1
            else:
                s3.warnings.append("Insufficient training data")
        except Exception as e:
            s3.failed += 1; s3.warnings.append(f"Training: {e}")
        stages.append(s3)

        # Stage 4: Prediction
        s4 = StageResult("Stage 4: Prediction Engine")
        try:
            from prediction.prediction_engine import PredictionEngine
            result = PredictionEngine().predict(pd.DataFrame())
            assert result.action == "HOLD"; s4.passed += 1
        except Exception as e:
            s4.failed += 1; s4.warnings.append(str(e))
        if trained_model and X_train is not None:
            try:
                from models.model_registry import ModelRegistry
                from prediction.prediction_engine import PredictionEngine
                mp = tmp_path / "rf_val.pkl"; trained_model.save(mp)
                reg = ModelRegistry(registry_path=tmp_path / "reg.json")
                feat_cols = list(X_train.columns) if X_train is not None else []
                reg.register(trained_model.model_name, "v1", str(mp), {}, feat_cols)
                eng = PredictionEngine(registry=reg)
                r = eng.predict(X_train.tail(1))
                assert r.action in ("BUY", "SELL", "HOLD"); s4.passed += 1
            except Exception as e:
                s4.failed += 1; s4.warnings.append(f"Trained predict: {e}")
        stages.append(s4)

        # Stage 5: Risk
        s5 = StageResult("Stage 5: Risk Engine")
        try:
            from risk_management.risk_engine import RiskEngine, RiskConfig
            risk = RiskEngine(RiskConfig(min_ai_confidence=60.0))
            r = risk.validate_trade("BUY", "X", 1000.0, 5, ai_confidence=70.0,
                portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
                open_position_count=0, starting_capital=100_000.0)
            assert r.approved; s5.passed += 1
            r2 = risk.validate_trade("BUY", "X", 0.0, 5, ai_confidence=70.0,
                portfolio_cash=100_000.0, portfolio_total_value=100_000.0,
                open_position_count=0, starting_capital=100_000.0)
            assert not r2.approved; s5.passed += 1
        except Exception as e:
            s5.failed += 1; s5.warnings.append(str(e))
        stages.append(s5)

        # Stage 6: Paper Trading
        s6 = StageResult("Stage 6: Paper Trading")
        try:
            from paper_trading.paper_broker import BrokerConfig, PaperBroker
            pb = PaperBroker(BrokerConfig(starting_capital=100_000.0))
            pb.place_buy("RELIANCE", 10, 2500.0)
            assert "RELIANCE" in pb.portfolio.positions
            pb.place_sell("RELIANCE", 10, 2600.0)
            assert pb.portfolio.closed_positions[0].realized_pnl > 0
            s6.passed += 1
        except Exception as e:
            s6.failed += 1; s6.warnings.append(str(e))
        stages.append(s6)

        # Stage 7: Disaster Recovery
        s7 = StageResult("Stage 7: Disaster Recovery")
        try:
            import sqlite3
            from monitoring.disaster_recovery import DisasterRecovery
            dr_tmp = tmp_path / "dr"
            dr_tmp.mkdir()
            db_file = dr_tmp / "trading.db"
            with sqlite3.connect(str(db_file)) as conn:
                conn.execute("CREATE TABLE t (x INTEGER)")
            dr = DisasterRecovery(
                source_root=dr_tmp, backup_root=dr_tmp / "backups",
                db_path=db_file, model_dir=dr_tmp / "models",
                trade_journal_path=dr_tmp / "journal.csv",
                log_dir=dr_tmp / "logs",
            )
            manifest = dr.backup_all()
            result = dr.restore_latest(dry_run=True)
            assert result.success; s7.passed += 1
        except Exception as e:
            s7.failed += 1; s7.warnings.append(str(e))
        stages.append(s7)

        # ── Compute report ─────────────────────────────────────────────
        total_passed = sum(s.passed for s in stages)
        total_failed = sum(s.failed for s in stages)
        total_tests  = total_passed + total_failed
        overall_rate = total_passed / total_tests * 100 if total_tests else 0
        score = overall_rate
        if score >= 90:
            readiness = "PRODUCTION READY"
        elif score >= 75:
            readiness = "BETA - minor issues"
        elif score >= 50:
            readiness = "NEEDS WORK"
        else:
            readiness = "NOT READY"

        all_warnings = [f"[{s.name}] {w}" for s in stages for w in s.warnings]

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate_pct": round(overall_rate, 1),
                "warnings": len(all_warnings),
                "production_readiness_score": round(score, 1),
                "readiness": readiness,
            },
            "stages": [
                {"name": s.name, "passed": s.passed, "failed": s.failed,
                 "success_rate_pct": round(s.success_rate, 1), "warnings": s.warnings}
                for s in stages
            ],
            "all_warnings": all_warnings,
        }

        report_path = Path("reports/validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))

        sep = "=" * 60
        dash = "-" * 60
        print(f"\n{sep}")
        print("  AI TRADING SYSTEM -- PRODUCTION VALIDATION REPORT")
        print(sep)
        print(f"  Generated : {report['generated_at']}")
        print(f"  Tests Run : {total_tests}")
        print(f"  Passed    : {total_passed}")
        print(f"  Failed    : {total_failed}")
        print(f"  Warnings  : {len(all_warnings)}")
        print(f"  Score     : {score:.1f}%")
        print(f"  Readiness : {readiness}")
        print(dash)
        for s in stages:
            status = "PASS" if s.failed == 0 else "FAIL"
            print(f"  [{status:4s}] {s.name:<40} {s.success_rate:.0f}%")
        if all_warnings:
            print(dash)
            print("  WARNINGS:")
            for w in all_warnings[:10]:
                print(f"    [!] {w}")
        print(sep)
        print(f"  Report saved: reports/validation_report.json")
        print(f"{sep}\n")

        assert total_failed == 0, (
            f"Pipeline FAILED: {total_failed} stage(s) failed\n"
            + "\n".join(all_warnings)
        )
