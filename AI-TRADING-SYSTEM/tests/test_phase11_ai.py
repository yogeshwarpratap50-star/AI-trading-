"""Tests for Phase 11 — Advanced AI Optimization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from advanced_ai.feature_selector import FeatureSelector
from advanced_ai.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationResult
from advanced_ai.market_regime import MarketRegimeDetector, Regime, RegimeResult
from advanced_ai.adaptive_strategy import AdaptiveStrategy
from advanced_ai.performance_tracker import AIPerformanceTracker, PerformanceReport
from advanced_ai.drift_detector import DriftDetector, DriftReport
from advanced_ai.auto_retrainer import AutoRetrainer, RetrainingConfig, RetrainingTrigger
from advanced_ai.ensemble_model import EnsembleModel, EnsembleConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Simple trending OHLCV for regime / drift tests."""
    n = 100
    np.random.seed(42)
    close = 2500 + np.cumsum(np.random.randn(n) * 10 + 0.5)
    return pd.DataFrame({
        "open": close - 5,
        "high": close + 20,
        "low": close - 20,
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n),
    })


@pytest.fixture
def feature_df() -> pd.DataFrame:
    np.random.seed(7)
    n = 200
    return pd.DataFrame({
        f"feat_{i}": np.random.randn(n) for i in range(10)
    })


@pytest.fixture
def classification_data(feature_df):
    y = pd.Series((feature_df["feat_0"] > 0).astype(int), name="label")
    return feature_df, y


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

class TestFeatureSelector:
    def test_fit_from_df(self) -> None:
        imp = pd.DataFrame({"feature": [f"f{i}" for i in range(10)], "importance": np.linspace(0.1, 1.0, 10)[::-1]})
        sel = FeatureSelector(top_n=5)
        sel.fit_from_df(imp)
        assert len(sel.selected_features) == 5

    def test_transform_drops_unwanted(self, feature_df) -> None:
        imp = pd.DataFrame({"feature": feature_df.columns[:3].tolist(), "importance": [0.5, 0.3, 0.2]})
        sel = FeatureSelector(top_n=3)
        sel.fit_from_df(imp)
        result = sel.transform(feature_df)
        assert result.shape[1] == 3

    def test_fit_from_sklearn_model(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        sel = FeatureSelector(top_n=5)
        sel.fit(model, X.columns.tolist())
        assert len(sel.selected_features) <= 5

    def test_importance_report_df(self) -> None:
        imp = pd.DataFrame({"feature": ["a", "b"], "importance": [0.6, 0.4]})
        sel = FeatureSelector(top_n=2)
        sel.fit_from_df(imp)
        report = sel.importance_report()
        assert "feature" in report.columns
        assert "importance" in report.columns

    def test_min_importance_filter(self) -> None:
        imp = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.5, 0.0005, 0.3]})
        sel = FeatureSelector(top_n=10, min_importance=0.001)
        sel.fit_from_df(imp)
        assert "b" not in sel.selected_features


# ---------------------------------------------------------------------------
# HyperparameterOptimizer
# ---------------------------------------------------------------------------

class TestHyperparameterOptimizer:
    def test_random_search_returns_result(self, classification_data) -> None:
        X, y = classification_data
        opt = HyperparameterOptimizer(
            estimator_fn=lambda n_estimators, max_depth: RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            ),
            param_space={"n_estimators": [10, 20], "max_depth": [3, 5]},
            method="random",
        )
        result = opt.optimize(X, y, n_trials=5)
        assert isinstance(result, OptimizationResult)
        assert result.best_score > 0
        assert "n_estimators" in result.best_params

    def test_grid_search(self, classification_data) -> None:
        X, y = classification_data
        opt = HyperparameterOptimizer(
            estimator_fn=lambda n_estimators: RandomForestClassifier(n_estimators=n_estimators, random_state=42),
            param_space={"n_estimators": [5, 10]},
            method="grid",
        )
        result = opt.optimize(X, y)
        assert result.method == "grid"
        assert len(result.all_results) == 2

    def test_optuna_falls_back_to_random(self, classification_data, monkeypatch) -> None:
        monkeypatch.setattr("advanced_ai.hyperparameter_optimizer._try_optuna", lambda: None)
        X, y = classification_data
        opt = HyperparameterOptimizer(
            estimator_fn=lambda n_estimators: RandomForestClassifier(n_estimators=n_estimators, random_state=42),
            param_space={"n_estimators": [5, 10]},
            method="optuna",
        )
        result = opt.optimize(X, y, n_trials=3)
        assert result.method == "random"


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------

class TestMarketRegimeDetector:
    def test_detect_returns_regime_result(self, ohlcv_df) -> None:
        result = MarketRegimeDetector().detect(ohlcv_df)
        assert isinstance(result, RegimeResult)

    def test_regime_is_valid(self, ohlcv_df) -> None:
        result = MarketRegimeDetector().detect(ohlcv_df)
        assert result.regime in list(Regime)

    def test_confidence_between_0_and_1(self, ohlcv_df) -> None:
        result = MarketRegimeDetector().detect(ohlcv_df)
        assert 0.0 <= result.confidence <= 1.0

    def test_volatility_positive(self, ohlcv_df) -> None:
        result = MarketRegimeDetector().detect(ohlcv_df)
        assert result.volatility_pct >= 0.0

    def test_to_dict(self, ohlcv_df) -> None:
        result = MarketRegimeDetector().detect(ohlcv_df)
        d = result.to_dict()
        assert "regime" in d
        assert "confidence" in d

    def test_bull_regime_on_uptrend(self) -> None:
        n = 120
        close = 1000 + np.arange(n) * 5.0   # strong uptrend
        df = pd.DataFrame({
            "open": close - 2, "high": close + 5, "low": close - 5, "close": close,
            "volume": np.ones(n) * 1_000_000,
        })
        result = MarketRegimeDetector().detect(df)
        assert result.trend_strength > 0


# ---------------------------------------------------------------------------
# AdaptiveStrategy
# ---------------------------------------------------------------------------

class TestAdaptiveStrategy:
    def test_adjust_bull_increases_size(self) -> None:
        from risk_management.risk_engine import RiskConfig
        base = RiskConfig(risk_per_trade_pct=1.0)
        adj = AdaptiveStrategy(base).adjust(Regime.BULL)
        assert adj.risk_per_trade_pct > base.risk_per_trade_pct

    def test_adjust_bear_reduces_size(self) -> None:
        from risk_management.risk_engine import RiskConfig
        base = RiskConfig(risk_per_trade_pct=1.0)
        adj = AdaptiveStrategy(base).adjust(Regime.BEAR)
        assert adj.risk_per_trade_pct < base.risk_per_trade_pct

    def test_adjust_high_vol_smallest_size(self) -> None:
        from risk_management.risk_engine import RiskConfig
        base = RiskConfig(risk_per_trade_pct=1.0)
        adj = AdaptiveStrategy(base).adjust(Regime.HIGH_VOLATILITY)
        assert adj.risk_per_trade_pct <= 0.5

    def test_all_adjustments_has_five_entries(self) -> None:
        data = AdaptiveStrategy.all_adjustments()
        assert len(data) == 5


# ---------------------------------------------------------------------------
# AIPerformanceTracker
# ---------------------------------------------------------------------------

class TestAIPerformanceTracker:
    def test_log_and_resolve(self) -> None:
        tracker = AIPerformanceTracker()
        tracker.log_prediction("RF", "RELIANCE", "BUY", 75.0)
        tracker.log_outcome("RELIANCE", "BUY", pnl=500.0)
        report = tracker.report()
        assert report.total_predictions == 1
        assert report.correct_predictions == 1

    def test_accuracy_correct_predictions(self) -> None:
        tracker = AIPerformanceTracker()
        tracker.log_prediction("RF", "TCS", "BUY", 70.0)
        tracker.log_outcome("TCS", "SELL", pnl=-100.0)   # wrong
        report = tracker.report()
        assert report.accuracy == 0.0

    def test_empty_report(self) -> None:
        tracker = AIPerformanceTracker()
        report = tracker.report()
        assert report.total_predictions == 0
        assert report.accuracy == 0.0

    def test_as_dataframe(self) -> None:
        tracker = AIPerformanceTracker()
        tracker.log_prediction("XGB", "INFY", "SELL", 68.0)
        df = tracker.as_dataframe()
        assert not df.empty


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def test_detect_no_drift(self, feature_df) -> None:
        det = DriftDetector()
        det.fit_reference(feature_df.iloc[:100])
        report = det.detect(feature_df.iloc[:100])  # same data → near-zero PSI
        assert report.data_drift < 0.5

    def test_detect_large_drift(self, feature_df) -> None:
        det = DriftDetector()
        det.fit_reference(feature_df.iloc[:100])
        # Shift data by large constant → high PSI
        shifted = feature_df.iloc[100:].copy() + 10
        report = det.detect(shifted)
        assert report.data_drift > 0

    def test_report_has_feature_psi(self, feature_df) -> None:
        det = DriftDetector()
        det.fit_reference(feature_df.iloc[:100])
        report = det.detect(feature_df.iloc[100:])
        assert isinstance(report.feature_drift, dict)

    def test_to_dict(self, feature_df) -> None:
        det = DriftDetector()
        det.fit_reference(feature_df.iloc[:100])
        report = det.detect(feature_df.iloc[100:])
        d = report.to_dict()
        assert "data_drift" in d
        assert "has_critical_drift" in d

    def test_requires_fit_reference(self, feature_df) -> None:
        det = DriftDetector()
        with pytest.raises(RuntimeError):
            det.detect(feature_df)


# ---------------------------------------------------------------------------
# AutoRetrainer
# ---------------------------------------------------------------------------

class TestAutoRetrainer:
    def test_no_triggers_when_healthy(self) -> None:
        from advanced_ai.performance_tracker import PerformanceReport
        from datetime import datetime
        retrainer = AutoRetrainer(RetrainingConfig(weekly_retrain=False, monthly_retrain=False))
        perf = PerformanceReport(100, 60, 0.6, 0.6, 100.0, 10000.0, {}, 30)
        triggers = retrainer.evaluate(perf_report=perf, last_retrain=datetime.now())
        assert len(triggers) == 0

    def test_low_accuracy_trigger(self) -> None:
        from advanced_ai.performance_tracker import PerformanceReport
        from datetime import datetime
        retrainer = AutoRetrainer(RetrainingConfig(min_accuracy=0.6, weekly_retrain=False, monthly_retrain=False))
        perf = PerformanceReport(50, 20, 0.4, 0.4, -50.0, -2500.0, {}, 30)
        triggers = retrainer.evaluate(perf_report=perf, last_retrain=datetime.now())
        assert any(t.reason == "low_accuracy" for t in triggers)

    def test_drift_trigger(self) -> None:
        from datetime import datetime
        from advanced_ai.drift_detector import DriftReport
        retrainer = AutoRetrainer(RetrainingConfig(retrain_on_drift=True, weekly_retrain=False, monthly_retrain=False))
        drift = DriftReport(
            feature_drift={"feat_0": 0.3},
            data_drift=0.3,
            prediction_drift=0.0,
            alerts=["CRITICAL drift in feat_0"],
        )
        triggers = retrainer.evaluate(drift_report=drift, last_retrain=datetime.now())
        assert any(t.reason == "drift_detected" for t in triggers)

    def test_weekly_schedule_trigger(self) -> None:
        from datetime import datetime, timedelta
        retrainer = AutoRetrainer(RetrainingConfig(weekly_retrain=True, monthly_retrain=False))
        last = datetime.now() - timedelta(days=8)
        triggers = retrainer.evaluate(last_retrain=last)
        assert any(t.reason == "weekly_schedule" for t in triggers)

    def test_retrain_skips_insufficient_data(self) -> None:
        from datetime import datetime
        retrainer = AutoRetrainer(RetrainingConfig(min_samples=500))
        trigger = RetrainingTrigger("test", datetime.now().isoformat(), "accuracy", 0.4, 0.6)
        small_df = pd.DataFrame({"a": range(10), "label": [0] * 10})
        result = retrainer.retrain(small_df, trigger)
        assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    def test_predict_without_loaded_models_returns_hold(self) -> None:
        ens = EnsembleModel()
        import pandas as pd
        result = ens.predict(pd.DataFrame({"a": [1.0]}))
        assert result.action == "HOLD"
        assert result.confidence == 0.0

    def test_feature_importances_empty_without_models(self) -> None:
        ens = EnsembleModel()
        imp = ens.feature_importances()
        assert imp.empty or list(imp.columns) == ["feature", "importance"]

    def test_fit_and_predict(self, classification_data) -> None:
        X, y = classification_data
        ens = EnsembleModel(EnsembleConfig(rf_weight=0.5, xgb_weight=0.5, min_confidence=0.0))
        ens.fit(X.iloc[:150], y.iloc[:150])
        result = ens.predict(X.iloc[150:])
        assert result.action in ("BUY", "SELL", "HOLD")
        assert 0 <= result.confidence <= 100
