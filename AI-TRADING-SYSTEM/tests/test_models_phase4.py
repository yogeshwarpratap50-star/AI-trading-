import pandas as pd
from sklearn.model_selection import train_test_split

from models.random_forest_model import RandomForestTradingModel
from models.xgboost_model import XGBoostTradingModel
from training.dataset_builder import MODEL_FEATURES, DatasetBuilder
from tests.test_training_dataset import sample_ohlcv


def training_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = DatasetBuilder().build(sample_ohlcv(90))
    x = dataset[MODEL_FEATURES]
    y = dataset["target_next_day_up"].astype(int)
    return x, y


def assert_model_trains_saves_and_loads(model, tmp_path) -> None:
    x, y = training_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.train(x_train, y_train)
    metrics = model.evaluate(x_test, y_test, include_roc_auc=model.model_name == "xgboost")
    path = tmp_path / f"{model.model_name}.joblib"
    model.save(path)
    loaded = model.__class__()
    loaded.load(path)

    assert path.exists()
    assert metrics["accuracy"] >= 0
    assert len(loaded.predict(x_test.head(2))) == 2


def test_random_forest_model_train_save_load(tmp_path) -> None:
    assert_model_trains_saves_and_loads(RandomForestTradingModel(n_estimators=10), tmp_path)


def test_xgboost_model_train_save_load(tmp_path) -> None:
    assert_model_trains_saves_and_loads(XGBoostTradingModel(n_estimators=10), tmp_path)
