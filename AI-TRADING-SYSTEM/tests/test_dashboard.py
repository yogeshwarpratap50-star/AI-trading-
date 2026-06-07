import importlib


def test_dashboard_module_imports() -> None:
    module = importlib.import_module("dashboard.streamlit_app")

    assert hasattr(module, "main")
    assert hasattr(module, "overview_page")
