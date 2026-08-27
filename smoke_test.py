"""Render the Streamlit script through Streamlit's official AppTest harness."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.title or app.markdown, "The application rendered no visible content."
    assert len(app.sidebar.radio) == 1, "Navigation radio was not rendered."
    assert len(app.metric) >= 4, "Overview metrics were not rendered."

    pages = [
        "Overview",
        "Individual Prediction",
        "Batch Prediction",
        "Model Evaluation",
        "Feature Analysis",
        "Dataset Explorer",
        "Methodology & About",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=180)
        assert not app.exception, f"{page}: {[str(item.value) for item in app.exception]}"

    app.sidebar.radio[0].set_value("Individual Prediction")
    app.run(timeout=180)
    app.button[0].click()
    app.run(timeout=180)
    assert not app.exception, [str(item.value) for item in app.exception]
    assert any("Final prediction:" in item.value for item in app.success)
    print("STREAMLIT APPTEST PASSED: 7 pages and individual form submission")


if __name__ == "__main__":
    main()
