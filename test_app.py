"""App smoke test (skipped until a trained model is in model/)."""

from pathlib import Path

import pytest

from captioner import model_files_exist

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_without_model_or_with_model():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert "Caption Generator" in at.title[0].value
    if not model_files_exist():
        assert at.error  # shows the "model files not found" message
    else:
        pytest.skip("model present: upload flow needs a browser; covered by test_core")
