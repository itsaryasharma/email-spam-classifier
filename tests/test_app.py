"""Headless tests of the Streamlit app using streamlit.testing.AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_empty_state():
    at = _run()
    assert len(at.tabs) == 4
    assert any("Paste a message" in i.value for i in at.info)


def test_examples_give_expected_verdicts():
    at = _run()
    at.button(key="ex_Prize offer").click().run()
    assert not at.exception
    assert "Likely spam" in _text(at)
    at.button(key="ex_Project update").click().run()
    assert "Likely legitimate" in _text(at)


def test_short_message_warning_and_threshold():
    at = _run()
    at.text_area(key="message").input("hello there").run()
    assert not at.exception
    assert any("fewer than" in w.value for w in at.warning)
    at.slider(key="threshold").set_value(0.95).run()
    assert not at.exception
    assert "threshold 95%" in _text(at)
