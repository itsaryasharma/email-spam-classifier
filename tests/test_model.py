"""Tests for text processing, parsing, prediction, explanation and saved metrics."""
import math
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pytest
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_PATH, RANDOM_STATE, TEST_SIZE
from src.data import parse_email, parse_enron_file
from src.predict import (
    classify,
    ensure_trained,
    highlight_segments,
    load_metrics,
    load_model,
    metrics_at_threshold,
    prepare_text,
)
from src.text import html_to_text, normalize

SPAM = ("Subject: Congratulations! You have been selected\n\nDear friend, you have been selected to receive a "
        "FREE $500 gift card. Click here now to claim your prize: http://example.com/claim")
HAM = ("Subject: Re: notes from Tuesday's project meeting\n\nHi Priya, thanks for sending the notes. I have "
       "updated the test plan and pushed the changes to the repository. Can we review it on Thursday?")


@pytest.fixture(scope="module")
def model():
    ensure_trained()
    return load_model()


@pytest.fixture(scope="module")
def metrics():
    ensure_trained()
    return load_metrics()


# ---- text processing -------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Visit https://example.com/x?a=1 now", "urltoken"),
    ("go to www.example.org today", "urltoken"),
    ("http : / / www . site . com / page . php", "urltoken"),       # Enron-style spaced URL
    ("mail me at a.b@example.com", "emailtoken"),
    ("write to kaminski @ enron . com", "emailtoken"),              # Enron-style spaced address
    ("only $99", "moneytoken"),
    ("call 555 1234", "numtoken"),
])
def test_normalize_placeholders(raw, expected):
    assert expected in normalize(raw)


def test_normalize_leaves_plain_text_alone():
    assert normalize("Normal sentence. Next one / here").split() == ["normal", "sentence.", "next", "one", "/", "here"]


def test_html_to_text():
    out = html_to_text("<html><style>p{}</style><p>Hello&nbsp;<b>World</b></p><br>Bye</html>")
    assert "Hello" in out and "World" in out and "Bye" in out and "<" not in out and "p{}" not in out


def test_parse_mime_prefers_plain_text():
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Test subject"
    msg.attach(MIMEText("plain body", "plain"))
    msg.attach(MIMEText("<p>html body</p>", "html"))
    subject, body = parse_email(msg.as_bytes())
    assert subject == "Test subject" and body == "plain body"


def test_parse_mime_html_only():
    msg = MIMEText("<p>Only <i>html</i></p>", "html")
    msg["Subject"] = "S"
    assert parse_email(msg.as_bytes())[1] == "Only html"


def test_parse_enron_file():
    assert parse_enron_file(b"Subject: meeting\nsee you at 3\n") == ("meeting", "see you at 3")


# ---- validation and prediction ---------------------------------------------
@pytest.mark.parametrize("bad", ["", "   \n ", None, 42])
def test_invalid_input_rejected(model, bad):
    with pytest.raises(ValueError):
        classify(model, bad)


def test_short_and_long_input_warnings():
    assert prepare_text("hi there")[1]                       # fewer than 5 words
    text, warns = prepare_text("word " * 20_000)
    assert len(text) == 50_000 and any("first" in w for w in warns)


def test_invalid_threshold(model):
    with pytest.raises(ValueError):
        classify(model, HAM, threshold=1.5)


def test_obvious_examples(model):
    assert classify(model, SPAM)["spam_probability"] > 0.9
    assert classify(model, HAM)["spam_probability"] < 0.1


def test_explanation_is_exact(model):
    """Base + sum of contributions must reproduce the model's own probability."""
    for text in (SPAM, HAM):
        r = classify(model, text)
        log_odds = r["explanation"]["log_odds"]
        assert math.isclose(1 / (1 + math.exp(-log_odds)), r["spam_probability"], rel_tol=1e-6)
        assert math.isclose(r["explanation"]["base"] + sum(r["explanation"]["contributions"].values()), log_odds)


def test_unknown_words_warning(model):
    r = classify(model, "zzqx vbnmq plokij qwertyuiop asdfghjklz")
    assert r["explanation"]["n_known_terms"] == 0
    assert any("vocabulary" in w for w in r["warnings"])


def test_highlight_segments_cover_whole_text():
    text = "Click http://x.com now!"
    segs = highlight_segments(text, {"click": 1.0, "urltoken": 0.5})
    assert "".join(s for s, _ in segs) == text
    assert dict(segs)["Click"] == 1.0 and dict(segs)["http://x.com"] == 0.5


# ---- saved metrics ----------------------------------------------------------
def test_stored_predictions_reproduce_reported_metrics(metrics):
    t = metrics["test"]
    r = metrics_at_threshold(t["y_true"], t["y_proba"], metrics["threshold"])
    assert [[r["tn"], r["fp"]], [r["fn"], r["tp"]]] == t["confusion_matrix"]
    assert math.isclose(r["precision"], t["precision"], abs_tol=1e-9)
    assert math.isclose(r["recall"], t["recall"], abs_tol=1e-9)
    assert t["f1"] > 0.95 and t["roc_auc"] > 0.99


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="dataset not downloaded (run python -m src.data)")
def test_saved_model_reproduces_stored_test_probabilities(model, metrics):
    import pandas as pd
    df = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    _, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["corpus"] + "/" + df["label"].astype(str),
                                  random_state=RANDOM_STATE)
    proba = model.predict_proba(test_df["text"])[:, 1]
    assert list(test_df["label"]) == metrics["test"]["y_true"]
    assert all(abs(a - b) < 1e-4 for a, b in zip(proba, metrics["test"]["y_proba"]))


def test_selected_model_is_consistent_everywhere(model, metrics):
    """Selection rule result == stored choice == class of the loaded artifact (spam F1 audit)."""
    from src.train import select_family, selection_summary
    results = {k: {"cv_f1": v["cv_f1"]} for k, v in metrics["model_search"].items()}
    expected = {"Multinomial Naive Bayes": "MultinomialNB", "Logistic Regression": "LogisticRegression"}
    assert select_family(results) == metrics["selected_model"]
    assert type(model.named_steps["clf"]).__name__ == expected[metrics["selected_model"]]
    outcome = selection_summary(results)
    assert outcome == metrics["selection_outcome"]
    assert outcome["selected"] == metrics["selected_model"]
    assert outcome["tie_break_applied"] == (outcome["gap"] <= outcome["tolerance"] and outcome["best_by_cv"] != outcome["simplest"])
    for name, value in metrics["selected_params"].items():
        assert getattr(model.named_steps["clf"], name) == value


def test_selection_summary_handles_a_near_tie():
    from src.train import selection_summary
    s = selection_summary({"Multinomial Naive Bayes": {"cv_f1": 0.980}, "Logistic Regression": {"cv_f1": 0.983}})
    assert s["tie_break_applied"] and s["selected"] == "Multinomial Naive Bayes"
