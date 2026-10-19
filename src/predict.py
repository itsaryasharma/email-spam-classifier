"""Prediction helpers used by the Streamlit app and the tests (no Streamlit code)."""
import json
import re

import joblib

from src.config import DEFAULT_THRESHOLD, MAX_CHARS, METRICS_PATH, MIN_WORDS_RELIABLE, MODEL_PATH
from src.explain import explain
from src.text import EMAIL_RE, URL_RE


def ensure_trained() -> None:
    """Train the model if the saved artifacts are missing (downloads the data first)."""
    if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
        from src.train import main as train_main
        train_main()


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_text(text) -> tuple[str, list[str]]:
    """Validate and clean user input.

    Returns (clean_text, warnings). Raises ValueError for unusable input.
    """
    if not isinstance(text, str):
        raise ValueError("The message must be text.")
    text = text.strip()
    if not text:
        raise ValueError("Please paste or type a message to analyse.")
    warnings = []
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append(f"The message is very long; only the first {MAX_CHARS:,} characters were analysed.")
    if len(text.split()) < MIN_WORDS_RELIABLE:
        warnings.append(
            f"The message has fewer than {MIN_WORDS_RELIABLE} words. With so little text the "
            "prediction is unreliable."
        )
    return text, warnings


def classify(model, text, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Classify one message and explain the decision."""
    if not 0 < threshold < 1:
        raise ValueError("The threshold must be between 0 and 1.")
    clean, warnings = prepare_text(text)
    proba = float(model.predict_proba([clean])[0, 1])
    exp = explain(model, clean)
    if exp["n_known_terms"] == 0:
        warnings.append(
            "None of the words in this message appear in the model's vocabulary, so the result "
            "reflects only the overall spam rate in the training data."
        )
    return {
        "text": clean,
        "spam_probability": proba,
        "is_spam": proba >= threshold,
        "threshold": threshold,
        "warnings": warnings,
        "explanation": exp,
    }


_SEGMENT_RE = re.compile(
    rf"(?P<url>{URL_RE.pattern})|(?P<email>{EMAIL_RE.pattern})|(?P<word>[A-Za-z]+)|(?P<other>[^A-Za-z]+)",
    re.IGNORECASE,
)


def highlight_segments(text: str, contributions: dict) -> list[tuple[str, float | None]]:
    """Split text into (segment, weight) pairs for colour-highlighting.

    Words get the contribution of the matching single-word feature; links and
    e-mail addresses get the contribution of their placeholder token. Segments
    the model ignores (stop words, punctuation, unknown words) get None.
    """
    out = []
    for m in _SEGMENT_RE.finditer(text):
        seg = m.group(0)
        if m.lastgroup == "url":
            out.append((seg, contributions.get("urltoken")))
        elif m.lastgroup == "email":
            out.append((seg, contributions.get("emailtoken")))
        elif m.lastgroup == "word":
            out.append((seg, contributions.get(seg.lower())))
        else:
            out.append((seg, None))
    return out


def metrics_at_threshold(y_true: list[int], y_proba: list[float], threshold: float) -> dict:
    """Recompute test-set confusion counts and metrics for any threshold."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_proba):
        pred = p >= threshold
        if pred and t:
            tp += 1
        elif pred and not t:
            fp += 1
        elif not pred and t:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": (tp + tn) / len(y_true)}
