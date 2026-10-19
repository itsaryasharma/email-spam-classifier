"""Train and evaluate the spam classifier.

    python -m src.train

Pipeline: normalise text -> TF-IDF (unigrams + bigrams) -> classifier.

Two classifier families are tuned with GridSearchCV on the TRAINING split
(5-fold stratified CV, F1 on the spam class):
  * Multinomial Naive Bayes (the classic text-classification baseline)
  * Logistic Regression
The family with the best CV F1 is chosen; if they are within 0.005 the
simpler Naive Bayes model is preferred. The chosen model is then evaluated
exactly once on the held-out test split.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    CORPUS_IDENTITY_TERMS,
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    N_JOBS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import build_dataset
from src.explain import global_top_terms
from src.text import TOKEN_PATTERN, normalize

SELECTION_TOLERANCE = 0.005


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize,       # lower-case + placeholder tokens (src/text.py)
        token_pattern=TOKEN_PATTERN,  # alphabetic tokens of length >= 2
        # Drop common words ("the", "and", ...) and corpus-identity words (see config).
        stop_words=sorted(ENGLISH_STOP_WORDS.union(CORPUS_IDENTITY_TERMS)),
        ngram_range=(1, 2),           # single words and two-word phrases
        min_df=2,                     # ignore terms seen in only one training e-mail
        max_df=0.95,                  # ignore terms in > 95% of e-mails
        max_features=30_000,          # cap vocabulary size (keeps the model file small)
        sublinear_tf=True,            # 1 + log(tf): 10 repetitions != 10x the evidence
    )


def build_searches(cv) -> dict:
    """Candidate model families, simplest first, each with a small grid."""
    return {
        "Multinomial Naive Bayes": GridSearchCV(
            Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
            scoring="f1", cv=cv, n_jobs=N_JOBS,
        ),
        "Logistic Regression": GridSearchCV(
            Pipeline([("tfidf", make_vectorizer()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"clf__C": [1.0, 10.0, 100.0, 1000.0]},
            scoring="f1", cv=cv, n_jobs=N_JOBS,
        ),
    }


def select_family(results: dict) -> str:
    best = max(r["cv_f1"] for r in results.values())
    for name, r in results.items():
        if r["cv_f1"] >= best - SELECTION_TOLERANCE:
            return name
    raise RuntimeError("unreachable")


def selection_summary(results: dict) -> dict:
    """Describe the outcome of the selection rule, not just the rule itself.

    Records which family had the best CV F1, the gap to the simpler family,
    and whether the "prefer the simpler model" tie-break was actually applied.
    """
    names = list(results)                       # insertion order = simplest first
    simplest = names[0]
    best = max(names, key=lambda n: results[n]["cv_f1"])
    chosen = select_family(results)
    gap = results[best]["cv_f1"] - results[simplest]["cv_f1"]
    tie_break_applied = chosen != best
    if best == simplest:
        text = f"{simplest} had the best CV F1 ({results[best]['cv_f1']:.4f}), so it was selected directly."
    elif tie_break_applied:
        text = (f"{best} had the best CV F1 ({results[best]['cv_f1']:.4f}), but {simplest} was within "
                f"{SELECTION_TOLERANCE} ({gap:.4f} lower), so the tie-break selected the simpler {simplest}.")
    else:
        text = (f"{best} had the best CV F1 ({results[best]['cv_f1']:.4f}). {simplest} scored "
                f"{results[simplest]['cv_f1']:.4f}, {gap:.4f} lower: more than the {SELECTION_TOLERANCE} "
                f"tolerance, so the tie-break in favour of {simplest} was not triggered and {best} was selected.")
    return {"tolerance": SELECTION_TOLERANCE, "best_by_cv": best, "simplest": simplest,
            "gap": float(gap), "tie_break_applied": tie_break_applied, "selected": chosen, "text": text}


def _downsample_curve(x, y, max_points=200) -> dict:
    """Keep at most `max_points` evenly spaced points of a curve (for plotting)."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(max_points, len(x))).astype(int))
    return {"x": [float(v) for v in np.asarray(x)[idx]], "y": [float(v) for v in np.asarray(y)[idx]]}


def train_and_evaluate():
    df, cleaning = build_dataset()
    # Stratify on corpus AND label so both corpora keep their spam ratio in train and test.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["corpus"] + "/" + df["label"].astype(str),
        random_state=RANDOM_STATE,
    )
    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    searches = build_searches(cv)
    results = {}
    for name, search in searches.items():
        print(f"Tuning {name} ...")
        search.fit(X_train, y_train)
        i = search.best_index_
        results[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][i]),
            "grid": [
                {"params": {k.replace("clf__", ""): v for k, v in p.items()},
                 "mean_f1": float(m), "std_f1": float(s)}
                for p, m, s in zip(search.cv_results_["params"],
                                   search.cv_results_["mean_test_score"],
                                   search.cv_results_["std_test_score"])
            ],
        }

    chosen = select_family(results)
    model = searches[chosen].best_estimator_  # refit on the full training split by GridSearchCV

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= DEFAULT_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    # Per-corpus results reveal whether the model only works on one source.
    by_corpus = {}
    for corpus, idx in test_df.groupby("corpus").groups.items():
        mask = test_df.index.isin(idx)
        yt, yp, pr = y_test[mask], y_pred[mask], proba[mask]
        by_corpus[corpus] = {
            "n": int(mask.sum()),
            "spam_share": float(yt.mean()),
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp)),
            "recall": float(recall_score(yt, yp)),
            "f1": float(f1_score(yt, yp)),
            "roc_auc": float(roc_auc_score(yt, pr)),
            "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        }

    vocab_size = len(model.named_steps["tfidf"].vocabulary_)
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            **cleaning,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "spam_share": float(df["label"].mean()),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "median_words": {
                "legitimate": float(df.loc[df.label == 0, "text"].str.split().str.len().median()),
                "spam": float(df.loc[df.label == 1, "text"].str.split().str.len().median()),
            },
        },
        "cv_folds": CV_FOLDS,
        "model_search": results,
        "selected_model": chosen,
        "selected_params": results[chosen]["best_params"],
        "selection_rule": (
            f"Best {CV_FOLDS}-fold CV F1 (spam class) on the training split; families within "
            f"{SELECTION_TOLERANCE} of the best are tie-broken in favour of Naive Bayes (simpler)."
        ),
        "selection_outcome": selection_summary(results),
        "vocabulary_size": vocab_size,
        "threshold": DEFAULT_THRESHOLD,
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_curve": _downsample_curve(fpr, tpr),
            "pr_curve": _downsample_curve(rec, prec),
            # Per-message test probabilities let the app recompute metrics at any threshold.
            "y_true": [int(v) for v in y_test],
            "y_proba": [round(float(p), 5) for p in proba],
            "n_false_positives": int(((y_pred == 1) & (y_test.values == 0)).sum()),
            "n_false_negatives": int(((y_pred == 0) & (y_test.values == 1)).sum()),
            "by_corpus": by_corpus,
        },
        "top_terms": global_top_terms(model, n=20),
    }
    return model, metrics


def main() -> dict:
    model, metrics = train_and_evaluate()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for name, r in metrics["model_search"].items():
        print(f"  {name:<26} CV F1 {r['cv_f1']:.4f} +/- {r['cv_f1_std']:.4f}  best {r['best_params']}")
    print(f"Selected: {metrics['selected_model']} {metrics['selected_params']}")
    print("Selection outcome:", metrics["selection_outcome"]["text"])
    t = metrics["test"]
    print(f"Test  acc {t['accuracy']:.4f}  P {t['precision']:.4f}  R {t['recall']:.4f}  "
          f"F1 {t['f1']:.4f}  ROC-AUC {t['roc_auc']:.4f}  AP {t['average_precision']:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:", t["confusion_matrix"])
    for corpus, r in t["by_corpus"].items():
        print(f"  {corpus:<13} n={r['n']:5d} acc {r['accuracy']:.4f} P {r['precision']:.4f} "
              f"R {r['recall']:.4f} F1 {r['f1']:.4f} AUC {r['roc_auc']:.4f} CM {r['confusion_matrix']}")
    print("Vocabulary size:", metrics["vocabulary_size"])
    print("Top spam terms:", [w for w, _ in metrics["top_terms"]["spam"][:15]])
    print("Top legit terms:", [w for w, _ in metrics["top_terms"]["legitimate"][:15]])
    return metrics


if __name__ == "__main__":
    main()
