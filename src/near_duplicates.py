"""Supplementary analysis: how much do near-duplicate e-mails inflate the test score?

    python -m src.near_duplicates      (needs the downloaded data; ~1 minute)

Exact duplicates are removed before the train/test split, but spam campaigns
and newsletters often send almost identical copies. If a test e-mail has a
near-twin in the training split, classifying it is easier than classifying a
genuinely new e-mail, so the headline test score is slightly optimistic.

For every test e-mail we find the highest cosine similarity to any training
e-mail (TF-IDF vectors are L2-normalised, so a dot product is the cosine), then
recompute the metrics on the test e-mails that have NO near-twin above several
thresholds. The official evaluation in metrics.json is left unchanged; this
file only adds context. Results go to models/near_duplicates.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, PROCESSED_PATH, RANDOM_STATE, TEST_SIZE
from src.data import load_dataset
from src.predict import load_model

OUT_PATH = MODELS_DIR / "near_duplicates.json"
THRESHOLDS = [0.95, 0.9, 0.8]


def nearest_training_similarity(model, train_text, test_text, block: int = 400) -> np.ndarray:
    vec = model.named_steps["tfidf"]
    train_t = vec.transform(train_text).T.tocsc()
    test_m = vec.transform(test_text).tocsr()
    out = np.zeros(test_m.shape[0])
    for start in range(0, test_m.shape[0], block):
        out[start:start + block] = (test_m[start:start + block] @ train_t).toarray().max(axis=1)
    return out


def main() -> dict:
    df = load_dataset() if not PROCESSED_PATH.exists() else pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["corpus"] + "/" + df["label"].astype(str), random_state=RANDOM_STATE
    )
    model = load_model()
    sim = nearest_training_similarity(model, train_df["text"], test_df["text"])
    y = test_df["label"].to_numpy()
    pred = (model.predict_proba(test_df["text"])[:, 1] >= 0.5).astype(int)

    def scores(mask):
        return {"n": int(mask.sum()), "share": float(mask.mean()),
                "accuracy": float(accuracy_score(y[mask], pred[mask])),
                "precision": float(precision_score(y[mask], pred[mask])),
                "recall": float(recall_score(y[mask], pred[mask])),
                "f1": float(f1_score(y[mask], pred[mask]))}

    result = {
        "exact_overlap": len(set(train_df["text"]) & set(test_df["text"])),
        "similarity_percentiles": {str(q): float(np.quantile(sim, q / 100)) for q in (25, 50, 75, 90, 99)},
        "all": scores(np.ones(len(y), bool)),
        "without_near_twin": {str(t): scores(sim < t) for t in THRESHOLDS},
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
