# Model Card: Email Spam Classifier

## Overview
| | |
|---|---|
| Task | Binary text classification of an e-mail's subject + body: spam (1) or legitimate (0) |
| Model | `Pipeline(TfidfVectorizer(...), LogisticRegression(C=10, max_iter=5000))` (L2, lbfgs) |
| Data | Apache SpamAssassin public corpus (2002–2005) + Enron-Spam pre-processed (1999–2005); 36,383 unique e-mails, 45.0% spam. See `DATA_SOURCES.md` |
| Code | `src/data.py` (parsing), `src/text.py` (normalisation), `src/train.py`, `src/explain.py`, `src/predict.py`, `src/near_duplicates.py` |
| Artifacts | `models/spam_model.joblib`, `models/metrics.json`, `models/near_duplicates.json` |

## Preprocessing
- MIME parsing; keep the subject + readable body (prefer text/plain, otherwise HTML converted to text); headers dropped.
- Empty messages (51) and exact duplicate texts (3,328) removed **before** splitting.
- Normalisation: lower-case; URLs → `urltoken`, e-mail addresses → `emailtoken`, currency → `moneytoken`, numbers → `numtoken` (both normal and Enron's spaced format).
- TF-IDF: alphabetic tokens of ≥ 2 letters, unigrams + bigrams, `min_df=2`, `max_df=0.95`, 30,000 terms, sublinear tf, English stop words plus 14 corpus-identity words (e.g. *enron*, *vince*, *houston*) that would otherwise act as source shortcuts.

## Model selection
- Stratified 80/20 split by corpus and label (`random_state=42`): 29,106 / 7,277.
- `GridSearchCV`, 5-fold stratified CV, F1, **training split only**: Multinomial NB (α ∈ 0.001–1) vs Logistic Regression (C ∈ 1–1000).
- Rule: best CV F1; if within 0.005 of the best, prefer Naive Bayes (simpler).
- **Outcome:** LR (C = 10) CV F1 **0.9842**; NB (α = 0.01) **0.9771**; gap 0.0071 > 0.005, so the tie-break was **not** triggered and LR was selected. The saved artifact is this LR pipeline; a test checks the selection, the stored metrics and the loaded model all agree.

## Results (held-out test set, 7,277 e-mails, threshold 0.5)
| Metric (spam = positive) | Value |
|---|---|
| Accuracy | 0.9847 |
| Precision / Recall / F1 | 0.9803 / 0.9859 / 0.9831 |
| ROC-AUC / Average precision | 0.9989 / 0.9986 |
| Confusion matrix | TN 3,939 · FP 65 · FN 46 · TP 3,227 |
| Per corpus F1 | Enron-Spam 0.986 · SpamAssassin 0.959 |

**Near-duplicate check** (`python -m src.near_duplicates`): 27% of test e-mails have a training e-mail with
TF-IDF cosine similarity ≥ 0.9. On test e-mails without such a twin, F1 = **0.975** (≥ 0.8 cut: 0.970).
The headline score is therefore mildly optimistic for genuinely new e-mails.

**Threshold trade-off** (stored test predictions): 0.5 → 65 false alarms / 46 missed spam; 0.9 → 5 / 330.

## Explanations
Logistic regression is linear in log-odds: `log-odds = intercept + Σ tfidf_i × w_i`. The app shows these
exact per-term contributions. The starting point is the model's intercept (+0.74), not the class prior.

## Intended use and limitations
- An educational text classifier, **not a security tool**. It doesn't inspect links, attachments, senders, headers or images, and a "legitimate" verdict doesn't mean an e-mail is safe.
- Trained on 1999–2005 e-mail. High scores on held-out data from the same corpora **do not** indicate performance on modern spam or phishing, which was not measured.
- Remaining source bias: some domain-specific words (e.g. *gas*, *energy*, first names) lean legitimate.
- Very short messages are unreliable (the app warns below 5 words).
- Probability calibration was not evaluated.
