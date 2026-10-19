# Development Log: Email Spam Classifier

A chronological record of the work actually done, all on **19 October 2026**.

## Environment
Windows 11, Python 3.13.1; Streamlit 1.66.0, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4, Plotly 7.1.0,
joblib 1.6.0, pytest 9.1.1 (pinned in `requirements.txt`).

## 1. Dataset choice
- **Considered:** the SMS Spam Collection (UCI), used by most student projects. **Rejected** because the
  project is an *e-mail* classifier and SMS messages are much shorter and stylistically different.
- **Chosen first:** the Apache SpamAssassin public corpus (real e-mail). Checked the readme: copyright
  remains with the senders, so the data must be downloaded by script and not committed.

## 2. Parsing (SpamAssassin)
- Wrote a MIME parser keeping only the subject and body (plain text preferred, otherwise HTML converted to text).
- Each archive contains a `cmds` helper file, which is skipped.
- Result: 6,046 messages parsed (readme: 6,047; one mislabelled message was removed by the maintainers in
  2005); 124 exact duplicates removed, leaving 5,922.
- Inspected samples: subjects and bodies were extracted cleanly; HTML was stripped.

## 3. First model (SpamAssassin only)
- TF-IDF + GridSearchCV over NB α ∈ {0.01…1} and LR C ∈ {1, 10, 100}.
- CV F1: NB 0.9673 (α = 0.01), LR 0.9704 (C = 100). Within the 0.005 tolerance, so NB was selected.
  Test F1 0.9605, ROC-AUC 0.9968.
- **Problem 1: best hyperparameters on the grid edge** (α = 0.01 is the smallest value, C = 100 the
  largest), so the true optimum might lie outside the grid. **Fix:** widened the grids.
- **Problem 2: dataset bias.** The top "legitimate" terms were `razor`, `perl`, `exmh`, `redhat`, `rpm
  list`: the ham is mostly Linux/Perl mailing lists. Tested on hand-written modern e-mails: invoice
  e-mail **64% spam**, internship-report notice **61% spam** (both false positives). The model had partly
  learned "technical mailing list ⇒ legitimate".

## 4. Second corpus: Enron-Spam
- Checked availability (aueb.gr), the readme and the citation. Ham = real business e-mail. Added all six
  pre-processed subsets.
- Combined dataset: 39,762 parsed → 51 empty + 3,328 duplicates removed → **36,383** e-mails.
  0 duplicate texts had conflicting labels.
- Split changed to stratify on **corpus + label**, and per-corpus test metrics were added.
- Added **corpus-identity stop words** (`enron`, `ect`, `hou`, `spamassassin`, `taint`).
- Training run: LR CV F1 0.9847 vs NB 0.9794, so LR was selected. Test F1 0.9834.

## 5. Problem 3: link encoding differed between corpora
- Inspection of top terms: `http` was the #1 *spam* term, while `urltoken` leaned *legitimate*.
- Cause: Enron files are pre-tokenised, so links look like `http : / / www . site . com`, which the URL
  regex did not match. In SpamAssassin, links became `urltoken`, and most SpamAssassin ham (85%) contains links.
- Consequence: a real link in a pasted e-mail would have pushed it towards "legitimate". This is a genuine bug.
- **Fix:** added spaced-URL and spaced-e-mail patterns (`SPACED_URL_RE`, `SPACED_EMAIL_RE`), checked on real samples.
- Also added Enron mailbox-owner names (vince, kaminski, louise, sally, beck, daren, michelle, lokay) and
  `houston` to the identity terms, because `vince` and `louise` had appeared among the top legitimate terms.

## 6. Final training run
```
Multinomial Naive Bayes    CV F1 0.9771 +/- 0.0009  best {'alpha': 0.01}
Logistic Regression        CV F1 0.9842 +/- 0.0018  best {'C': 10.0}
Selected: Logistic Regression {'C': 10.0}
Test  acc 0.9847  P 0.9803  R 0.9859  F1 0.9831  ROC-AUC 0.9989  AP 0.9986
Confusion matrix [[TN FP] [FN TP]]: [[3939, 65], [46, 3227]]
  Enron-Spam    n= 6092 acc 0.9869 P 0.9866 R 0.9859 F1 0.9862 AUC 0.9991
  SpamAssassin  n= 1185 acc 0.9738 P 0.9323 R 0.9862 F1 0.9585 AUC 0.9982
```
Both best hyperparameters are now inside their grids. Top spam terms: remove, click, money, viagra…;
top legitimate: wrote, thanks, attached, questions, forwarded… Training takes about 7 minutes; the model
file is 0.6 MB.

Hand-written e-mail check after the fix: invoice 64% → **3.3%**, internship notice 61% → **1.3%**;
the spam samples stayed at 99–100%.

## 7. Other issues fixed during development

| Issue | Fix |
|---|---|
| The banner text said "calibrated confidence"; Naive Bayes probabilities are known to be poorly calibrated, and calibration was never measured | Changed to "the model's confidence"; calibration is listed as future work |
| An `assert` inside `classify()` (explanation must match the probability) would crash the app if it ever failed, and is stripped by `python -O` | Moved the check into a unit test (`test_explanation_is_exact`) |
| The "How it works" tab only described Naive Bayes | Added the Logistic Regression explanation, which is shown when LR is selected |
| One of my own tests expected a double space after HTML stripping (`"Only  html"`), but whitespace is correctly collapsed | Fixed the test expectation |

## 8. Verification
- `pytest`: 27 passed (see TESTING.md).
- Headless `streamlit run app.py`: health `ok`, page HTTP 200, no errors in the log.
- Installed `requirements.txt` into a brand-new virtual environment and ran the full suite there: **27 passed**.
  (The first attempt failed with `WinError 206: filename too long` because the test environment was placed
  at a very long path; at a shorter path it worked. This led to the "Windows note" in the README.)

## 9. Not done / future work
- No recent (post-2010) e-mail data; no probability calibration; no batch CSV scoring; not tested on mobile.

## 10. Refinement phase (19 October 2026)
- **Model-selection audit:** the Model card showed "Selected: Logistic Regression" next to a rule mentioning a Naive Bayes tie-break. Checked: the loaded artifact is `LogisticRegression(C=10)`; the selection rule recomputed from the stored CV scores gives LR; `metrics.json` agrees; the saved model reproduces the stored test probabilities. The gap (0.0071) exceeds the 0.005 tolerance, so the tie-break never applied. This was a display issue: the rule was shown without its outcome. `train.py` now stores `selection_outcome` (gap, tolerance, whether the tie-break applied) and the app shows it; a new test locks the consistency.
- **MemoryError when retraining:** `GridSearchCV(n_jobs=-1)` started 12 workers, each building a bigram vocabulary; this exhausted memory on a 16 GB laptop while another job ran. Fixed with `N_JOBS = 4` in `src/config.py` (speed only). Retraining reproduced identical metrics.
- **Explanation text bug:** the "Starting point" was described as the class prior, which is true only for Naive Bayes. For the selected Logistic Regression it is the intercept (+0.74, whereas the prior log-odds would be −0.20). The help text now depends on the model.
- **Near-duplicate analysis (new, supplementary):** `python -m src.near_duplicates`. 27% of test e-mails have a training e-mail with TF-IDF cosine ≥ 0.9; F1 on the rest is 0.975 (vs 0.983 overall). Shown on the Model card; the official evaluation is unchanged.
- **Model card:** plain-language error summary, and an explicit warning that 1999–2005 scores don't measure modern spam. The model-selection table height was fixed (it had hidden the LR rows).
- **Light/dark mode and visuals** as in the other projects; checked with screenshots in both modes.
- **Docs:** added `docs/MODEL_CARD.md`; the long-form study notes are no longer part of the repository; README quick start added.
