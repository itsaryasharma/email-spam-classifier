# Testing: Email Spam Classifier

## How to run
```bash
pytest          # from the project folder
pytest -v       # one line per test
```
One test (`test_saved_model_reproduces_stored_test_probabilities`) needs the downloaded dataset and is
**skipped** automatically if `data/processed/emails.csv.gz` is missing. Run `python -m src.data` first to include it.

## What is tested (27 tests)

| Area | Tests |
|---|---|
| Normalisation | URLs (normal and Enron-style spaced), e-mail addresses (both styles), money symbols, numbers → placeholders; plain text unchanged |
| Parsing | HTML → text (scripts/styles removed, entities decoded); MIME prefers `text/plain`; HTML-only MIME; Enron file format |
| Input validation | Empty, whitespace-only, `None` and non-string input rejected; < 5 words warns; > 50,000 characters truncated with a warning; invalid threshold rejected |
| Prediction | Obvious spam > 0.9; obvious legitimate < 0.1 |
| **Explanation exactness** | `sigmoid(base + Σ contributions)` equals `predict_proba` (rel. tol. 1e-6) |
| Unknown words | Nonsense text gives 0 known terms and a vocabulary warning |
| Highlighting | Segments re-assemble the original text exactly; words and URLs get the right weights |
| **Metric honesty** | Stored test predictions reproduce the stored confusion matrix, precision and recall |
| **Model reproducibility** | Re-splitting the data reproduces the stored test labels, and the saved model reproduces the stored probabilities (within 1e-4) |
| App (headless `AppTest`) | Empty state shows a prompt; sample buttons give "Likely spam" / "Likely legitimate"; a short message shows a warning; changing the threshold updates the verdict |

## Results (19 October 2026)
```
...........................                                              [100%]
27 passed in 26.86s
```
(Run with `-W error::DeprecationWarning -W error::FutureWarning`, so deprecated APIs would have failed the run.)

Additional checks:
- `python -W error -m src.data` and `python -W error -m src.train`: no warnings.
- Headless server: `/_stcore/health` returned `ok` and `/` returned HTTP 200.

## Not covered automatically
- Visual layout in a real browser; mobile layout.
- Performance on modern (post-2005) e-mail: there is no labelled modern dataset in this project.

## Screenshots to capture for the report
1. Analyse tab with **Prize offer** loaded (verdict, score breakdown, highlighted words).
2. **Project update** sample: legitimate verdict with green highlights.
3. A very short message showing the reliability warning.
4. Model card: metrics, confusion matrix, ROC and PR curves.
5. Threshold explorer at 0.5 and at 0.9.
6. Dataset tab: corpus composition chart.

## Refinement-phase verification (19 October 2026)
Command: `python -m pytest -q -W error::DeprecationWarning -W error::FutureWarning` → **29 passed** (with the
downloaded data; without it, the data-dependent reproducibility test is skipped).

New tests:
- `test_selected_model_is_consistent_everywhere`: the selection rule recomputed from the stored CV scores == `metrics.json` selected model == class of the loaded artifact == stored hyperparameters; the stored selection outcome is reproduced.
- `test_selection_summary_handles_a_near_tie`: the Naive Bayes tie-break is applied when the gap is within the tolerance.

Additional checks: `python -W error -m src.train` (identical metrics after retraining), `python -W error -m
src.near_duplicates`, and screenshots of the Analyse and Model card tabs in light and dark mode.

**Fresh-clone check:** only the files Git would publish were copied, a new virtual environment was created with
`py -3.13 -m venv`, `pip install -r requirements.txt` succeeded, `python -m pytest` gave 28 passed, 1 skipped (no downloaded data in a fresh clone), and
`python -m streamlit run app.py` started (health `ok`, HTTP 200).
