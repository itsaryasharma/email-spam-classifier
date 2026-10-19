"""Exact, per-term explanations for the two linear text models we use.

Both Multinomial Naive Bayes and Logistic Regression are *linear* in the
TF-IDF features when written in log-odds form:

    log P(spam|x) / P(legit|x) = base + sum_i x_i * w_i

  * Naive Bayes:          w_i = log P(term_i|spam) - log P(term_i|legit)
                          base = log P(spam) - log P(legit)          (class priors)
  * Logistic Regression:  w_i = coef_i,  base = intercept

So x_i * w_i is the exact contribution of term i to the decision; no
approximation (such as SHAP or LIME) is needed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def term_weights(model) -> tuple[np.ndarray, float]:
    """Return (w, base) for the classifier inside the pipeline."""
    clf = model.named_steps["clf"]
    if isinstance(clf, MultinomialNB):
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        base = float(clf.class_log_prior_[1] - clf.class_log_prior_[0])
    elif isinstance(clf, LogisticRegression):
        w = clf.coef_[0]
        base = float(clf.intercept_[0])
    else:
        raise TypeError(f"Unsupported classifier: {type(clf).__name__}")
    return np.asarray(w), base


def explain(model, text: str) -> dict:
    """Per-term contributions (in log-odds) for one message."""
    vec = model.named_steps["tfidf"]
    w, base = term_weights(model)
    x = vec.transform([text])
    names = vec.get_feature_names_out()
    contrib = {names[j]: float(v * w[j]) for j, v in zip(x.indices, x.data)}
    log_odds = base + sum(contrib.values())
    return {
        "base": base,
        "log_odds": float(log_odds),
        "contributions": dict(sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)),
        "n_known_terms": len(contrib),
    }


def global_top_terms(model, n: int = 20, min_doc_freq_rank: int = 5000) -> dict:
    """The terms with the strongest spam / legitimate weights.

    To avoid listing very rare words, only the `min_doc_freq_rank` most common
    terms (lowest IDF) are considered.
    """
    vec = model.named_steps["tfidf"]
    w, _ = term_weights(model)
    names = vec.get_feature_names_out()
    common = np.argsort(vec.idf_)[:min_doc_freq_rank]
    order = common[np.argsort(w[common])]
    return {
        "spam": [(str(names[j]), float(w[j])) for j in order[::-1][:n]],
        "legitimate": [(str(names[j]), float(w[j])) for j in order[:n]],
    }
