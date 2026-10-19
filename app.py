"""Email Spam Classifier — Streamlit application.

Run with:  streamlit run app.py
"""
import html
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import DEFAULT_THRESHOLD, MODELS_DIR
from src.predict import (
    classify,
    ensure_trained,
    highlight_segments,
    load_metrics,
    load_model,
    metrics_at_threshold,
)
from src.text import PLACEHOLDERS

SPAM_COLOR = "#E11D48"
HAM_COLOR = "#059669"
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)

# Short sample messages written for this demo. They are NOT taken from the
# training data, so they show how the model behaves on new text.
EXAMPLES = {
    "Prize offer": (
        "Subject: Congratulations! You have been selected\n\n"
        "Dear friend, you have been selected to receive a FREE $500 gift card. Click here now to claim "
        "your prize: http://example.com/claim\nThis offer expires in 24 hours. To be removed from our "
        "mailing list, reply REMOVE."
    ),
    "Account warning": (
        "Subject: Urgent: verify your account\n\n"
        "Dear customer, we detected unusual activity on your account. Please verify your password within "
        "24 hours by clicking the link below or your account will be suspended.\nhttp://example.org/verify"
    ),
    "Project update": (
        "Subject: Re: notes from Tuesday's project meeting\n\n"
        "Hi Priya, thanks for sending the notes. I have updated the test plan and pushed the changes to the "
        "repository. Can we review the deployment script together on Thursday afternoon? I will bring the "
        "benchmark results.\n\nCheers,\nRahul"
    ),
    "Invoice": (
        "Subject: Invoice for October\n\n"
        "Hello, please find attached the invoice for the October consulting work. Payment is due within "
        "30 days. Let me know if you have any questions.\n\nRegards,\nMeera"
    ),
}


@st.cache_resource(show_spinner="Preparing the model (first run downloads and trains; ~5 minutes)…")
def get_model_and_metrics():
    ensure_trained()
    return load_model(), load_metrics()


def set_example(name: str) -> None:
    st.session_state["message"] = EXAMPLES[name]


def clear_message() -> None:
    st.session_state["message"] = ""


# --------------------------------------------------------------------------- #
# Analyse tab
# --------------------------------------------------------------------------- #
def highlighted_html(text: str, contributions: dict) -> str:
    segments = highlight_segments(text, contributions)
    weights = [abs(w) for _, w in segments if w]
    scale = max(weights) if weights else 1.0
    parts = []
    for seg, w in segments:
        safe = html.escape(seg).replace("\n", "<br>")
        if not w:
            parts.append(safe)
            continue
        alpha = 0.15 + 0.7 * min(abs(w) / scale, 1.0)
        rgb = "225,29,72" if w > 0 else "5,150,105"
        parts.append(f'<span title="{w:+.3f}" style="background:rgba({rgb},{alpha:.2f});'
                     f'border-radius:3px;padding:0 2px">{safe}</span>')
    return ('<div style="border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:12px 14px;max-height:320px;'
            'overflow:auto;line-height:1.7;font-size:.95rem;background:rgba(128,128,128,.05)">' + "".join(parts) + "</div>")


def render_result(result: dict, model_name: str) -> None:
    p, thr = result["spam_probability"], result["threshold"]
    margin = abs(p - thr)
    if result["is_spam"]:
        verdict, color = "Likely spam", SPAM_COLOR
    else:
        verdict, color = "Likely legitimate", HAM_COLOR
    note = "close to the decision threshold, so treat this result with caution" if margin < 0.15 else \
        "well away from the decision threshold"
    ui.result_card("Model verdict", verdict, f"Spam probability {p:.1%} · threshold {thr:.0%} · {note}", color)

    fig = go.Figure(go.Bar(x=[p], y=[""], orientation="h", marker_color=color, width=0.5,
                           text=[f"{p:.1%}"], textposition="inside"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#64748B",
                  annotation_text=f"threshold {thr:.0%}", annotation_position="top")
    fig.update_layout(height=110, margin=dict(l=10, r=10, t=28, b=10), plot_bgcolor="rgba(128,128,128,0.15)",
                      xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    for w in result["warnings"]:
        st.warning(w)

    exp = result["explanation"]
    contrib = exp["contributions"]
    spam_ev = sum(v for v in contrib.values() if v > 0)
    ham_ev = sum(v for v in contrib.values() if v < 0)
    c1, c2, c3, c4 = st.columns(4)
    if "Bayes" in model_name:
        start_help = "Log-odds of spam from the class balance alone (the prior), before reading any words."
    else:
        start_help = ("The model's intercept: the score a message would get if it contained none of the "
                      "words the model knows. It is learned during training and is not the class balance.")
    c1.metric("Starting point", f"{exp['base']:+.2f}", help=start_help, border=True)
    c2.metric("Spam evidence", f"{spam_ev:+.2f}", help="Sum of contributions of spam-leaning terms.", border=True)
    c3.metric("Legit evidence", f"{ham_ev:+.2f}", help="Sum of contributions of legitimate-leaning terms.", border=True)
    c4.metric("Final score", f"{exp['log_odds']:+.2f}",
              help="Log-odds = start + spam evidence + legitimate evidence. Probability = 1 / (1 + e^-score).", border=True)


def render_explanation(result: dict) -> None:
    contrib = result["explanation"]["contributions"]
    if not contrib:
        return
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("**Which words pushed the decision?**")
        st.markdown(highlighted_html(result["text"], contrib), unsafe_allow_html=True)
        st.caption("Red words push towards *spam*, green words towards *legitimate*; stronger colour means a "
                   "stronger push. Hover a word to see its exact contribution. Uncoloured words are ignored "
                   "by the model (common words, or words it never saw in training). Two-word phrases are "
                   "shown in the chart.")
    with right:
        st.markdown("**Strongest signals (words and phrases)**")
        items = list(contrib.items())
        top = items[:8] + [kv for kv in items[-8:] if kv not in items[:8]]
        top = [kv for kv in top if abs(kv[1]) > 1e-9]
        top.sort(key=lambda kv: kv[1])
        labels = [f"[{PLACEHOLDERS[k]}]" if k in PLACEHOLDERS else k for k, _ in top]
        fig = go.Figure(go.Bar(
            x=[v for _, v in top], y=labels, orientation="h",
            marker_color=[SPAM_COLOR if v > 0 else HAM_COLOR for _, v in top],
        ))
        fig.update_layout(height=max(220, 26 * len(top) + 40), margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Contribution to log-odds (→ spam)")
        st.plotly_chart(fig, width="stretch")


def render_analyse(model, metrics: dict) -> None:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Message")
        st.caption("Paste an e-mail (subject and body) or load a sample. Samples were written for this demo; "
                   "they are not from the training data.")
        cols = st.columns(len(EXAMPLES))
        for col, name in zip(cols, EXAMPLES):
            col.button(name, on_click=set_example, args=(name,), width="stretch", key=f"ex_{name}")
        st.text_area("E-mail text", key="message", height=260, label_visibility="collapsed",
                     placeholder="Subject: ...\n\nDear ...")
        b1, b2 = st.columns([1, 1])
        b1.button("Analyse message", type="primary", width="stretch")
        b2.button("Clear", on_click=clear_message, width="stretch")
        with st.expander("Advanced: decision threshold"):
            st.slider("Flag as spam when spam probability is at least", 0.05, 0.95,
                      DEFAULT_THRESHOLD, 0.05, key="threshold")
            st.caption("A higher threshold means fewer legitimate e-mails wrongly flagged, but more spam gets "
                       "through. See the *Model card* tab for the measured trade-off.")
    with right:
        st.subheader("Analysis")
        text = st.session_state.get("message", "")
        if not text.strip():
            st.info("Paste a message, or click one of the samples, to see the analysis here.")
            result = None
        else:
            try:
                result = classify(model, text, st.session_state.get("threshold", DEFAULT_THRESHOLD))
            except ValueError as err:
                result = None
                st.error(str(err))
            if result:
                render_result(result, metrics["selected_model"])
    if result:
        st.divider()
        render_explanation(result)
    st.divider()
    st.markdown(
        "🛡️ **This is a text classifier, not a security scanner.** It reads only the words in the message. "
        "It does **not** open links, scan attachments, check the sender's address or domain, or detect "
        "malware. A *legitimate* verdict does **not** mean a message is safe. If a message asks for "
        "passwords, payments or urgent action, verify it through another channel."
    )


# --------------------------------------------------------------------------- #
# Model card tab
# --------------------------------------------------------------------------- #
def render_model_card(metrics: dict) -> None:
    t, ds = metrics["test"], metrics["dataset"]
    st.subheader("How well does the model perform?")
    st.caption(
        f"Evaluated once on {ds['n_test']:,} held-out e-mails (stratified 80/20 split by corpus and label, "
        f"random_state={ds['random_state']}). All numbers are produced by `src/train.py` and read from "
        f"`models/metrics.json`. Trained {metrics['trained_at_utc']} with scikit-learn "
        f"{metrics['sklearn_version']}."
    )
    m = st.columns(5)
    m[0].metric("Accuracy", f"{t['accuracy']:.1%}", border=True)
    m[1].metric("Precision (spam)", f"{t['precision']:.3f}", border=True)
    m[2].metric("Recall (spam)", f"{t['recall']:.3f}", border=True)
    m[3].metric("F1 (spam)", f"{t['f1']:.3f}", border=True)
    m[4].metric("ROC-AUC", f"{t['roc_auc']:.4f}", border=True)
    (tn, fp), (fn, tp) = t["confusion_matrix"]
    st.markdown(
        f":material/info: **In plain words:** of {tn + fp:,} legitimate test e-mails, **{fp}** "
        f"({fp / (tn + fp):.1%}) were wrongly flagged as spam; of {fn + tp:,} spam e-mails, **{fn}** "
        f"({fn / (fn + tp):.1%}) slipped through."
    )
    st.warning(
        ":material/history: **These scores describe e-mails from 1999–2005**, tested on held-out e-mails from "
        "the *same* two corpora. They do **not** tell us how well the model would work on today's spam and "
        "phishing, which uses different wording. No modern labelled e-mails were available to measure that."
    )

    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown("**Confusion matrix (threshold 50%)**")
        fig = px.imshow(t["confusion_matrix"], x=["legitimate", "spam"], y=["legitimate", "spam"],
                        text_auto=True, color_continuous_scale="Reds",
                        labels=dict(x="Predicted", y="Actual", color="E-mails"))
        fig.update_layout(height=300, margin=PLOT_MARGIN, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown(f"**ROC curve (AUC {t['roc_auc']:.4f})**")
        fig = go.Figure(go.Scatter(x=t["roc_curve"]["x"], y=t["roc_curve"]["y"], mode="lines",
                                   line=dict(color=SPAM_COLOR, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#9CA3AF")))
        fig.update_layout(height=300, margin=PLOT_MARGIN, showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
        st.plotly_chart(fig, width="stretch")
    with c:
        st.markdown(f"**Precision–recall curve (AP {t['average_precision']:.4f})**")
        fig = go.Figure(go.Scatter(x=t["pr_curve"]["x"], y=t["pr_curve"]["y"], mode="lines",
                                   line=dict(color=SPAM_COLOR, width=2.5)))
        fig.update_layout(height=300, margin=PLOT_MARGIN, showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Threshold explorer")
    st.caption("Wrongly flagging a real e-mail (false positive) is usually worse than letting one spam through. "
               "Move the threshold to see the trade-off, recomputed from the stored test-set predictions.")
    thr = st.slider("Threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05, key="card_threshold")
    r = metrics_at_threshold(t["y_true"], t["y_proba"], thr)
    k = st.columns(5)
    k[0].metric("Precision", f"{r['precision']:.3f}", border=True)
    k[1].metric("Recall", f"{r['recall']:.3f}", border=True)
    k[2].metric("F1", f"{r['f1']:.3f}", border=True)
    k[3].metric("Legit flagged as spam", f"{r['fp']:,}", border=True)
    k[4].metric("Spam missed", f"{r['fn']:,}", border=True)

    st.markdown("#### Performance by source corpus")
    rows = [{"Corpus": name, "Test e-mails": v["n"], "Spam share": f"{v['spam_share']:.1%}",
             "Accuracy": f"{v['accuracy']:.3f}", "Precision": f"{v['precision']:.3f}",
             "Recall": f"{v['recall']:.3f}", "F1": f"{v['f1']:.3f}", "ROC-AUC": f"{v['roc_auc']:.4f}"}
            for name, v in t["by_corpus"].items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("A large gap between corpora would mean the model only works on one kind of e-mail.")

    nd_path = MODELS_DIR / "near_duplicates.json"
    if nd_path.exists():
        nd = json.loads(nd_path.read_text(encoding="utf-8"))
        st.markdown("#### Near-duplicate check (supplementary analysis)")
        st.caption("Exact duplicates were removed before splitting, but campaigns and newsletters send *almost* "
                   "identical copies. A test e-mail with a near-twin in the training data is easier to classify, so "
                   "the headline score is slightly optimistic. Below: the same predictions, scored only on test "
                   "e-mails without a near-twin (cosine similarity of TF-IDF vectors). Produced by "
                   "`python -m src.near_duplicates`.")
        rows = [{"Test e-mails scored": "All (reported score)", "E-mails": nd["all"]["n"],
                 "Accuracy": nd["all"]["accuracy"], "F1": nd["all"]["f1"]}]
        rows += [{"Test e-mails scored": f"No training twin with similarity ≥ {t}", "E-mails": v["n"],
                  "Accuracy": v["accuracy"], "F1": v["f1"]} for t, v in nd["without_near_twin"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "E-mails": "{:,}"}),
                     width="stretch", hide_index=True)
        f1_novel = nd["without_near_twin"]["0.9"]["f1"]
        st.caption(f"Reading: about {1 - nd['without_near_twin']['0.9']['share']:.0%} of test e-mails have a "
                   f"near-twin (≥ 0.9) in training. On the rest, F1 is {f1_novel:.3f} instead of "
                   f"{nd['all']['f1']:.3f}: still high, but this is the fairer figure for genuinely new e-mails "
                   "from the same era.")

    d, e = st.columns(2, gap="large")
    with d:
        st.markdown(f"**Model selection ({metrics['cv_folds']}-fold CV on the training split)**")
        rows = []
        for fam, res in metrics["model_search"].items():
            for g in res["grid"]:
                rows.append({"Model": fam, "Setting": ", ".join(f"{k}={v:g}" for k, v in g["params"].items()),
                             "CV F1": g["mean_f1"], "± std": g["std_f1"]})
        st.dataframe(pd.DataFrame(rows).style.format({"CV F1": "{:.4f}", "± std": "{:.4f}"}),
                     width="stretch", hide_index=True, height=35 * (len(rows) + 1) + 3)
        params = ", ".join(f"{k} = {v:g}" for k, v in metrics["selected_params"].items())
        st.success(f"**Selected: {metrics['selected_model']} ({params}).** "
                   f"{metrics['selection_outcome']['text']}")
        st.caption(f"Rule declared in `src/train.py` before the final run: {metrics['selection_rule']} "
                   "The application loads exactly this selected model from `models/spam_model.joblib`.")
    with e:
        st.markdown("**Most indicative terms (global model weights)**")
        top = metrics["top_terms"]
        frame = pd.DataFrame(
            [(w, v, "spam") for w, v in top["spam"][:12]] + [(w, v, "legitimate") for w, v in top["legitimate"][:12]],
            columns=["term", "weight", "class"],
        ).sort_values("weight")
        fig = px.bar(frame, x="weight", y="term", color="class", orientation="h",
                     color_discrete_map={"spam": SPAM_COLOR, "legitimate": HAM_COLOR})
        fig.update_layout(height=520, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          xaxis_title="log P(term|spam) − log P(term|legit)" if "Bayes" in metrics["selected_model"]
                          else "coefficient")
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# How it works tab
# --------------------------------------------------------------------------- #
def render_how_it_works(metrics: dict) -> None:
    st.subheader("From raw text to a verdict")
    st.markdown(
        f"""
1. **Validation**: empty input is rejected; very long input is truncated to 50,000 characters; fewer than
   5 words triggers a reliability warning.
2. **Normalisation**: lower-case; links → `urltoken`, e-mail addresses → `emailtoken`, currency symbols →
   `moneytoken`, numbers → `numtoken`. The model learns "contains a link", not specific links.
3. **Tokenisation**: alphabetic words of 2+ letters, minus common English stop words and corpus-identity
   words (e.g. *enron*), plus two-word phrases (bigrams).
4. **TF-IDF weighting** over a vocabulary of {metrics['vocabulary_size']:,} terms learned from the training split.
5. **{metrics['selected_model']}** turns the weighted terms into a spam probability.
6. **Decision**: spam if the probability ≥ the threshold (default 50%).
        """
    )
    st.markdown("**TF-IDF**: a term matters if it is frequent in *this* e-mail but rare across all e-mails:")
    st.latex(r"\text{tfidf}(t,d) = \big(1+\log \text{tf}(t,d)\big)\cdot\Big(\log\frac{1+N}{1+\text{df}(t)}+1\Big)"
             r"\quad\text{(then each e-mail vector is scaled to length 1)}")
    if "Bayes" in metrics["selected_model"]:
        st.markdown("**Multinomial Naive Bayes** applies Bayes' theorem and assumes the terms are independent "
                    "given the class. In log-odds form the decision is a sum of per-term contributions, which "
                    "is why the app can show *exactly* how each word moved the score:")
        st.latex(r"\log\frac{P(\text{spam}\mid x)}{P(\text{legit}\mid x)} = "
                 r"\underbrace{\log\frac{P(\text{spam})}{P(\text{legit})}}_{\text{starting point}} + "
                 r"\sum_i x_i\,\underbrace{\Big(\log P(t_i\mid\text{spam}) - \log P(t_i\mid\text{legit})\Big)}_{\text{term weight}}")
        st.markdown(f"Smoothing α = {metrics['selected_params'].get('alpha')} prevents zero probabilities for "
                    "terms never seen with one of the classes.")
    else:
        st.markdown("**Logistic Regression** learns one weight per term. The weighted sum of the e-mail's TF-IDF "
                    "values (the log-odds) is passed through the sigmoid function to give a probability. "
                    "Because the score is a plain sum, the app can show *exactly* how each word moved it:")
        st.latex(r"\log\frac{P(\text{spam}\mid x)}{P(\text{legit}\mid x)} = "
                 r"\underbrace{b}_{\text{starting point}} + \sum_i x_i\,\underbrace{w_i}_{\text{term weight}}"
                 r"\qquad P(\text{spam}\mid x) = \frac{1}{1+e^{-\text{log-odds}}}")
        st.markdown(f"The weights are learned by minimising log-loss with L2 regularisation "
                    f"(C = {metrics['selected_params'].get('C'):g}; larger C = weaker regularisation). "
                    "Multinomial Naive Bayes, the classic baseline, was also tuned and is compared on the "
                    "*Model card* tab.")
    st.markdown("#### What this model can and cannot tell you")
    st.markdown(
        """
- ✅ Whether the **wording** of a message resembles the spam or the legitimate e-mails it was trained on.
- ⚠️ It was trained on e-mails from **1999–2005**. Modern spam and phishing use new wording, so
  accuracy on today's e-mail will be lower than the test score.
- ⚠️ It cannot see links, attachments, sender reputation, images or e-mail headers.
- ⚠️ Probabilities are the model's confidence, **not** a guarantee. A determined spammer can avoid trigger words.
- ℹ️ It is a demonstration of text classification, not a replacement for a production spam filter.
        """
    )


# --------------------------------------------------------------------------- #
# Dataset tab
# --------------------------------------------------------------------------- #
def render_dataset(metrics: dict) -> None:
    ds = metrics["dataset"]
    st.subheader("Training data: two public e-mail corpora")
    st.markdown(
        """
| Corpus | Years | Legitimate e-mail | Spam |
|---|---|---|---|
| **Apache SpamAssassin public corpus** | 2002–2005 | Mostly technical mailing lists and newsletters | Spam received by the corpus maintainers |
| **Enron-Spam** (Metsis, Androutsopoulos & Paliouras, CEAS 2006) | 1999–2005 | Business e-mail of six Enron employees | Spam from several spam traps and personal collections |

Combining them gives the model both technical and business-style legitimate e-mail. Only the
**subject and body text** are used, matching what a user can paste into the app. Neither corpus has an
open licence (copyright in the messages remains with the senders), so the data is downloaded by
`src/data.py` and is **not** stored in this repository.
        """
    )
    rows = [{"Corpus / class": k, "Parsed": ds["parsed_by_corpus_label"].get(k, 0),
             "After cleaning": ds["final_by_corpus_label"].get(k, 0)}
            for k in ds["final_by_corpus_label"]]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(
            f"- Parsed **{ds['parsed']:,}** e-mails; removed **{ds['dropped_empty']}** empty and "
            f"**{ds['dropped_duplicates']:,}** exact duplicates (duplicates could appear in both train and test "
            f"and inflate the score). Duplicate texts with conflicting labels: "
            f"{ds['duplicate_texts_with_conflicting_labels']}.\n"
            f"- Final dataset: **{ds['final']:,}** e-mails, **{ds['spam_share']:.1%} spam**. "
            f"Training split {ds['n_train']:,} · test split {ds['n_test']:,}.\n"
            f"- Median length: {ds['median_words']['legitimate']:.0f} words (legitimate), "
            f"{ds['median_words']['spam']:.0f} words (spam)."
        )
    with right:
        frame = pd.DataFrame(rows)
        frame["corpus"] = frame["Corpus / class"].str.split(" / ").str[0]
        frame["class"] = frame["Corpus / class"].str.split(" / ").str[1]
        fig = px.bar(frame, x="corpus", y="After cleaning", color="class", barmode="group",
                     color_discrete_map={"spam": SPAM_COLOR, "legitimate": HAM_COLOR}, text_auto=True)
        fig.update_layout(height=320, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          yaxis_title="E-mails", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Email Spam Classifier", page_icon="✉️", layout="wide")
    ui.inject_css()
    try:
        model, metrics = get_model_and_metrics()
    except Exception as exc:  # e.g. no internet on first run
        st.error(f"The model could not be loaded or trained: {exc}")
        st.info("Check your internet connection (the first run downloads the datasets), then run "
                "`python -m src.train` from the project folder and reload this page.")
        st.stop()

    st.session_state.setdefault("message", "")
    t = metrics["test"]
    ui.hero(
        "Email Spam Classifier",
        "Paste an e-mail to get a spam verdict, the model's confidence, and the exact words that "
        "drove the decision.",
        [f"TF-IDF + {metrics['selected_model']}", f"F1 {t['f1']:.3f} · ROC-AUC {t['roc_auc']:.3f}",
         f"{metrics['dataset']['final']:,} training e-mails"],
    )
    tab_a, tab_m, tab_h, tab_d = st.tabs(["✉️ Analyse", "📊 Model card", "🧠 How it works", "🗂️ Dataset"])
    with tab_a:
        render_analyse(model, metrics)
    with tab_m:
        render_model_card(metrics)
    with tab_h:
        render_how_it_works(metrics)
    with tab_d:
        render_dataset(metrics)
    ui.footer("Email Spam Classifier · Arya Kumar · Educational machine-learning project")


main()
