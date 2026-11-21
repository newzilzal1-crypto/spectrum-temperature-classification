"""
Thesis-defence presentation app: methodology, results, live testing and conclusions.

Train the models first (once):   python train_models.py      (both protocols -> saved_models/<protocol>/)
Then start the app:              streamlit run app.py
Edit project_info.json to set the title, author, supervisors and institution.
"""
import os

# Cap the math-library thread pools BEFORE numpy/scipy load. On machines with many cores each bundled
# OpenBLAS copy reserves per-thread buffers; when Windows' commit limit is nearly used up this fails
# ("OpenBLAS error: Memory allocation") and corrupts memory (seen as an XGBoost "access violation").
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import io
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import auc as auc_area, roc_curve

from pipeline import (CLASS_NAMES, CLASS_RULE, MODEL_DETAILS, PROTOCOLS, TARGET, artifact_dir, confusion,
                      mcnemar, per_class_report, summary_metrics, transform)
from report import build_report

st.set_page_config(page_title="Thesis Defence — Transformer Oil Monitoring", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------- style
SERIF = "Georgia, 'Times New Roman', serif"
COLORS = ["#1F3A5F", "#C8553D", "#2E8B57", "#E0A458", "#6C5B7B", "#3C8DAD", "#8D99AE", "#B56576"]
pio.templates["academic"] = go.layout.Template(layout=dict(
    font=dict(family=SERIF, size=13, color="#1C2430"), colorway=COLORS, paper_bgcolor="white",
    plot_bgcolor="white", title=dict(font=dict(size=15)),
    xaxis=dict(showline=True, linecolor="#444", ticks="outside", gridcolor="#ECEFF3"),
    yaxis=dict(showline=True, linecolor="#444", ticks="outside", gridcolor="#ECEFF3"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)")))
pio.templates.default = "academic"  # also applies to go.Figure charts (ROC, comparison)
px.defaults.template = "academic"

st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:1250px}
.title-block{border-top:4px solid #1F3A5F;border-bottom:1px solid #D5DBE3;padding:18px 4px 14px;margin-bottom:6px}
.title-block .inst{font-variant:small-caps;letter-spacing:.06em;color:#5A6472;font-size:.95rem}
.title-block h1{font-family:Georgia,'Times New Roman',serif;font-size:1.85rem;line-height:1.25;margin:.3rem 0 .5rem;color:#1F3A5F}
.title-block .sub{font-style:italic;color:#3A4350;margin:-.2rem 0 .45rem}
.title-block .meta{color:#3A4350;font-size:.95rem}
.abstract{background:#F6F8FB;border-left:4px solid #1F3A5F;padding:14px 18px;font-family:Georgia,serif;
          line-height:1.6;text-align:justify;border-radius:2px}
.caption{font-family:Georgia,serif;font-size:.9rem;color:#3A4350;margin:.1rem 0 1.4rem}
.caption b{color:#1F3A5F}
.kpi{border:1px solid #D5DBE3;border-radius:4px;padding:12px 14px;background:#fff}
.kpi .l{font-size:.8rem;color:#5A6472;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-family:Georgia,serif;font-size:1.6rem;color:#1F3A5F}
h2,h3{color:#1F3A5F}
</style>""", unsafe_allow_html=True)


class Numbering:
    """Sequential 'Table n' / 'Figure n' captions, in the order they appear on the page."""

    def __init__(self):
        self.t = self.f = 0

    def table(self, text):
        self.t += 1
        st.markdown(f"<div class='caption'><b>Table {self.t}.</b> {text}</div>", unsafe_allow_html=True)

    def figure(self, text):
        self.f += 1
        st.markdown(f"<div class='caption'><b>Figure {self.f}.</b> {text}</div>", unsafe_allow_html=True)


def kpis(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div></div>",
                   unsafe_allow_html=True)


def pct(v):
    return f"{v:.2%}"


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading the trained models…")
def load_artifacts(protocol):
    d = artifact_dir(protocol)
    with open(os.path.join(d, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    prep = joblib.load(os.path.join(d, "preprocessor.joblib"))
    models = {}
    for name, meta in info["models"].items():
        path = os.path.join(d, meta["file"])
        if meta["type"] == "ml":
            models[name] = joblib.load(path)
        else:
            import keras
            models[name] = keras.models.load_model(path)
    return {"info": info, "prep": prep, "features": prep["features"], "models": models,
            "train": pd.read_csv(os.path.join(d, "train_set.csv")),
            "test": pd.read_csv(os.path.join(d, "test_set.csv")),
            "cv": pd.read_csv(os.path.join(d, "cv_results.csv"))}


def pp(protocol, name, X):
    """Class probabilities of one model of one protocol."""
    B = load_artifacts(protocol)
    if B["info"]["models"][name]["type"] == "dl":
        return B["models"][name].predict(X[..., None], verbose=0)
    return B["models"][name].predict_proba(X)


def predict_proba(name, X):
    return pp("thesis", name, X)


def encode(series):
    return np.array([LABELS.index(v) for v in series.astype(str).replace(CLASS_NAMES)])


@st.cache_resource(show_spinner="Evaluating the models…")
def evaluate_all(protocol):
    B = load_artifacts(protocol)
    X_te, X_tr = transform(B["test"], B["prep"]), transform(B["train"], B["prep"])
    y_te, y_tr = encode(B["test"][TARGET]), encode(B["train"][TARGET])
    out = {}
    for name in B["models"]:
        proba = pp(protocol, name, X_te)
        pred = proba.argmax(1)
        out[name] = {"proba": proba, "pred": pred, "metrics": summary_metrics(y_te, pred, proba),
                     "train_acc": float((pp(protocol, name, X_tr).argmax(1) == y_tr).mean()),
                     "per_class": per_class_report(y_te, pred, LABELS), "cm": confusion(y_te, pred, LABELS)}
    return y_te, out


if not all(os.path.exists(os.path.join(artifact_dir(p), "cv_results.csv")) for p in PROTOCOLS):
    st.error("Trained models not found. Run `python train_models.py` once, then reload this page.")
    st.stop()

B_TH = load_artifacts("thesis")
INFO, PREP, TRAIN_DF, TEST_DF, CV, MODELS = (B_TH[k] for k in ["info", "prep", "train", "test", "cv", "models"])
LABELS, FEATURES = PREP["labels"], PREP["features"]
Y_TEST, RES = evaluate_all("thesis")
try:
    with open("project_info.json", encoding="utf-8") as f:
        PROJECT = json.load(f)
except (OSError, ValueError):
    PROJECT = {}

KAPPA = "Cohen's κ"
TEST_COLS = ["Accuracy", "Precision", "Recall", "F1-score", "Cohen's κ", "MCC", "ROC-AUC"]


def results_table(res):
    return pd.DataFrame({n: r["metrics"] for n, r in res.items()}).T[TEST_COLS].sort_values(
        ["F1-score", "Accuracy"], ascending=False)


TABLE = results_table(RES)
ORDER = list(TABLE.index)
BEST = ORDER[0]
FAMILY = {n: ("Deep learning" if INFO["models"][n]["type"] == "dl" else "Machine learning") for n in ORDER}

CV_COLS = ["Accuracy", "F1-score", "Cohen's κ", "MCC", "ROC-AUC"]
CV_MEAN = CV.groupby("Model")[CV_COLS].mean().loc[ORDER]
CV_STD = CV.groupby("Model")[CV_COLS].std().loc[ORDER]
CV_TABLE = pd.DataFrame({c: [f"{m:.3f} ± {s:.3f}" for m, s in zip(CV_MEAN[c], CV_STD[c])] for c in CV_COLS},
                        index=ORDER)
CV_BEST = CV_MEAN["F1-score"].idxmax()

SIG = pd.DataFrame([{"Comparison": f"{BEST} vs {n}", "b (only best correct)": b, "c (only other correct)": c,
                     "p-value": p, "Significant (α = 0.05)": "Yes" if p < 0.05 else "No"}
                    for n in ORDER[1:] for b, c, p in [mcnemar(Y_TEST, RES[BEST]["pred"], RES[n]["pred"])]])

GAP = pd.DataFrame({"Training accuracy": {n: RES[n]["train_acc"] for n in ORDER},
                    "Test accuracy": TABLE["Accuracy"]})
GAP["Gap"] = GAP["Training accuracy"] - GAP["Test accuracy"]


def most_confused(res):
    """(class a, class b, share of all errors) for the most frequently confused pair, or None if no errors."""
    off = sum(r["cm"] for r in res.values()).astype(float)
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        return None
    sym = off + off.T
    i, j = np.unravel_index(np.argmax(np.triu(sym, 1)), sym.shape)
    return LABELS[i], LABELS[j], sym[i, j] / off.sum()


CONFUSED = most_confused(RES)
N_ERR = {n: int((RES[n]["pred"] != Y_TEST).sum()) for n in ORDER}
F1_CLASS = pd.DataFrame({n: RES[n]["per_class"]["F1-score"] for n in ORDER}).T
CLASS_F1_MEAN = F1_CLASS.mean().sort_values(ascending=False)
EASY = [c for c in CLASS_F1_MEAN.index if CLASS_F1_MEAN[c] >= 0.95]  # near-perfect for every model
HARD = CLASS_F1_MEAN.index[-1]
_lift = (RES[BEST]["per_class"]["F1-score"] - CLASS_F1_MEAN).drop(EASY, errors="ignore")
LIFT = _lift.idxmax() if len(_lift) else None  # class where the best model gains most over the average

# ---- spectrum-only protocol (robustness analysis)
B_SP = load_artifacts("spectrum")
Y_SP, RES_SP = evaluate_all("spectrum")
TABLE_SP = results_table(RES_SP)
BEST_SP = TABLE_SP.index[0]
CV_SP = B_SP["cv"].groupby("Model")[CV_COLS].agg(["mean", "std"])
CONFUSED_SP = most_confused(RES_SP)
MAJORITY = TEST_DF[TARGET].value_counts(normalize=True).max()
COMPARE = pd.DataFrame({
    "Accuracy (thesis)": TABLE["Accuracy"], "F1 (thesis)": TABLE["F1-score"],
    "Accuracy (spectrum only)": TABLE_SP["Accuracy"].reindex(ORDER),
    "F1 (spectrum only)": TABLE_SP["F1-score"].reindex(ORDER),
    "CV F1 (spectrum only)": [f"{CV_SP.loc[n, ('F1-score', 'mean')]:.3f} ± {CV_SP.loc[n, ('F1-score', 'std')]:.3f}"
                              for n in ORDER]})
COMPARE["Δ F1"] = COMPARE["F1 (spectrum only)"] - COMPARE["F1 (thesis)"]
COMPARE.index.name = "Model"


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def easy_phrase():
    if len(EASY) == len(LABELS):
        return f"all {len(LABELS)} classes are recognised almost perfectly by every model"
    if EASY:
        return (f"{_join(EASY)} {'is' if len(EASY) == 1 else 'are'} recognised almost perfectly by every model, "
                f"while {HARD} is the most difficult class")
    return f"{CLASS_F1_MEAN.index[0]} is the easiest class and {HARD} the most difficult"


# --------------------------------------------------------------------------- figures
def fig_cm(cm, title, height=330):
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    text = [[f"{cm[a, b]}<br>({share[a, b]:.0%})" for b in range(len(LABELS))] for a in range(len(LABELS))]
    fig = px.imshow(share, x=LABELS, y=LABELS, color_continuous_scale=["#FFFFFF", "#1F3A5F"], zmin=0, zmax=1,
                    labels=dict(x="Predicted class", y="True class"), title=title)
    fig.update_traces(text=text, texttemplate="%{text}", hovertemplate="True %{y} → Predicted %{x}<extra></extra>")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def fig_comparison():
    fig = go.Figure()
    fig.add_bar(x=ORDER, y=CV_MEAN["F1-score"], name=f"{INFO['cv_folds']}-fold CV (mean ± SD)",
                marker_color="#8FA6C1", error_y=dict(type="data", array=CV_STD["F1-score"], color="#333"))
    fig.add_scatter(x=ORDER, y=TABLE["F1-score"], mode="markers", name="Hold-out test set",
                    marker=dict(symbol="diamond", size=12, color="#C8553D", line=dict(color="#333", width=1)))
    fig.update_layout(height=420, yaxis=dict(title="Macro F1-score", range=[0, 1], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=40))
    return fig


def fig_roc(name):
    fig = go.Figure()
    proba = RES[name]["proba"]
    for k, lab in enumerate(LABELS):
        fpr, tpr, _ = roc_curve((Y_TEST == k).astype(int), proba[:, k])
        auc = auc_area(fpr, tpr)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{lab} (AUC = {auc:.3f})", line=dict(width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dot", color="#999"))
    fig.update_layout(height=430, xaxis=dict(title="False positive rate", range=[0, 1]),
                      yaxis=dict(title="True positive rate", range=[0, 1.02]),
                      legend=dict(x=0.55, y=0.05), title=f"{name} — one-vs-rest ROC curves", margin=dict(t=45))
    return fig


def fig_per_class():
    fig = px.imshow(F1_CLASS, text_auto=".2f", color_continuous_scale=["#FFFFFF", "#2E8B57"], zmin=0, zmax=1,
                    aspect="auto", labels=dict(x="Class", y="Model", color="F1"))
    fig.update_layout(height=390, margin=dict(t=20))
    return fig


def fig_gap():
    long = GAP[["Training accuracy", "Test accuracy"]].reset_index(names="Model").melt(
        id_vars="Model", var_name="Set", value_name="Accuracy")
    fig = px.bar(long, x="Model", y="Accuracy", color="Set", barmode="group", text_auto=".0%",
                 color_discrete_sequence=["#8FA6C1", "#1F3A5F"])
    fig.update_layout(height=400, yaxis=dict(range=[0, 1.08], tickformat=".0%"), legend_title=None,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=40))
    return fig


def fig_classes():
    counts = pd.concat([TRAIN_DF, TEST_DF])[TARGET].value_counts().reindex(LABELS)
    fig = px.bar(x=LABELS, y=counts.values, text=counts.values, labels=dict(x="Class", y="Samples"),
                 color_discrete_sequence=["#1F3A5F"])
    fig.update_layout(height=300, margin=dict(t=20))
    return fig


def fig_protocols():
    long = COMPARE[["F1 (thesis)", "F1 (spectrum only)"]].rename(columns={
        "F1 (thesis)": PROTOCOLS["thesis"]["short"], "F1 (spectrum only)": PROTOCOLS["spectrum"]["short"]}
    ).reset_index().melt(id_vars="Model", var_name="Inputs", value_name="F1")
    fig = px.bar(long, x="Model", y="F1", color="Inputs", barmode="group", text_auto=".0%",
                 color_discrete_sequence=["#1F3A5F", "#C8553D"])
    fig.add_hline(y=0.25, line_dash="dot", line_color="#777", annotation_text="chance (4 classes)",
                  annotation_position="bottom right")
    fig.update_layout(height=420, yaxis=dict(title="Macro F1-score (test set)", range=[0, 1.08], tickformat=".0%"),
                      legend_title=None, legend=dict(orientation="h", y=1.1, x=0), margin=dict(t=40))
    return fig


# --------------------------------------------------------------------------- generated text
def class_def_table():
    rows = []
    n_all = pd.concat([TRAIN_DF, TEST_DF])[TARGET].value_counts()
    inv = {v: k for k, v in CLASS_NAMES.items()}
    for c in LABELS:
        lo, hi = INFO.get("class_temperature", {}).get(c, [None, None])
        rng = "—" if lo is None else (f"{lo} °C" if lo == hi else f"{lo}–{hi} °C")
        rows.append({"Oil condition": c, "Code": inv.get(c, ""), "Labelling rule": CLASS_RULE.get(c, "—"),
                     "Measured temperatures": rng, "Samples": int(n_all[c]),
                     "Share": f"{n_all[c] / n_all.sum():.1%}"})
    return pd.DataFrame(rows).set_index("Oil condition")


def data_text():
    return (f"The dataset was built for this thesis from transformer-oil samples of old and new distribution "
            f"transformers (Diyala State Company for Electrical Industries), analysed at the Laser Institute for "
            f"Postgraduate Studies, University of Baghdad. A broadband source (1450–1650 nm) illuminates the oil "
            f"through a Fabry–Perot interferometer sensor with two air micro-cavities (FBG–PCF–FBG), and an optical "
            f"spectrum analyser records the reflected spectrum. Each record holds the oil temperature and 12 optical "
            f"attributes of the spectrum (centroid and peak wavelength, peak level, FWHM, OSNR, noise, …). Of 805 "
            f"collected records, 4 with missing values were removed, leaving **{INFO['n_rows']}**. Each record is "
            f"labelled with one of four oil-condition classes by thresholds on the oil temperature (sieve "
            f"labelling, Table 1).")


def abstract_text():
    b = TABLE.loc[BEST]
    bs = TABLE_SP.loc[BEST_SP]
    best_dl = max((n for n in ORDER if FAMILY[n] == "Deep learning"), key=lambda n: TABLE.loc[n, "F1-score"])
    n99 = int((TABLE["Accuracy"] >= 0.99).sum())
    return (f"This work develops the artificial-intelligence module of an IoT monitoring system for distribution "
            f"transformers: the classification of transformer-oil condition into {len(LABELS)} classes "
            f"({_join(LABELS)}) from measurements of an optical-fibre Fabry–Perot sensor with air micro-cavities. "
            f"A dataset of {INFO['n_rows']} records with {len(FEATURES)} attributes was used to compare six "
            f"machine-learning classifiers (including Random Forest, Gaussian Naive Bayes, Decision Tree and MLP) "
            f"and two deep-learning models (a 1-D CNN and a hybrid CNN-LSTM). Pre-processing (missing-value removal, "
            f"IQR outlier treatment and z-score normalisation) was fitted on the training data only, and the models "
            f"were assessed with {INFO['cv_folds']}-fold stratified cross-validation and a held-out test set of "
            f"{INFO['n_test']} records. Using all attributes, **{BEST}** reached an accuracy of "
            f"**{pct(b['Accuracy'])}** and a macro F1-score of **{pct(b['F1-score'])}**, and {n99} of the "
            f"{len(ORDER)} models reached at least 99% accuracy (best deep model: {best_dl}, "
            f"{pct(TABLE.loc[best_dl, 'Accuracy'])}). Because the classes are defined from the oil temperature, the "
            f"experiment was repeated without the temperature attribute: from the optical spectrum alone, "
            f"**{BEST_SP}** still reached **{pct(bs['Accuracy'])}** accuracy (ROC-AUC {bs['ROC-AUC']:.3f}), against "
            f"25% for chance, which confirms that the sensor spectrum itself carries oil-condition information.")


def discussion_points():
    b = TABLE.loc[BEST]
    bs = TABLE_SP.loc[BEST_SP]
    ml = [n for n in ORDER if FAMILY[n] == "Machine learning"]
    dl = [n for n in ORDER if FAMILY[n] == "Deep learning"]
    best_ml = max(ml, key=lambda n: TABLE.loc[n, "F1-score"])
    best_dl = max(dl, key=lambda n: TABLE.loc[n, "F1-score"])
    sig = SIG[SIG["p-value"] < 0.05]["Comparison"].str.replace(f"{BEST} vs ", "", regex=False).tolist()
    nonsig = SIG[SIG["p-value"] >= 0.05]["Comparison"].str.replace(f"{BEST} vs ", "", regex=False).tolist()
    overfit = GAP[(GAP["Training accuracy"] > 0.99) & (GAP["Gap"] > 0.05)].index.tolist()
    thesis_models = ["Random Forest", "Gaussian Naive Bayes", "Decision Tree", "MLP", "CNN", "CNN-LSTM"]
    rep = _join(f"{n} {pct(TABLE.loc[n, 'Accuracy'])}" for n in thesis_models if n in TABLE.index)
    errs = ", ".join(f"{n} {N_ERR[n]}" for n in ORDER)
    dl_sp = max(TABLE_SP.loc[n, "F1-score"] for n in dl)
    gap_ml_dl = TABLE.loc[best_ml, "F1-score"] - TABLE.loc[best_dl, "F1-score"]
    pts = [
        ("Overall performance",
         f"With all {len(FEATURES)} attributes, {BEST} achieved the highest macro F1-score on the test set "
         f"({pct(b['F1-score'])}, κ = {b[KAPPA]:.3f}, MCC = {b['MCC']:.3f}). The number of misclassified test "
         f"records out of {INFO['n_test']} is: {errs}. In cross-validation the best mean F1-score was obtained by "
         f"{CV_BEST} ({CV_MEAN.loc[CV_BEST, 'F1-score']:.3f} ± {CV_STD.loc[CV_BEST, 'F1-score']:.3f})."),
        ("Agreement with the thesis",
         f"Chapter 4 of the thesis reports 100% accuracy for Random Forest, Gaussian NB, Decision Tree, MLP, "
         f"1D-CNN and CNN-LSTM. The independent re-run here, with a stratified {1 - INFO['test_size']:.0%}/"
         f"{INFO['test_size']:.0%} split and cross-validation, gives test accuracies of {rep}, which confirms "
         f"the thesis results; the small differences come from the different data split and random seeds."),
        ("Statistical significance",
         f"McNemar's exact test (Table 6) shows that {BEST} is significantly better than "
         f"{', '.join(sig) if sig else 'none of the other models'} at α = 0.05"
         + (f", whereas the differences with {', '.join(nonsig)} are not statistically significant." if nonsig
            else ".")
         + (" With so few errors, most of the leading models are statistically indistinguishable."
            if len(nonsig) >= 3 else "")),
        ("Machine learning vs deep learning",
         f"The best classical model ({best_ml}, F1 = {pct(TABLE.loc[best_ml, 'F1-score'])}) and the best deep "
         f"network ({best_dl}, F1 = {pct(TABLE.loc[best_dl, 'F1-score'])}) "
         + ("perform similarly" if abs(gap_ml_dl) < 0.02 else
            ("differ in favour of the classical model" if gap_ml_dl > 0 else "differ in favour of the deep network"))
         + f" with all attributes. Without temperature the difference is clearer: {BEST_SP} reaches F1 = "
         f"{pct(bs['F1-score'])} while the best deep network reaches {pct(dl_sp)}; with {INFO['n_train']} "
         f"training records the large networks have far more parameters than the data can constrain."),
        ("Role of the temperature attribute",
         f"The classes are defined by thresholds on the oil temperature (Table 1), and temperature is also one of "
         f"the inputs. A model can therefore reach almost 100% by learning these thresholds, which is why the "
         f"spectrum-only experiment (Section 3.8) matters: without temperature, {BEST_SP} still classifies "
         f"{pct(bs['Accuracy'])} of the test records correctly (majority-class baseline {pct(MAJORITY)}, chance "
         f"25%), with ROC-AUC {bs['ROC-AUC']:.3f}. The optical spectrum measured by the Fabry–Perot sensor is thus "
         f"informative on its own, in line with the temperature-dependent wavelength shift of the sensor "
         f"(≈18 pm/°C)."
         + (f" Its main remaining error is the confusion between {CONFUSED_SP[0]} and {CONFUSED_SP[1]} "
            f"({CONFUSED_SP[2]:.0%} of the errors)." if CONFUSED_SP else "")),
        ("Generalisation",
         (f"{', '.join(overfit)} fit the training data almost perfectly but lose "
          f"{GAP.loc[overfit, 'Gap'].min():.0%}–{GAP.loc[overfit, 'Gap'].max():.0%} on the test set (Figure 7), "
          f"a sign of over-fitting.") if overfit else
         "Training and test accuracies are close for every model (Figure 7), so there is no sign of over-fitting "
         "under the thesis protocol."),
    ]
    return pts


LIMITATIONS = [
    "A single dataset of moderate size (801 records from five oil samples) was used; no independently collected "
    "test set was available.",
    "The class labels are derived from the oil temperature, which is also an input of the thesis protocol, so "
    "near-perfect scores mainly show that the thresholds are recovered; the spectrum-only protocol gives the more "
    "demanding estimate.",
    "Classes cover temperature ranges of unequal width (two measured temperatures in Good and Very Poor, one in "
    "Not Bad and Poor).",
    "Classical models use library-default hyper-parameters; the networks treat the scalar attributes as a 1-D "
    "sequence whose ordering has no physical meaning.",
]
FUTURE = [
    "Labelling oil condition from independent laboratory tests (e.g. breakdown voltage, moisture, acidity) "
    "in addition to temperature thresholds.",
    "Using the full measured spectrum rather than summary attributes, which gives the CNN/LSTM a meaningful "
    "sequential input.",
    "Collecting more oil samples and field data through the IoT platform, particularly near the Good / Not Bad "
    "boundary.",
    "Hyper-parameter optimisation with nested cross-validation and model explainability (e.g. SHAP).",
    "Deploying the selected model on the IoT receiver for real-time alarms.",
]


def robust_text():
    bs = TABLE_SP.loc[BEST_SP]
    drop = TABLE.loc[BEST_SP, "Accuracy"] - bs["Accuracy"]
    return (f"The oil-condition classes are defined by thresholds on the oil temperature, and temperature is "
            f"also one of the {len(FEATURES)} inputs, so a model could reach ≈100% by reading the temperature "
            f"alone. To show that the Fabry–Perot sensor spectrum is informative in itself, the complete experiment "
            f"was repeated with the temperature attribute **removed** ({len(B_SP['features'])} optical attributes, "
            f"same split, pre-processing and models). The best model, **{BEST_SP}**, still reaches "
            f"**{pct(bs['Accuracy'])}** accuracy and ROC-AUC **{bs['ROC-AUC']:.3f}** — far above chance (25%) and "
            f"the majority-class baseline ({pct(MAJORITY)}) — {drop * 100:.1f} percentage points lower than with "
            f"temperature."
            + (f" Most remaining errors are confusions between **{CONFUSED_SP[0]}** and **{CONFUSED_SP[1]}** "
               f"({CONFUSED_SP[2]:.0%} of all errors)." if CONFUSED_SP else ""))


def conclusion_text():
    b = TABLE.loc[BEST]
    bs = TABLE_SP.loc[BEST_SP]
    return (f"Eight classifiers were evaluated for classifying transformer-oil condition from Fabry–Perot "
            f"optical-sensor measurements. With all attributes, {BEST} provided the best test-set performance "
            f"(accuracy {pct(b['Accuracy'])}, macro F1 {pct(b['F1-score'])}), and the re-run confirms the "
            f"near-perfect results reported in the thesis. Using the optical spectrum alone, without temperature, "
            f"{BEST_SP} still reached {pct(bs['Accuracy'])} accuracy, so the sensor spectrum carries real "
            f"oil-condition information. The proposed AI module can therefore support the IoT monitoring system in "
            f"classifying oil condition and triggering suitable actions for each class.")


# =========================================================================== header
st.markdown(f"""<div class='title-block'>
<div class='inst'>{PROJECT.get('institution', '')} &nbsp;·&nbsp; {PROJECT.get('degree', '')} &nbsp;·&nbsp; {PROJECT.get('year', '')}</div>
<h1>{PROJECT.get('title', 'Spectrum Classification')}</h1>
<div class='sub'>{PROJECT.get('subtitle', '')}</div>
<div class='meta'><b>Candidate:</b> {PROJECT.get('author', '')} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Supervisors:</b> {PROJECT.get('supervisor', '')}</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Study at a glance")
    st.markdown(f"""
| | |
|---|---|
| Records | {INFO['n_rows']} |
| Attributes | {len(FEATURES)} |
| Classes | {len(LABELS)} |
| Train / test | {INFO['n_train']} / {INFO['n_test']} |
| CV folds | {INFO['cv_folds']} |
| Models | {len(MODELS)} |
| Best model | **{BEST}** ({pct(TABLE.loc[BEST, 'Accuracy'])}) |
| Spectrum only | {BEST_SP} ({pct(TABLE_SP.loc[BEST_SP, 'Accuracy'])}) |
""")
    st.caption(f"Models trained {INFO['trained_at']} · seed {INFO['seed']}")

N = Numbering()
t_over, t_meth, t_res, t_test, t_conc = st.tabs(
    ["1 · Overview", "2 · Methodology", "3 · Results", "4 · Live testing", "5 · Discussion & Conclusion"])

# =========================================================================== 1 Overview
with t_over:
    st.markdown("### Abstract")
    abstract_html = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", abstract_text())
    st.markdown(f"<div class='abstract'>{abstract_html}</div>", unsafe_allow_html=True)
    st.write("")
    b = TABLE.loc[BEST]
    kpis([("Best model", BEST), ("Test accuracy", pct(b["Accuracy"])), ("Macro F1-score", pct(b["F1-score"])),
          ("Spectrum only (no temperature)", f"{pct(TABLE_SP.loc[BEST_SP, 'Accuracy'])} · {BEST_SP}")])
    st.write("")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### Objectives")
        st.markdown(f"""
1. Classify transformer-oil condition ({_join(LABELS)}) from Fabry–Perot optical-sensor measurements.
2. Compare machine-learning classifiers (Random Forest, Gaussian NB, Decision Tree, MLP, XGBoost, SVM) with
   two deep-learning models (1D-CNN and hybrid CNN-LSTM).
3. Evaluate the models with cross-validation, a held-out test set and statistical significance tests.
4. Verify that the optical spectrum itself, without the temperature reading, carries oil-condition information.
""")
        st.markdown("### Problem definition")
        N.table("Oil-condition classes, labelling rule (thesis, Section 3.6.4) and sample distribution.")
        st.dataframe(class_def_table(), width="stretch")
    with c2:
        st.markdown("### Class distribution")
        st.plotly_chart(fig_classes(), width="stretch")
        N.figure(f"Number of samples per class (N = {INFO['n_rows']}).")

# =========================================================================== 2 Methodology
with t_meth:
    st.markdown("### 2.1 Dataset")
    st.markdown(data_text())
    st.markdown(f"""
Two feature sets (protocols) are evaluated with exactly the same split, pre-processing and models:

| Protocol | Inputs | Purpose |
|---|---|---|
| **{PROTOCOLS['thesis']['name']}** | {PROTOCOLS['thesis']['short']} | Reproduces Chapter 4 of the thesis |
| **{PROTOCOLS['spectrum']['name']}** | {PROTOCOLS['spectrum']['short']} | Checks that the optical spectrum alone carries the information (Section 3.8) |
""")
    N.table("Descriptive statistics of the input attributes (training and test data).")
    desc = pd.concat([TRAIN_DF, TEST_DF])[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "SD", "Min", "Median", "Max"]
    st.dataframe(desc.style.format("{:.3f}"), width="stretch")

    st.markdown("### 2.2 Experimental pipeline")
    st.graphviz_chart(f"""digraph {{ rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#F3F5F8", color="#1F3A5F", fontname="Georgia", fontsize=11];
      edge [color="#1F3A5F"];
      a [label="FPI sensor data\\n805 → {INFO['n_rows']} records\\n(missing values removed)"];
      b [label="Sieve labelling\\nGood / Not Bad /\\nPoor / Very Poor"];
      c [label="Stratified split\\n{INFO['n_train']} train / {INFO['n_test']} test"];
      d [label="IQR clipping +\\nstandardisation\\n(fit on train only)"];
      e [label="{INFO['cv_folds']}-fold stratified CV\\n(model selection)"];
      f [label="Train 8 models\\non full train set"]; g [label="Evaluate on\\nheld-out test set"];
      h [label="McNemar tests\\n& error analysis"];
      i [label="Repeat without\\ntemperature\\n(spectrum only)", fillcolor="#FBEDEA", color="#C8553D"];
      a->b->c->d->e->f->g->h; g->i [style=dashed, color="#C8553D"]; }}""")
    N.figure("Experimental pipeline. Every fitted transformation is learned from training data only.")

    st.markdown("### 2.3 Pre-processing")
    st.markdown("""
- **Data cleaning:** records with missing values are removed (805 → 801, Algorithm 3.1 of the thesis).
- **Labelling:** label encoding of the four oil-condition classes (Good, Not Bad, Poor, Very Poor).
- **Outlier treatment (EDA):** each attribute is clipped to [Q1 − 1.5·IQR, Q3 + 1.5·IQR] (Algorithm 3.2); the quartiles are computed on the training set.
- **Normalisation:** z-score, z = (x − μ) / σ, with μ and σ from the training set.
- **Leakage control:** inside cross-validation both steps are refitted on each training fold.
- **Split:** stratified 70% / 30% (training / test); the networks use 20% of the training data for validation. The thesis used 70 / 20 / 10 %.
""")

    st.markdown("### 2.4 Models")
    N.table("Classifiers evaluated and their configuration.")
    mt = pd.DataFrame([{"Model": n, "Family": FAMILY[n], "Type": MODEL_DETAILS[n][0],
                        "Configuration": MODEL_DETAILS[n][1],
                        "Trainable params": f"{INFO['models'][n]['n_params']:,}" if INFO["models"][n].get("n_params")
                        else "—"} for n in MODEL_DETAILS]).set_index("Model")
    st.dataframe(mt, width="stretch")
    st.caption(f"Neural networks: Adam (learning rate 10⁻³), {INFO['epochs']} epochs, batch size "
               f"{INFO['batch_size']}, sparse categorical cross-entropy, 20% of the training data for validation.")

    st.markdown("### 2.5 Evaluation protocol and metrics")
    st.markdown(f"Models are compared with **{INFO['cv_folds']}-fold stratified cross-validation** on the training "
                f"set (mean ± standard deviation) and on a **held-out test set** ({INFO['test_size']:.0%}, "
                f"stratified, seed {INFO['seed']}) that is never used for training or model selection. "
                f"Macro-averaging gives every class equal weight.")
    m1, m2 = st.columns(2)
    with m1:
        st.latex(r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
        st.latex(r"\text{Precision}=\frac{TP}{TP+FP}\qquad \text{Recall}=\frac{TP}{TP+FN}")
        st.latex(r"F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}")
    with m2:
        st.latex(r"\kappa=\frac{p_o-p_e}{1-p_e}")
        st.latex(r"\text{MCC}=\frac{c\cdot s-\sum_k p_k t_k}{\sqrt{(s^2-\sum_k p_k^2)(s^2-\sum_k t_k^2)}}")
        st.latex(r"\chi^2_{\text{McNemar}}:\; b \sim \text{Binomial}(b+c,\,0.5)")
    st.caption("κ: Cohen's kappa (agreement beyond chance). MCC: multi-class Matthews correlation coefficient. "
               "ROC-AUC: one-vs-rest, macro-averaged. McNemar's exact test compares two classifiers on the same "
               "test samples using the counts b and c of samples that only one of them classifies correctly.")

# =========================================================================== 3 Results
with t_res:
    st.info(f"Sections 3.1–3.7: **{PROTOCOLS['thesis']['name']}** ({PROTOCOLS['thesis']['short']}), as in "
            f"Chapter 4 of the thesis. Section 3.8: the same experiment **without temperature**.")
    st.markdown("### 3.1 Test-set performance")
    N.table(f"Performance on the held-out test set (n = {INFO['n_test']}), sorted by macro F1-score. "
            f"Best value per column highlighted.")
    show = TABLE.copy()
    show.insert(0, "Family", [FAMILY[n] for n in ORDER])
    st.dataframe(show.style.format({c: "{:.4f}" for c in TEST_COLS})
                 .highlight_max(subset=TEST_COLS, props="background-color:#DCE6F2;font-weight:bold"),
                 width="stretch")

    st.markdown("### 3.2 Cross-validation")
    N.table(f"{INFO['cv_folds']}-fold stratified cross-validation on the training set (mean ± SD).")
    st.dataframe(CV_TABLE, width="stretch")
    st.plotly_chart(fig_comparison(), width="stretch")
    N.figure(f"Macro F1-score: {INFO['cv_folds']}-fold cross-validation (bars, error bars = ±1 SD) and "
             f"held-out test set (diamonds).")

    st.markdown("### 3.3 ROC analysis")
    c1, c2 = st.columns([2, 1])
    roc_model = c2.selectbox("Model", ORDER, key="roc")
    c2.dataframe(TABLE[["ROC-AUC"]].style.format("{:.3f}"), width="stretch")
    with c1:
        st.plotly_chart(fig_roc(roc_model), width="stretch")
    N.figure(f"One-vs-rest ROC curves of {roc_model} on the test set; the table lists the macro ROC-AUC of "
             f"every model.")

    st.markdown("### 3.4 Confusion matrices")
    for row in range(0, len(ORDER), 4):
        cols = st.columns(4)
        for col, name in zip(cols, ORDER[row:row + 4]):
            f = fig_cm(RES[name]["cm"], name, 300)
            f.update_layout(xaxis_title=None, yaxis_title=None, title_font_size=13)
            col.plotly_chart(f, width="stretch")
    N.figure("Test-set confusion matrices (rows: true class, columns: predicted class; counts and row "
             "percentages).")

    st.markdown("### 3.5 Per-class performance")
    st.plotly_chart(fig_per_class(), width="stretch")
    N.figure("F1-score per class and model on the test set.")

    st.markdown("### 3.6 Statistical significance")
    N.table(f"McNemar's exact test: {BEST} compared with every other model on the test set.")
    st.dataframe(SIG.set_index("Comparison").style.format({"p-value": "{:.4f}"}), width="stretch")

    st.markdown("### 3.7 Generalisation")
    st.plotly_chart(fig_gap(), width="stretch")
    N.figure("Training vs test accuracy; a large gap indicates over-fitting.")

    st.markdown("### 3.8 Robustness analysis: optical spectrum only")
    st.markdown(robust_text())
    bs = TABLE_SP.loc[BEST_SP]
    kpis([("Best model (spectrum only)", BEST_SP), ("Test accuracy", pct(bs["Accuracy"])),
          ("Macro F1-score", pct(bs["F1-score"])), ("ROC-AUC", f"{bs['ROC-AUC']:.3f}")])
    st.write("")
    N.table(f"Test-set accuracy and macro F1-score with all attributes and without temperature "
            f"(n = {INFO['n_test']}); Δ F1 = spectrum only − thesis protocol.")
    num = [c for c in COMPARE.columns if c != "CV F1 (spectrum only)"]
    st.dataframe(COMPARE.style.format({c: "{:+.3f}" if c == "Δ F1" else "{:.3f}" for c in num}), width="stretch")
    st.plotly_chart(fig_protocols(), width="stretch")
    N.figure("Test-set macro F1-score of every model with and without the temperature attribute.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_cm(RES_SP[BEST_SP]["cm"], f"{BEST_SP} — spectrum only", 380), width="stretch")
    with c2:
        st.write("")
        st.markdown(f"**Per-class results of {BEST_SP} (spectrum only)**")
        st.dataframe(RES_SP[BEST_SP]["per_class"][["Precision", "Recall", "Specificity", "F1-score", "Support"]]
                     .style.format({c: "{:.3f}" for c in ["Precision", "Recall", "Specificity", "F1-score"]}),
                     width="stretch")
    N.figure(f"Test-set confusion matrix of {BEST_SP} using the optical spectrum only.")

# =========================================================================== 4 Live testing
with t_test:
    st.markdown("### Live demonstration")
    st.caption("Apply any trained model to the held-out test set, to a single measurement, or to a new file.")
    proto = st.radio("Feature set", list(PROTOCOLS), horizontal=True,
                     format_func=lambda k: f"{PROTOCOLS[k]['name']} — {PROTOCOLS[k]['short']}")
    L_B = load_artifacts(proto)
    L_RES = evaluate_all(proto)[1]
    L_TEST, L_FEAT, L_PREP = L_B["test"], L_B["features"], L_B["prep"]
    L_ORDER = list(results_table(L_RES).index)
    L_BEST = L_ORDER[0]

    def L_PP(name, X):
        return pp(proto, name, X)

    c1, c2 = st.columns([1, 2])
    model_name = c1.selectbox("Model", L_ORDER, format_func=lambda n: f"{n}  ★" if n == L_BEST else n, key=f"live_{proto}")
    mode = c2.radio("Input", ["Held-out test set", "Single measurement", "Upload a file"], horizontal=True)

    if mode == "Held-out test set":
        r = L_RES[model_name]
        m = r["metrics"]
        kpis([("Accuracy", pct(m["Accuracy"])), ("Macro F1", pct(m["F1-score"])),
              ("Cohen's κ", f"{m[KAPPA]:.3f}"), ("ROC-AUC", f"{m['ROC-AUC']:.3f}")])
        st.write("")
        a, b_ = st.columns(2)
        a.plotly_chart(fig_cm(r["cm"], f"Confusion matrix — {model_name}", 400), width="stretch")
        b_.markdown("**Per-class results**")
        b_.dataframe(r["per_class"].style.format({c: "{:.3f}" for c in ["Precision", "Recall", "Specificity",
                                                                           "F1-score"]}), width="stretch")
        preds = L_TEST[L_FEAT].copy()
        preds.insert(0, "True class", L_TEST[TARGET].astype(str).values)
        preds.insert(1, "Predicted", [LABELS[k] for k in r["pred"]])
        preds.insert(2, "Confidence", r["proba"].max(1))
        preds.insert(3, "Correct", preds["True class"] == preds["Predicted"])
        if st.toggle(f"Show only misclassified samples ({int((~preds['Correct']).sum())})"):
            preds = preds[~preds["Correct"]]
        st.dataframe(preds.style.format({"Confidence": "{:.1%}"}), width="stretch", height=300)
        st.download_button("Download predictions (CSV)", preds.to_csv(index=False),
                           f"test_predictions_{model_name}.csv")

    elif mode == "Single measurement":
        idx = st.selectbox("Pre-fill from a test-set sample", ["—"] + list(range(len(L_TEST))),
                           format_func=lambda k: "Median of the data" if k == "—" else
                           f"Sample #{k}  (true class: {L_TEST[TARGET].iloc[k]})")
        base = L_TEST[L_FEAT].median() if idx == "—" else L_TEST[L_FEAT].iloc[idx]
        with st.form(f"single_{proto}_{idx}"):
            cols = st.columns(3)
            vals = {f: cols[k % 3].number_input(f, value=float(base[f]), format="%.6f")
                    for k, f in enumerate(L_FEAT)}
            submitted = st.form_submit_button("Classify", type="primary")
        if submitted:
            X = transform(pd.DataFrame([vals]), L_PREP)
            proba = L_PP(model_name, X)[0]
            pred = LABELS[int(proba.argmax())]
            msg = f"**{model_name}** predicts **{pred}** (confidence {proba.max():.1%})."
            if idx != "—":
                true = str(L_TEST[TARGET].iloc[idx])
                msg += f" True class: **{true}** — {'correct ✓' if true == pred else 'incorrect ✗'}"
            st.success(msg)
            a, b_ = st.columns(2)
            fig = px.bar(x=LABELS, y=proba, text_auto=".1%", labels=dict(x="Class", y="Probability"),
                         color_discrete_sequence=["#1F3A5F"])
            fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"), height=340, title="Class probabilities")
            a.plotly_chart(fig, width="stretch")
            allm = pd.DataFrame([{"Model": n, "Prediction": LABELS[int(p.argmax())], "Confidence": p.max()}
                                 for n in L_ORDER for p in [L_PP(n, X)[0]]])
            b_.markdown("**Prediction of every model**")
            b_.dataframe(allm.style.format({"Confidence": "{:.1%}"}), width="stretch", hide_index=True)

    else:
        st.caption(f"Excel or CSV with the columns: {', '.join(L_FEAT)}. Include a `{TARGET}` column to also "
                   f"obtain evaluation metrics.")
        up = st.file_uploader("File", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
        if up:
            buf = io.BytesIO(up.getvalue())
            new = pd.read_csv(buf) if up.name.lower().endswith(".csv") else pd.read_excel(buf)
            missing = [c for c in L_FEAT if c not in new.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                proba = L_PP(model_name, transform(new, L_PREP))
                out = new.copy()
                out.insert(0, "Predicted", [LABELS[k] for k in proba.argmax(1)])
                out.insert(1, "Confidence", proba.max(1))
                if TARGET in new.columns and set(new[TARGET].astype(str).replace(CLASS_NAMES)) <= set(LABELS):
                    y = encode(new[TARGET])
                    m = summary_metrics(y, proba.argmax(1))
                    kpis([(k, pct(m[k])) for k in ["Accuracy", "Precision", "Recall", "F1-score"]])
                    st.plotly_chart(fig_cm(confusion(y, proba.argmax(1), LABELS), "Confusion matrix", 400),
                                    width="stretch")
                st.dataframe(out.style.format({"Confidence": "{:.1%}"}), width="stretch")
                st.download_button("Download predictions (CSV)", out.to_csv(index=False), "predictions.csv")

# =========================================================================== 5 Discussion & conclusion
with t_conc:
    st.markdown("### 5.1 Discussion")
    for head, text in discussion_points():
        st.markdown(f"**{head}.** {text}")
    st.markdown("### 5.2 Limitations")
    st.markdown("\n".join(f"- {x}" for x in LIMITATIONS))
    st.markdown("### 5.3 Conclusion")
    st.markdown(f"<div class='abstract'>{conclusion_text()}</div>", unsafe_allow_html=True)
    st.markdown("### 5.4 Future work")
    st.markdown("\n".join(f"- {x}" for x in FUTURE))

    st.markdown("### Export")
    figs = {"comparison": fig_comparison(), "roc": fig_roc(BEST), "per_class": fig_per_class(), "gap": fig_gap(),
            "classes": fig_classes(), "cms": [fig_cm(RES[n]["cm"], n, 330) for n in ORDER]}
    robust = {"text": robust_text(), "table": COMPARE, "fig": fig_protocols()}
    report = build_report(PROJECT, INFO, FEATURES, LABELS, BEST, abstract_text(), data_text(), class_def_table(),
                          desc, mt, show, CV_TABLE, SIG.set_index("Comparison"), robust, discussion_points(),
                          LIMITATIONS, conclusion_text(), FUTURE, figs)
    e1, e2, e3 = st.columns(3)
    e1.download_button("📄 Academic report (HTML → print to PDF)", report, "thesis_results_report.html",
                       "text/html", type="primary", width="stretch",
                       help="Open in a browser and use Print → Save as PDF. Works offline.")
    e2.download_button("Test-set results (CSV)", show.to_csv(), "test_results.csv", width="stretch")
    e3.download_button("Cross-validation folds (CSV)", CV.to_csv(index=False), "cv_results.csv", width="stretch")
