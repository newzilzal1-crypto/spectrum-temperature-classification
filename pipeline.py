"""Shared preprocessing, model definitions and metrics for train_models.py and app.py."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, multilabel_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join("Data", "Data.xlsx")
ARTIFACT_DIR = "saved_models"
TARGET = "Class"
# 'temperature' maps one-to-one onto Class (0-20 -> Class1, 40 -> Class2, 60 -> Class3, 80-100 -> Class4),
# so it is excluded: otherwise every model scores ~100% without using the spectrum.
DROP_FEATURES = ["temperature"]
TEST_SIZE = 0.30
SEED = 42

ML_MODELS = {
    "XGBoost": lambda: XGBClassifier(random_state=SEED, eval_metric="mlogloss"),
    "Random Forest": lambda: RandomForestClassifier(random_state=SEED),
    "Gaussian Naive Bayes": lambda: GaussianNB(),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED),
    "SVM": lambda: SVC(probability=True, random_state=SEED),
    "MLP": lambda: MLPClassifier(max_iter=1000, random_state=SEED),
}
DL_MODELS = ["CNN", "CNN-LSTM"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


# --------------------------------------------------------------------------- preprocessing
def fit_preprocessor(X_train: pd.DataFrame) -> dict:
    """IQR clipping bounds + StandardScaler, learned from the training split only."""
    q1, q3 = X_train.quantile(0.25), X_train.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    scaler = StandardScaler().fit(X_train.clip(lower, upper, axis=1))
    return {"features": list(X_train.columns), "lower": lower, "upper": upper, "scaler": scaler}


def transform(X: pd.DataFrame, prep: dict) -> np.ndarray:
    X = X[prep["features"]].astype(float).clip(prep["lower"], prep["upper"], axis=1)
    return prep["scaler"].transform(X)


def load_and_split(path=DATA_PATH):
    df = pd.read_excel(path)
    features = [c for c in df.columns if c != TARGET and c not in DROP_FEATURES]
    le = LabelEncoder().fit(df[TARGET].astype(str))
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    return df, train_df, test_df, features, le


# --------------------------------------------------------------------------- deep learning
def build_dl_model(kind, n_features, n_classes, lr=1e-3):
    from keras import Input, Sequential
    from keras.layers import (LSTM, Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten,
                              MaxPooling1D)
    from keras.optimizers import Adam

    m = Sequential([Input(shape=(n_features, 1)), Conv1D(512, 1), Activation("relu"), MaxPooling1D(2),
                    Conv1D(256, 1), Activation("relu"), MaxPooling1D(2)])
    if kind == "CNN":
        m.add(Dropout(0.25))
        m.add(BatchNormalization())
        m.add(Flatten())
    else:
        m.add(LSTM(128))
        m.add(Activation("relu"))
        m.add(BatchNormalization())
        m.add(Dropout(0.15))
    m.add(Dense(1024, activation="relu"))
    m.add(Dense(n_classes, activation="softmax"))
    m.compile(optimizer=Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# Human-readable description of every model, used in the methodology section.
MODEL_DETAILS = {
    "XGBoost": ("Gradient-boosted trees", "100 trees, max depth 6, learning rate 0.3 (library defaults)"),
    "Random Forest": ("Bagged decision trees", "100 trees, Gini impurity, √p features per split"),
    "Gaussian Naive Bayes": ("Probabilistic", "Gaussian class-conditional likelihoods, independent features"),
    "Decision Tree": ("Single tree", "Gini impurity, grown until leaves are pure"),
    "SVM": ("Kernel method", "RBF kernel, C = 1, γ = 'scale', Platt-scaled probabilities"),
    "MLP": ("Feed-forward neural network", "1 hidden layer of 100 ReLU units, Adam, ≤1000 iterations"),
    "CNN": ("1-D convolutional network",
            "Conv1D(512)→MaxPool→Conv1D(256)→MaxPool→Dropout 0.25→BatchNorm→Dense(1024)→Softmax"),
    "CNN-LSTM": ("Convolutional-recurrent network",
                 "Conv1D(512)→MaxPool→Conv1D(256)→MaxPool→LSTM(128)→BatchNorm→Dropout 0.15→Dense(1024)→Softmax"),
}
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-score", "Cohen's κ", "MCC"]


# --------------------------------------------------------------------------- metrics
def summary_metrics(y_true, y_pred, proba=None) -> dict:
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Cohen's κ": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if proba is not None:
        out["ROC-AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    return out


def mcnemar(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on the discordant pairs. Returns (b, c, p-value)."""
    from scipy.stats import binomtest
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    rows = {}
    for name, m in zip(labels, mcm):
        tn, fp, fn, tp = (int(v) for v in m.ravel())
        rec = tp / (tp + fn) if tp + fn else 0.0
        spe = tn / (tn + fp) if tn + fp else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        rows[name] = {"Precision": pre, "Recall": rec, "Specificity": spe,
                      "F1-score": 2 * pre * rec / (pre + rec) if pre + rec else 0.0,
                      "Support": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return pd.DataFrame(rows).T


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=range(len(labels)))
