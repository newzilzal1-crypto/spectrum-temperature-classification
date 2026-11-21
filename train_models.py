"""Train every model once and save everything the Streamlit app needs into ./saved_models/<protocol>/.

    python train_models.py                    # both protocols, 5-fold CV + final models (100 epochs)
    python train_models.py --protocol thesis  # only one protocol (thesis | spectrum | all)
    python train_models.py --epochs 5 --folds 3   # quick run

Outputs (per protocol)
    preprocessor.joblib     clipping bounds + scaler + class labels (fitted on the training split)
    train_set.csv / test_set.csv   the exact stratified split used
    <model>.joblib / .keras final models trained on the full training split
    cv_results.csv          per-fold validation metrics (stratified k-fold on the training split)
    <net>_history.csv       training curves of the networks
    info.json               metadata
"""
import os

# Cap the math-library thread pools BEFORE numpy/scipy load. On machines with many cores each bundled
# OpenBLAS copy reserves per-thread buffers; when Windows' commit limit is nearly used up this fails
# ("OpenBLAS error: Memory allocation") and corrupts memory (seen as an XGBoost "access violation").
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline import (CLASS_RULE, DL_MODELS, ML_MODELS, PROTOCOLS, SEED, TARGET, TEST_SIZE, artifact_dir,
                      build_dl_model, fit_preprocessor, load_and_split, slug, summary_metrics, transform)


def fit_model(name, X, y, n_classes, epochs, batch_size, verbose=0):
    if name in ML_MODELS:
        return ML_MODELS[name]().fit(X, y), None
    import tensorflow as tf
    tf.random.set_seed(SEED)
    model = build_dl_model(name, X.shape[1], n_classes)
    hist = model.fit(X[..., None], y, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                     verbose=verbose)
    return model, hist


def predict_proba(name, model, X):
    return model.predict(X[..., None], verbose=0) if name in DL_MODELS else model.predict_proba(X)


def main(protocol: str, epochs: int, batch_size: int, folds: int):
    ARTIFACT_DIR = artifact_dir(protocol)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"=== {PROTOCOLS[protocol]['name']} -> {ARTIFACT_DIR}")
    df, train_df, test_df, features, le = load_and_split(protocol)
    labels = list(le.classes_)
    y_train = le.transform(train_df[TARGET].astype(str))
    names = list(ML_MODELS) + DL_MODELS

    # ---- 1. stratified k-fold cross-validation on the training split
    cv_rows = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(train_df, y_train), start=1):
        prep_f = fit_preprocessor(train_df.iloc[tr][features].astype(float))  # refit inside the fold
        X_tr, X_va = transform(train_df.iloc[tr], prep_f), transform(train_df.iloc[va], prep_f)
        for name in names:
            model, _ = fit_model(name, X_tr, y_train[tr], len(labels), epochs, batch_size)
            proba = predict_proba(name, model, X_va)
            cv_rows.append({"Model": name, "Fold": fold,
                            **summary_metrics(y_train[va], proba.argmax(1), proba)})
        print(f"CV fold {fold}/{folds} done")
    pd.DataFrame(cv_rows).to_csv(os.path.join(ARTIFACT_DIR, "cv_results.csv"), index=False)

    # ---- 2. final models on the whole training split
    prep = fit_preprocessor(train_df[features].astype(float))
    X_train = transform(train_df, prep)
    joblib.dump({**prep, "labels": labels}, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    train_df.to_csv(os.path.join(ARTIFACT_DIR, "train_set.csv"), index=False)
    test_df.to_csv(os.path.join(ARTIFACT_DIR, "test_set.csv"), index=False)

    info = {"models": {}, "n_rows": len(df), "n_train": len(train_df), "n_test": len(test_df),
            "protocol": protocol, "features": features, "dropped_features": PROTOCOLS[protocol]["drop"],
            "labels": labels, "class_rule": CLASS_RULE,
            "test_size": TEST_SIZE, "seed": SEED, "cv_folds": folds, "epochs": epochs,
            "batch_size": batch_size, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "class_temperature": {c: [int(g.min()), int(g.max())]
                                  for c, g in df.groupby(TARGET)["temperature"]} if "temperature" in df else {}}

    for name in names:
        t0 = time.time()
        model, hist = fit_model(name, X_train, y_train, len(labels), epochs, batch_size)
        if name in ML_MODELS:
            file = f"{slug(name)}.joblib"
            joblib.dump(model, os.path.join(ARTIFACT_DIR, file))
        else:
            file = f"{slug(name)}.keras"
            model.save(os.path.join(ARTIFACT_DIR, file))
            pd.DataFrame(hist.history).to_csv(os.path.join(ARTIFACT_DIR, f"{slug(name)}_history.csv"),
                                              index=False)
        info["models"][name] = {"type": "ml" if name in ML_MODELS else "dl", "file": file,
                                "fit_seconds": round(time.time() - t0, 2),
                                "n_params": int(model.count_params()) if name in DL_MODELS else None}
        print(f"{name:22s} trained in {time.time() - t0:.1f}s")

    with open(os.path.join(ARTIFACT_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Saved to ./{ARTIFACT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    a = p.parse_args()
    for key in (PROTOCOLS if a.protocol == "all" else [a.protocol]):
        np.random.seed(SEED)
        main(key, a.epochs, a.batch_size, a.folds)
