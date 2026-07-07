"""Train a SINGLE validity judge (one model, any camera view).

The model takes the per-rep measures and outputs one of three classes:
VALIDA / INVALIDA_PROFUNDIDAD / INVALIDA_LOCKOUT. It is NOT told the camera view.

Training data is synthesized from every valid rep we have (RepCount + Exercise
Recognition, hundreds of reps across many subjects): the full rep is a VALIDA
example, and trimming its trajectory (counterfactual) yields shallow and lockout
failures with exact labels. This turns the scarce-negative problem into an abundant
one without needing to record more subjects.

Test is always the REAL recordings (never seen in training), across the three views.

Usage: python squat/scripts/train_single_validator.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from squat.application.counterfactual import (lockout_fail, real_features,
                                              rep_windows, shallow_window_frac)

HERE = os.path.dirname(__file__)
SRC_CACHE = os.path.join(HERE, "..", "data", "cache", "repcount")
REAL = os.path.join(HERE, "..", "data", "cache", "validation_dataset.npz")

VALIDA, SHALLOW, LOCKOUT = "VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"
SHALLOW_FRACS = [0.35, 0.45, 0.55, 0.65, 0.75]     # partial depth reached (never parallel)
LOCKOUT_TARGETS = [148, 152, 156, 160, 164, 168]   # hip extension short of full lockout


def _feats(rf):
    return [rf.depth_at_bottom, rf.knee_min, rf.hip_ext_at_top]


def build_training_set():
    """Synthesize (X, y) from every valid rep across all source videos."""
    X, y = [], []
    n_valid = n_src = 0
    for f in sorted(glob.glob(os.path.join(SRC_CACHE, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "reps" not in d or len(d["reps"]) == 0:
            continue
        for w in rep_windows(d["features"], d["reps"]):
            n_src += 1
            X.append(_feats(real_features(w))); y.append(VALIDA); n_valid += 1
            for frac in SHALLOW_FRACS:
                sw = shallow_window_frac(w, frac)
                if sw is not None and len(sw) >= 2:
                    X.append(_feats(real_features(sw))); y.append(SHALLOW)
            for t in LOCKOUT_TARGETS:
                rf = lockout_fail(w, t)
                if rf is not None:
                    X.append(_feats(rf)); y.append(LOCKOUT)
    print(f"source reps: {n_src} (from {len(glob.glob(os.path.join(SRC_CACHE,'*.npz')))} videos)")
    return np.asarray(X, dtype=float), np.asarray(y)


def report(name, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = (y_true == y_pred).mean()
    inval = y_true != VALIDA
    fa = np.mean(y_pred[inval] == VALIDA) if inval.any() else 0.0
    print(f"\n=== {name}  (n={len(y_true)}) ===")
    print(f"  accuracy {acc:.3f} | false-accept (invalid passed as valid) {fa:.1%}")
    for i, lab in enumerate(labels):
        support = cm[i].sum()
        rec = cm[i, i] / support if support else 0.0
        col = cm[:, i].sum()
        prec = cm[i, i] / col if col else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {lab:22} prec {prec:.2f} rec {rec:.2f} F1 {f1:.2f} (n={support})")
    return acc, fa


def main():
    labels = [VALIDA, SHALLOW, LOCKOUT]
    Xtr, ytr = build_training_set()
    uniq, cnt = np.unique(ytr, return_counts=True)
    print("training samples:", dict(zip(uniq.tolist(), cnt.tolist())))

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=4, class_weight="balanced")
    clf.fit(Xtr, ytr)

    d = np.load(REAL, allow_pickle=True)
    order = list(d["feature_names"])
    print("real feature order:", order)  # expected: depth_at_bottom, knee_min, hip_ext_at_top
    Xte, yte, persp = d["X"].astype(float), d["y"].astype(str), d["perspective"].astype(str)
    pred = clf.predict(Xte)

    report("REAL recordings — ALL views (single model, view unknown)", yte, pred, labels)
    for v in ["lateral", "frontal", "45deg"]:
        m = persp == v
        if m.any():
            report(f"REAL — {v}", yte[m], pred[m], labels)


if __name__ == "__main__":
    main()
