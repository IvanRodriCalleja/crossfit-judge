"""Single validity judge on view-agnostic 2D image features (one model, any camera).

Same idea as train_single_validator.py but on the 2D image-plane features, which
stay observable from any camera angle (unlike the estimated 3D world depth, which
MediaPipe misjudges off the lateral view). Negatives are synthesized by trimming the
trajectory of the real VALID reps (counterfactual); the model never receives the
camera view as input.

Honest protocol: leave-one-take-out. Training negatives/positives are synthesized
from the valid reps of the OTHER takes; the held-out take's REAL reps (all views)
are the test. This keeps every test rep unseen (no rep/view of a take leaks).

Usage: python squat/scripts/train_single_validator_2d.py
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
                                              rep_windows_from_bottoms,
                                              shallow_window_frac)

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones_2d")
VALIDA, SHALLOW, LOCKOUT = "VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"
SHALLOW_FRACS = [0.35, 0.45, 0.55, 0.65, 0.75]
LOCKOUT_TARGETS = [148, 152, 156, 160, 164, 168]


def _feats(rf):
    return [rf.depth_at_bottom, rf.knee_min, rf.hip_ext_at_top]


def load_reps():
    """Every real rep as (features_vector, label, view, take)."""
    reps = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        name = os.path.basename(f)[:-4]           # e.g. frontal_2
        view, take = name.rsplit("_", 1)
        d = np.load(f, allow_pickle=True)
        feats, bottoms, labels = d["features"], list(d["bottoms"]), list(d["labels"])
        windows = rep_windows_from_bottoms(feats, bottoms)
        # rep_windows_from_bottoms keeps the same order as bottoms (drops none here)
        for w, lab in zip(windows, labels):
            reps.append((w, str(lab), view, take))
    return reps


def synth_from_valid(windows):
    """Abundant training samples from a set of VALID rep windows."""
    X, y = [], []
    for w in windows:
        X.append(_feats(real_features(w))); y.append(VALIDA)
        for frac in SHALLOW_FRACS:
            sw = shallow_window_frac(w, frac)
            if sw is not None and len(sw) >= 2:
                X.append(_feats(real_features(sw))); y.append(SHALLOW)
        for t in LOCKOUT_TARGETS:
            rf = lockout_fail(w, t)
            if rf is not None:
                X.append(_feats(rf)); y.append(LOCKOUT)
    return np.asarray(X, float), np.asarray(y)


def report(name, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = (y_true == y_pred).mean()
    inval = y_true != VALIDA
    fa = np.mean(y_pred[inval] == VALIDA) if inval.any() else 0.0
    print(f"\n=== {name}  (n={len(y_true)}) ===")
    print(f"  accuracy {acc:.3f} | false-accept {fa:.1%}")
    for i, lab in enumerate(labels):
        sup = cm[i].sum(); col = cm[:, i].sum()
        rec = cm[i, i] / sup if sup else 0.0
        prec = cm[i, i] / col if col else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"    {lab:22} prec {prec:.2f} rec {rec:.2f} F1 {f1:.2f} (n={sup})")


def main():
    labels = [VALIDA, SHALLOW, LOCKOUT]
    reps = load_reps()
    takes = sorted({t for _, _, _, t in reps})
    print(f"real reps: {len(reps)} across takes {takes}")

    y_true, y_pred, views = [], [], []
    for held in takes:                            # leave-one-take-out
        train_valid = [w for (w, lab, v, t) in reps if t != held and lab == VALIDA]
        Xtr, ytr = synth_from_valid(train_valid)
        if len(set(ytr)) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=4, class_weight="balanced")
        clf.fit(Xtr, ytr)
        test = [(w, lab, v) for (w, lab, v, t) in reps if t == held]
        Xte = np.asarray([_feats(real_features(w)) for w, _, _ in test], float)
        for (w, lab, v), p in zip(test, clf.predict(Xte)):
            y_true.append(lab); y_pred.append(p); views.append(v)

    y_true, y_pred, views = np.array(y_true), np.array(y_pred), np.array(views)
    report("2D single model — ALL views (camera unknown, leave-one-take-out)",
           y_true, y_pred, labels)
    for v in ["lateral", "frontal", "45deg"]:
        m = views == v
        if m.any():
            report(f"2D single model — {v}", y_true[m], y_pred[m], labels)


if __name__ == "__main__":
    main()
