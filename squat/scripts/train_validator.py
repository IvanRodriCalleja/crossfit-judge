"""Validity judge: compares the RULE baseline against an ML classifier on the
per-rep features, evaluated the way you'd evaluate a judge.

Key metric: the FALSE-ACCEPT RATE (passing a rep as VALID when it isn't).

Rigor with this data (a single subject): cross-validation groups BY TAKE
(GroupKFold), so the reps and the three views of the same take never land in train
and test at once. That's what keeps a plain CV from being over-optimistic.

Important caveat about lockout: the 48 INVALIDA_LOCKOUT examples come from a single
continuous take (take 3). With grouping by take, no fold can learn lockout and
evaluate it at the same time, so a lockout classifier isn't estimable without
leakage; for that criterion the solid judge is the rule (see validate_lockout_view.py).
The ML is evaluable for DEPTH, which shows up across several takes.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from squat.application.rep_features import RepFeatures
from squat.domain.validation import (INVALIDA_PROFUNDIDAD, VALIDA,
                                     RuleBasedValidator)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "validation_dataset.npz")
CLASSES = ["VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"]
VIEWS = ["lateral", "frontal", "45deg"]


def false_accept_rate(y_true, y_pred):
    invalid = y_true != VALIDA
    return float((y_pred[invalid] == VALIDA).mean()) if invalid.sum() else 0.0


def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=CLASSES, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=predicted):", CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))
    print(f"FALSE-ACCEPT RATE: {false_accept_rate(y_true, y_pred):.1%}")


def _grouped_cv_predict(X, y, groups):
    """Prediction via CV grouped by take (leave-one-take-out)."""
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    n = len(np.unique(groups))
    cv = GroupKFold(n_splits=n)
    return cross_val_predict(clf, X, y, cv=cv, groups=groups)


def depth_judge_by_view(X, y, persp, take):
    """Binary DEPTH judge (failure = INVALIDA_PROFUNDIDAD) per view: rule (domain
    threshold) vs logistic regression with CV grouped by take."""
    rule = RuleBasedValidator()
    print("\n########## DEPTH JUDGE — rule vs ML (CV grouped by take) ##########")
    print(f"{'view':8} {'n':>4} {'method':7} {'acc':>6} {'prec':>6} "
          f"{'recall':>7} {'F1':>6} {'false-acc':>12}")
    for v in VIEWS + ["TODAS"]:
        m = np.ones(len(y), bool) if v == "TODAS" else (persp == v)
        yb = (y[m] == INVALIDA_PROFUNDIDAD)  # positive = depth failure
        # rule: failure if depth_at_bottom < domain threshold
        pr_rule = X[m, 0] < rule.depth_threshold
        _print_binary(v, "rule", yb, pr_rule)
        # ML: only if the 'failure' class spans >=2 takes (otherwise leakage is unavoidable)
        pos_takes = len(np.unique(take[m][yb]))
        if pos_takes >= 2:
            pr_ml = _grouped_cv_predict(X[m], yb, take[m]).astype(bool)
            _print_binary(v, "ML", yb, pr_ml)
        else:
            print(f"{v:8} {int(m.sum()):4} {'ML':7} n/a (the 'failure' class is in "
                  f"a single take -> not evaluable without leakage)")


def _print_binary(view, method, yb, pred):
    tp = int((pred & yb).sum()); fp = int((pred & ~yb).sum())
    fn = int((~pred & yb).sum()); tn = int((~pred & ~yb).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(yb)
    fa = fn / (tp + fn) if tp + fn else 0.0
    print(f"{view:8} {len(yb):4} {method:7} {acc:6.2f} {prec:6.2f} "
          f"{rec:7.2f} {f1:6.2f} {fa:11.1%}")


def main():
    d = np.load(DATA, allow_pickle=True)
    X, y = d["X"], d["y"].astype(str)
    persp, take = d["perspective"].astype(str), d["take"].astype(str)

    # 1) Rules (explainable baseline, 3 classes) over all reps
    rules = RuleBasedValidator()
    y_rules = np.array([rules.judge(RepFeatures(*row))[0] for row in X])
    report("RULES (baseline, 3 classes) — all reps", y, y_rules)

    # 2) ML 3-class with CV grouped by take (no leakage)
    y_ml = _grouped_cv_predict(X, y, take)
    report("ML (logistic, 3 classes) — CV grouped by TAKE", y, y_ml)
    print("NOTE: the INVALIDA_LOCKOUT row is not interpretable (the whole class lives "
          "in a single take; with grouping by take the ML can't learn it).")

    # 3) Depth judge rule vs ML per view (here the ML is evaluable)
    depth_judge_by_view(X, y, persp, take)

    # 4) Cross-view transfer: train on lateral, test on frontal / 45°
    print("\n########## VIEW TRANSFER (depth, ML trained on LATERAL) ##########")
    lat = persp == "lateral"
    yb = (y == INVALIDA_PROFUNDIDAD)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[lat], yb[lat])
    for v in ["frontal", "45deg"]:
        m = persp == v
        _print_binary(v, "ML→", yb[m], clf.predict(X[m]).astype(bool))


if __name__ == "__main__":
    main()
