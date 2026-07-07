"""Build the PER-REP dataset for validation from the recordings cache (features +
detected reps + per-rep validity label).

Saves squat/data/cache/validation_dataset.npz (X: per-rep features, y: labels,
perspective, take) and shows whether the geometric measures separate the classes
(the check that the formalised criteria work).
"""
import json
import os
from collections import defaultdict

import numpy as np

from squat.application.rep_features import FEATURE_NAMES, extract_rep_features

REC_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "validation_dataset.npz")
TRUNCATED = {("45deg", "4"), ("45deg", "5")}  # incomplete views: excluded


def main():
    items = json.load(open(os.path.join(REC_CACHE, "manifest.json")))["items"]
    X, y, persp, take = [], [], [], []
    for it in items:
        if (it["perspective"], it["take"]) in TRUNCATED:
            continue
        npz = np.load(os.path.join(REC_CACHE, it["key"] + ".npz"))
        reps = extract_rep_features(npz["features"], npz["bottoms"])
        labels = [str(x) for x in npz["labels"]]
        for rf, lab in zip(reps, labels):
            X.append(rf.to_list()); y.append(lab)
            persp.append(it["perspective"]); take.append(it["take"])
    X = np.array(X, dtype=np.float32); y = np.array(y)
    np.savez(OUT, X=X, y=y, perspective=np.array(persp), take=np.array(take),
             feature_names=np.array(FEATURE_NAMES))
    print(f"per-rep dataset: {len(y)} reps -> {os.path.abspath(OUT)}")
    print("by class:", {c: int((y == c).sum()) for c in sorted(set(y))})

    # do the geometric measures separate the classes? (lateral view, the most reliable)
    print("\n== LATERAL view: per-class means (do the criteria separate?) ==")
    latmask = np.array(persp) == "lateral"
    Xl, yl = X[latmask], y[latmask]
    print(f"{'class':22} {'depth_at_bottom':>16} {'knee_min':>10} {'hip_ext_at_top':>15}  n")
    for c in sorted(set(yl)):
        m = Xl[yl == c].mean(0)
        print(f"{c:22} {m[0]:16.2f} {m[1]:10.0f} {m[2]:15.0f}  {int((yl==c).sum())}")
    print("\nExpected: PROFUNDIDAD -> low depth_at_bottom (doesn't cross 0); "
          "LOCKOUT -> low hip_ext_at_top; VALIDA -> both high.")


if __name__ == "__main__":
    main()
