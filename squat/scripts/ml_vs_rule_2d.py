"""Does ML beat the rule at MULTI-VIEW depth validation (with 2D features)?

With 2D image depth the signal is present in all three views, but at a different SCALE.
A rule uses a single threshold (fails outside the calibration view); an ML can learn
all three scales if it gets the view as context. They're compared with an honest
leave-one-take-out split (no leakage between takes) over the recordings.

Reads the cache/frontal_2d.npz cache (2D depth, knee, view, take per rep).
"""
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "frontal_2d.npz")
VIEWS = ["lateral", "frontal", "45deg"]


def scores(sh, pred):
    tp = int((pred & sh).sum()); fp = int((pred & ~sh).sum())
    fn = int((~pred & sh).sum()); tn = int((~pred & ~sh).sum())
    acc = (tp + tn) / len(sh)
    fa = fn / (tp + fn) if tp + fn else 0.0
    return acc, fa


def by_view(sh, pred, view, title):
    print(f"\n{title}")
    print(f"{'view':8} {'acc':>6} {'false-acc':>12}")
    for v in VIEWS:
        m = view == v
        if m.sum():
            a, fa = scores(sh[m], pred[m])
            print(f"{v:8} {a:6.2f} {fa:11.1%}")
    a, fa = scores(sh, pred)
    print(f"{'GLOBAL':8} {a:6.2f} {fa:11.1%}")


def main():
    d = np.load(CACHE, allow_pickle=True)
    depth, knee = d["depth2d"], d["knee"]
    y, view, take = d["y"].astype(str), d["perspective"].astype(str), d["take"].astype(str)
    keep = np.isin(y, ["VALIDA", "INVALIDA_PROFUNDIDAD"])
    depth, knee, view, take = depth[keep], knee[keep], view[keep], take[keep]
    sh = (y[keep] == "INVALIDA_PROFUNDIDAD")

    # (1) RULE: single threshold calibrated on lateral (best accuracy on lateral)
    lat = view == "lateral"
    best = None
    for thr in np.quantile(depth[lat], np.linspace(0.05, 0.95, 80)):
        a, _ = scores(sh[lat], depth[lat] < thr)
        if best is None or a > best[1]:
            best = (thr, a)
    by_view(sh, depth < best[0], view, f"(1) RULE single threshold (lateral, depth2d<{best[0]:+.3f})")

    # (2) ML: 2D depth + knee + view (one-hot) as context; LOTO, no leakage
    oh = np.stack([(view == v).astype(float) for v in VIEWS], axis=1)
    X = np.column_stack([depth, knee, oh])
    pred = np.empty(len(sh), bool)
    for t in np.unique(take):
        te = take == t
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=3, random_state=0)
        clf.fit(X[~te], sh[~te])
        pred[te] = clf.predict(X[te]).astype(bool)
    by_view(sh, pred, view, "(2) ML 2D depth + view (leave-one-take-out)")


if __name__ == "__main__":
    main()
