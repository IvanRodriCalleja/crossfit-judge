"""Validate the DEPTH criterion on Fitness-AQA (multi-subject).

Compares a RULE judge (depth threshold, calibrated on train) against an ML classifier
(logistic regression on the pose features), on the official test split (unseen
subjects). Judge metrics: precision, recall and F1 of the "shallow" class, accuracy
and the FALSE-ACCEPT RATE (passing a shallow rep as good), a judge's serious error.

Two readings:
  (1) FULL: all crops with pose (a mix of camera angles).
  (2) LATERAL: only lateral-view crops (view < cutoff), where depth is measurable with
      monocular pose. This is the comparison that tests our lateral-first judge on
      subjects other than the ones in our own recordings.

Usage: python squat/scripts/validate_depth.py [cutoff_lateral]
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "fitness_aqa", "shallow.npz")
FIG = os.path.join(os.path.dirname(__file__), "..", "data", "faqa_confusion.png")
SHALLOW = 1  # positive label = shallow (the error to detect)


def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == SHALLOW) & (y_pred == SHALLOW)))
    fp = int(np.sum((y_true == 0) & (y_pred == SHALLOW)))
    fn = int(np.sum((y_true == SHALLOW) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y_true)
    n_shallow = tp + fn
    false_accept = fn / n_shallow if n_shallow else 0.0  # shallow passed as good
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, false_accept=false_accept,
                cm=np.array([[tn, fp], [fn, tp]]))


def report(name, y_true, y_pred):
    m = metrics(y_true, y_pred)
    print(f"\n=== {name}  (n={len(y_true)}, shallow={int(np.sum(y_true==SHALLOW))}) ===")
    print(f"  accuracy {m['acc']:.3f} | precision {m['prec']:.3f} | "
          f"recall {m['rec']:.3f} | F1 {m['f1']:.3f}")
    print(f"  FALSE-ACCEPT RATE (shallow passed as good): {m['false_accept']:.1%}")
    print(f"  matrix [[TN {m['cm'][0,0]}, FP {m['cm'][0,1]}], "
          f"[FN {m['cm'][1,0]}, TP {m['cm'][1,1]}]]")
    return m


def best_threshold(depth, y):
    """Depth threshold and direction that maximizes balanced accuracy on train.
    Returns (thr, greater) where greater=True predicts shallow if depth > thr."""
    best = (None, True, -1.0)
    for thr in np.quantile(depth, np.linspace(0.02, 0.98, 97)):
        for greater in (True, False):
            pred = (depth > thr) if greater else (depth < thr)
            pred = np.where(pred, SHALLOW, 0)
            tpr = np.mean(pred[y == SHALLOW] == SHALLOW) if np.any(y == SHALLOW) else 0
            tnr = np.mean(pred[y == 0] == 0) if np.any(y == 0) else 0
            bal = (tpr + tnr) / 2
            if bal > best[2]:
                best = (float(thr), greater, bal)
    return best[0], best[1]


def evaluate(tag, feats, depth, y, tr, te, di):
    """Rules vs logistic for one set (train mask tr and test mask te)."""
    print(f"\n########## {tag} ##########")
    print(f"train {int(tr.sum())} | test {int(te.sum())}")
    # (1) Rule: depth threshold calibrated on train
    thr, greater = best_threshold(depth[tr], y[tr])
    sign = ">" if greater else "<"
    print(f"rule learned on train: shallow if depth {sign} {thr:+.3f}")
    pred_rule = np.where((depth[te] > thr) if greater else (depth[te] < thr), SHALLOW, 0)
    m_rule = report(f"{tag} · RULE (depth threshold)", y[te], pred_rule)
    # (2) ML: logistic regression on all features
    sc = StandardScaler().fit(feats[tr])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(feats[tr]), y[tr])
    pred_ml = clf.predict(sc.transform(feats[te]))
    m_ml = report(f"{tag} · ML (logistic regression)", y[te], pred_ml)
    return m_rule, m_ml


def main():
    cutoff = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
    d = np.load(CACHE, allow_pickle=True)
    names = list(d["feature_names"])
    di = names.index("hip_knee_depth")
    ok = d["pose_ok"]
    split, label, view = d["split"], d["label"], d["view"]

    # Pose coverage
    print("== POSE COVERAGE ==")
    for s in ("train", "val", "test", None):
        mask = np.ones_like(ok) if s is None else (split == s)
        name = "TOTAL" if s is None else s
        print(f"  {name:6}: {int(ok[mask].sum())}/{int(mask.sum())} "
              f"({ok[mask].mean():.1%})")

    figs = {}
    for rep in ("features", "features_2d"):  # world landmarks vs image plane
        m = ok & np.isfinite(view) & np.isfinite(d[rep]).all(1)
        feats = d[rep][m]; depth = feats[:, di]
        y, sp, vw = label[m], split[m], view[m]
        tr = np.isin(sp, ["train", "val"]); te = sp == "test"
        tag = "WORLD 3D" if rep == "features" else "IMAGE 2D"
        print(f"\n\n#################### REPRESENTATION: {tag} ####################")

        print("== depth separation by view ==")
        for lo, hi, lab in [(0, cutoff, f"LATERAL(<{cutoff})"), (cutoff, 1e9, "rest")]:
            vm = (vw >= lo) & (vw < hi)
            d0 = depth[vm & (y == 0)]; d1 = depth[vm & (y == SHALLOW)]
            print(f"  {lab:14}: n={int(vm.sum()):4}  ok={d0.mean():+.3f}  "
                  f"shallow={d1.mean():+.3f}  |Δ|={abs(d0.mean()-d1.mean()):.3f}")

        r_full = evaluate(f"{tag} · FULL (all views)", feats, depth, y, tr, te, di)
        lat = vw < cutoff
        r_lat = evaluate(f"{tag} · LATERAL (view<{cutoff})", feats[lat], depth[lat],
                         y[lat], tr[lat], te[lat], di)
        print(f"lateral coverage: {int(lat.sum())}/{len(vw)} ({lat.mean():.1%})")
        figs[tag] = (r_full, r_lat)

    _save_figure(figs)


def _save_figure(figs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(no matplotlib, skipping figure: {e})")
        return
    # One panel per representation (world 3D / image 2D) × {full, lateral} · rule
    panels = []
    for tag, (r_full, r_lat) in figs.items():
        panels.append((f"{tag}\nFull · rule", r_full[0]))
        panels.append((f"{tag}\nLateral · rule", r_lat[0]))
    fig, axgrid = plt.subplots(2, 2, figsize=(7.4, 7.6))
    axes = axgrid.flatten()
    for ax, (title, m) in zip(axes, panels):
        cm = m["cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["ok", "shallow"]); ax.set_yticklabels(["ok", "shallow"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title}\nF1={m['f1']:.2f} FA={m['false_accept']:.0%}", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print(f"\nfigure saved: {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
