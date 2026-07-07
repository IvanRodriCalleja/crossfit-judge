"""F4 — Ablation of the hip joint center (HJC) correction.

Compares the DEPTH judge with and without the HJC correction (Bell et al., 1989) on
Fitness-AQA (official test, multi-subject). Reports accuracy, F1, false-accept rate and
the class separation margin (depth |Δ| and Cohen's d), which measures whether the
correction pulls the "ok" and "shallow" means apart even when the threshold already
gets it right.

Usage: python squat/scripts/ablation_hjc.py [cutoff_lateral]
"""
import os
import sys

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "fitness_aqa", "shallow.npz")
SHALLOW = 1
DI = 4  # hip_knee_depth column


def best_threshold(depth, y):
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


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return abs(a.mean() - b.mean()) / sp if sp > 0 else float("nan")


def eval_depth(depth, y, tr, te):
    thr, greater = best_threshold(depth[tr], y[tr])
    pred = np.where((depth[te] > thr) if greater else (depth[te] < thr), SHALLOW, 0)
    tp = int(np.sum((y[te] == SHALLOW) & (pred == SHALLOW)))
    fp = int(np.sum((y[te] == 0) & (pred == SHALLOW)))
    fn = int(np.sum((y[te] == SHALLOW) & (pred == 0)))
    tn = int(np.sum((y[te] == 0) & (pred == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y[te])
    fa = fn / (tp + fn) if tp + fn else 0.0
    d = cohen_d(depth[te][y[te] == 0], depth[te][y[te] == SHALLOW])
    return dict(acc=acc, f1=f1, fa=fa, d=d, thr=thr)


def run(name, feats, y, tr, te):
    depth = feats[:, DI]
    m = eval_depth(depth, y, tr, te)
    print(f"  {name:14} acc {m['acc']:.3f} | F1 {m['f1']:.3f} | "
          f"false-acc {m['fa']:5.1%} | Cohen's d {m['d']:.2f}")
    return m


def main():
    cutoff = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
    d = np.load(CACHE, allow_pickle=True)
    ok = d["pose_ok"] & np.isfinite(d["view"])
    for rep in ("features", "features_hjc"):
        ok = ok & np.isfinite(d[rep]).all(1)
    y, split, view = d["label"][ok], d["split"][ok], d["view"][ok]
    base, hjc = d["features"][ok], d["features_hjc"][ok]
    tr = np.isin(split, ["train", "val"]); te = split == "test"

    print("== HJC ABLATION — depth judge (Fitness-AQA, official test) ==")
    print("\nFULL (all views):")
    b = run("no HJC", base, y, tr, te)
    h = run("with HJC", hjc, y, tr, te)
    print(f"  Δ false-accept: {(h['fa']-b['fa'])*100:+.1f} pp | "
          f"Δ F1: {h['f1']-b['f1']:+.3f} | Δ Cohen's d: {h['d']-b['d']:+.2f}")

    lat = view < cutoff
    print(f"\nLATERAL (view<{cutoff}, {int(lat.sum())} crops):")
    bl = run("no HJC", base[lat], y[lat], tr[lat], te[lat])
    hl = run("with HJC", hjc[lat], y[lat], tr[lat], te[lat])
    print(f"  Δ false-accept: {(hl['fa']-bl['fa'])*100:+.1f} pp | "
          f"Δ F1: {hl['f1']-bl['f1']:+.3f} | Δ Cohen's d: {hl['d']-bl['d']:+.2f}")

    nonlat = view >= cutoff
    print(f"\nNON-LATERAL (view>={cutoff}, {int(nonlat.sum())} crops; frontal/oblique/back):")
    bn = run("no HJC", base[nonlat], y[nonlat], tr[nonlat], te[nonlat])
    hn = run("with HJC", hjc[nonlat], y[nonlat], tr[nonlat], te[nonlat])
    print(f"  Δ false-accept: {(hn['fa']-bn['fa'])*100:+.1f} pp | "
          f"Δ F1: {hn['f1']-bn['f1']:+.3f} | Δ Cohen's d: {hn['d']-bn['d']:+.2f}")

    print("\nEffect of the correction on mean depth (test):")
    for lab, mask in [("ok", y[te] == 0), ("shallow", y[te] == SHALLOW)]:
        print(f"  {lab:8}: no HJC {base[te][mask,DI].mean():+.3f} -> "
              f"with HJC {hjc[te][mask,DI].mean():+.3f}")


if __name__ == "__main__":
    main()
