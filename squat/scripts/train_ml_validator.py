"""Machine-learning REP/NO-REP validator, multi-view, with 2D image features (F5).
Replaces the rule-based judge; the rules stay as the baseline.

Input: rich 2D per-rep features + the view as context. NO-REPs are augmented with
counterfactuals cropped from the real valid reps (per view, same scale), which also
covers the scarce classes in each view. Honest leave-one-take-out evaluation (the
three views of a take never get split) and a per-view breakdown, comparing the ML
against two rule baselines: single threshold and per-view threshold.
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from squat.application.counterfactual import (lockout_window, real_features,
                                              rep_windows_from_bottoms,
                                              shallow_window_frac)
from squat.application.rich_features import rich_features
from squat.domain.validation import INVALIDA_LOCKOUT, INVALIDA_PROFUNDIDAD, VALIDA

REC2D = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones_2d")
VIEWS = ["lateral", "frontal", "45deg"]
LOCK_TARGETS = [150, 155, 160, 165]      # hip extension (degrees; common scale)
SHAL_FRACS = [0.40, 0.55, 0.70, 0.82]    # fraction of the range (relative to the view)
DEPTH_COL = 0  # rich_features[0] = depth_max (depth at the bottom)


def load_real():
    X, W, y, view, take = [], [], [], [], []
    manifest = json.load(open(os.path.join(REC2D, "manifest.json")))["items"]
    for it in manifest:
        d = np.load(os.path.join(REC2D, it["key"] + ".npz"), allow_pickle=True)
        for w, lab in zip(rep_windows_from_bottoms(d["features"], d["bottoms"]),
                          [str(x) for x in d["labels"]]):
            X.append(rich_features(w)); W.append(w); y.append(lab)
            view.append(it["perspective"]); take.append(it["take"])
    return (np.array(X, np.float32), W, np.array(y), np.array(view), np.array(take))


def cf_from_windows(windows, views):
    """Synthetic no-reps by cropping real valid reps (per view, coherent scale)."""
    X, y, vw = [], [], []
    for w, v in zip(windows, views):
        rf = real_features(w)
        if rf.hip_ext_at_top < 150:
            continue
        # Only augment the SCARCE class (lockout). Depth already has enough real
        # examples across several takes; its counterfactual adds nothing and biases.
        for t in LOCK_TARGETS:
            lw = lockout_window(w, t)
            if lw is not None:
                X.append(rich_features(lw)); y.append(INVALIDA_LOCKOUT); vw.append(v)
    return np.array(X, np.float32), np.array(y), np.array(vw)


def report(name, y_true, y_pred, view):
    print(f"\n{name}")
    print(f"{'view':8} {'n':>4} {'acc':>6} {'false-acc':>12}")
    for v in VIEWS + ["GLOBAL"]:
        m = np.ones(len(y_true), bool) if v == "GLOBAL" else (view == v)
        if m.sum() == 0:
            continue
        acc = (y_pred[m] == y_true[m]).mean()
        inval = y_true[m] != VALIDA
        fa = (y_pred[m][inval] == VALIDA).mean() if inval.any() else 0.0
        print(f"{v:8} {int(m.sum()):4} {acc:6.2f} {fa:11.1%}")


def rule_depth(depth, sh_mask, thr):
    return np.where(depth < thr, INVALIDA_PROFUNDIDAD, VALIDA)


def calib_threshold(depth, y):
    """Depth threshold that maximizes binary valid/shallow accuracy."""
    sh = y == INVALIDA_PROFUNDIDAD
    keep = np.isin(y, [VALIDA, INVALIDA_PROFUNDIDAD])
    best = (None, -1)
    for thr in np.quantile(depth[keep], np.linspace(0.05, 0.95, 80)):
        acc = ((depth[keep] < thr) == sh[keep]).mean()
        if acc > best[1]:
            best = (float(thr), acc)
    return best[0]


def main():
    Xr, Wr, yr, view, take = load_real()
    print(f"real (2D): {len(yr)} reps {dict(zip(*np.unique(yr, return_counts=True)))}")

    # --- BASELINE 1: rule, SINGLE threshold calibrated on lateral ---
    lat = view == "lateral"
    thr1 = calib_threshold(Xr[lat, DEPTH_COL], yr[lat])
    # baseline only tells depth apart (not lockout) -> compared only on that task
    y1 = rule_depth(Xr[:, DEPTH_COL], None, thr1)

    # --- BASELINE 2: rule, PER-VIEW threshold (rule ceiling; needs calibrating each one) ---
    y2 = np.empty_like(yr)
    for v in VIEWS:
        m = view == v
        thr = calib_threshold(Xr[m, DEPTH_COL], yr[m])
        y2[m] = rule_depth(Xr[m, DEPTH_COL], None, thr)

    # --- ML: rich 2D + view one-hot; counterfactuals from valid reps of the train takes ---
    oh = np.stack([(view == v).astype(np.float32) for v in VIEWS], axis=1)
    Xr_ctx = np.column_stack([Xr, oh])
    y_ml = np.empty_like(yr)
    for t in np.unique(take):
        te = take == t
        tr = ~te
        valids = [(Wr[i], view[i]) for i in np.where(tr & (yr == VALIDA))[0]]
        if valids:
            ws, vs = zip(*valids)
            Xcf, ycf, vcf = cf_from_windows(list(ws), np.array(vs))
            oh_cf = np.stack([(vcf == v).astype(np.float32) for v in VIEWS], axis=1)
            Xcf_ctx = np.column_stack([Xcf, oh_cf])
        else:
            Xcf_ctx, ycf = np.empty((0, Xr_ctx.shape[1]), np.float32), np.array([])
        Xtr = np.vstack([Xr_ctx[tr], Xcf_ctx]); ytr = np.concatenate([yr[tr], ycf])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                             max_depth=3, random_state=0)
        clf.fit(Xtr, ytr)
        y_ml[te] = clf.predict(Xr_ctx[te])

    # To compare on the SAME task (depth), collapsing lockout->valid in a depth
    # judgement doesn't apply; we report the ML's full judge and the depth part of
    # the rules.
    dmask = np.isin(yr, [VALIDA, INVALIDA_PROFUNDIDAD])
    report("BASELINE rule, SINGLE threshold (depth only)", yr[dmask], y1[dmask], view[dmask])
    report("BASELINE rule, PER-VIEW threshold (depth only)", yr[dmask], y2[dmask], view[dmask])
    report("ML VALIDATOR (3 classes, multi-view, 2D + counterfactual)", yr, y_ml, view)
    report("  · ML restricted to depth (comparable with rules)",
           yr[dmask], y_ml[dmask], view[dmask])


if __name__ == "__main__":
    main()
