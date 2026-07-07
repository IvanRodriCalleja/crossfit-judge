"""F3 — LOCKOUT validation and quantitative analysis by VIEW (our own recordings).

Fitness-AQA doesn't label lockout; our own recordings do (INVALIDA_LOCKOUT). Here we
evaluate, per camera view, each rulebook criterion on its own:

  - Depth (breaking parallel): detect INVALIDA_PROFUNDIDAD with depth_at_bottom.
  - Lockout (hip extension at the top): detect INVALIDA_LOCKOUT with hip_ext_at_top.

Hypothesis: lockout is a joint ANGLE and should tolerate the change of view, whereas
depth is a vertical position and degrades away from the lateral view.

The judge is RULE based (threshold) calibrated on the LATERAL view (the reference) and
applied to every view: the realistic transfer (calibrate once on the reliable view).
No per-view ML: the LOCKOUT class comes from a single continuous take, so a classifier
would memorize the take instead of learning the criterion.

Usage: python squat/scripts/validate_lockout_view.py
"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "validation_dataset.npz")
FIG = os.path.join(os.path.dirname(__file__), "..", "data", "vista_criterios.png")
VIEWS = ["lateral", "frontal", "45deg"]
# validation_dataset columns: [depth_at_bottom, knee_min, hip_ext_at_top]
DEPTH, HIP_EXT = 0, 2


def best_threshold(val, pos):
    """Threshold that maximizes balanced accuracy: predicts failure if val < thr
    (positive = pos, the criterion not met)."""
    best = (None, -1.0)
    for thr in np.quantile(val, np.linspace(0.02, 0.98, 97)):
        pred = val < thr
        tpr = pred[pos].mean() if pos.any() else 0
        tnr = (~pred[~pos]).mean() if (~pos).any() else 0
        bal = (tpr + tnr) / 2
        if bal > best[1]:
            best = (float(thr), bal)
    return best[0]


def judge_metrics(val, pos, thr):
    """pos = mask of the 'failure' class; predicts failure if val < thr."""
    pred = val < thr
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & ~pos).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(val)
    false_accept = fn / (tp + fn) if tp + fn else 0.0  # failure passed as good
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, fa=false_accept, n=len(val),
                npos=int(pos.sum()))


def analyze(name, col, pos_class, neg_classes, X, y, persp):
    """Analyze a binary criterion (failure vs pass) per view. Threshold calibrated
    on lateral and applied to every view."""
    print(f"\n########## CRITERION: {name} ##########")
    keep = np.isin(y, [pos_class] + neg_classes)
    lat = keep & (persp == "lateral")
    thr = best_threshold(X[lat, col], y[lat] == pos_class)
    print(f"threshold calibrated on LATERAL: failure if {FEATURE_LABEL[col]} < {thr:.2f}")
    print(f"{'view':8} {'n':>4} {'fails':>7} {'acc':>6} {'prec':>6} "
          f"{'recall':>7} {'F1':>6} {'false-acc':>12}")
    rows = {}
    for v in VIEWS:
        m = keep & (persp == v)
        met = judge_metrics(X[m, col], y[m] == pos_class, thr)
        rows[v] = met
        print(f"{v:8} {met['n']:4} {met['npos']:7} {met['acc']:6.2f} "
              f"{met['prec']:6.2f} {met['rec']:7.2f} {met['f1']:6.2f} {met['fa']:11.1%}")
    return rows


def full_judge_by_view(X, y, persp):
    """Full REP/NO-REP judge (3 classes, explainable rules) per view: accuracy and
    false-accept rate (any invalid rep passed as VALID)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from squat.application.rep_features import RepFeatures
    from squat.domain.validation import VALIDA, RuleBasedValidator
    rules = RuleBasedValidator()
    print("\n########## FULL JUDGE (3 classes, rules) per view ##########")
    print(f"{'view':8} {'n':>4} {'acc':>6} {'false-acc':>12}")
    for v in VIEWS:
        m = persp == v
        pred = np.array([rules.judge(RepFeatures(*row))[0] for row in X[m]])
        acc = (pred == y[m]).mean()
        inval = y[m] != VALIDA
        fa = (pred[inval] == VALIDA).mean() if inval.any() else 0.0
        print(f"{v:8} {int(m.sum()):4} {acc:6.2f} {fa:11.1%}")


FEATURE_LABEL = {DEPTH: "depth_at_bottom", HIP_EXT: "hip_ext_at_top"}


def save_figure(X, y, persp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(no matplotlib: {e})"); return
    classes = ["VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"]
    colors = {"VALIDA": "#2a9d8f", "INVALIDA_PROFUNDIDAD": "#e76f51",
              "INVALIDA_LOCKOUT": "#e9c46a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], DEPTH, "Depth (depth_at_bottom)"),
                           (axes[1], HIP_EXT, "Lockout (hip_ext_at_top, °)")]:
        w = 0.25
        for k, c in enumerate(classes):
            means = [X[(persp == v) & (y == c), col].mean() for v in VIEWS]
            sds = [X[(persp == v) & (y == c), col].std() for v in VIEWS]
            xs = np.arange(len(VIEWS)) + (k - 1) * w
            ax.bar(xs, means, w, yerr=sds, capsize=3, color=colors[c],
                   label=c.replace("INVALIDA_", "").capitalize())
        ax.set_xticks(np.arange(len(VIEWS))); ax.set_xticklabels(VIEWS)
        ax.set_title(title)
        if col == HIP_EXT:
            ax.set_ylim(140, 180)  # zoom in where the lockout separation happens
        else:
            ax.axhline(0, color="gray", lw=0.6)  # ~0 = parallel broken
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Criterion separation by class and view "
                 "(depth only in lateral; lockout in all)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print(f"\nfigure saved: {os.path.abspath(FIG)}")


def main():
    d = np.load(DATA, allow_pickle=True)
    X, y, persp = d["X"], d["y"].astype(str), d["perspective"].astype(str)
    analyze("DEPTH (failure = shallow)", DEPTH,
            "INVALIDA_PROFUNDIDAD", ["VALIDA", "INVALIDA_LOCKOUT"], X, y, persp)
    analyze("LOCKOUT (failure = no hip extension)", HIP_EXT,
            "INVALIDA_LOCKOUT", ["VALIDA", "INVALIDA_PROFUNDIDAD"], X, y, persp)
    full_judge_by_view(X, y, persp)
    save_figure(X, y, persp)


if __name__ == "__main__":
    main()
