"""F3d — Counterfactual augmentation for DEPTH (same trick as lockout).

Crops the DESCENT of real valid reps to fabricate shallow no-reps, with an exact label
and without editing joints. Since depth depends on the view, everything happens within
a single view and scale: they're generated from the lateral recordings and validated
against real lateral no-reps.

It also shows MULTI-SEVERITY generation: from each valid rep we take several samples at
different target depths (a continuum of severity, borderline cases included),
multiplying the number of examples per subject.

Leakage-free split BY TAKE: generated from continuous valid takes (2 and 5) and tested
on takes with real shallow reps (1 and 4). No take is on both sides.
"""
import os

import numpy as np

from squat.application.counterfactual import (multi_severity, real_features,
                                              rep_windows_from_bottoms)

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones")
DI = 0  # depth_at_bottom en RepFeatures.to_list()
TRAIN_TAKES = ["2", "5"]   # continuous valid takes (generation source)
TEST_TAKES = ["1", "4"]    # contain real shallow reps


def load(take):
    d = np.load(os.path.join(CACHE, f"lateral_{take}.npz"), allow_pickle=True)
    return d["features"], d["bottoms"], [str(x) for x in d["labels"]]


def rep_vectors(take):
    """Real per-rep features (depth_at_bottom, knee_min, hip_ext) + label."""
    feats, bottoms, labels = load(take)
    from squat.application.rep_features import extract_rep_features
    return extract_rep_features(feats, bottoms), labels


def main():
    # --- TRAINING: only data GENERATED from valid reps of the continuous takes ---
    gen_shallow, gen_valid = [], []
    n_src = 0
    targets = [-0.15, -0.25, -0.35, -0.45]  # multi-severity (from borderline to clear)
    for take in TRAIN_TAKES:
        feats, bottoms, labels = load(take)
        for w, lab in zip(rep_windows_from_bottoms(feats, bottoms), labels):
            if lab != "VALIDA":
                continue
            n_src += 1
            gen_valid.append(real_features(w).to_list())
            gen_shallow += [r.to_list() for r in multi_severity(w, "shallow", targets)]
    Xtr = np.array(gen_valid + gen_shallow, np.float32)
    ytr = np.array([0] * len(gen_valid) + [1] * len(gen_shallow))
    print(f"SAMPLE MULTIPLICATION: {n_src} real valid reps -> "
          f"{len(gen_valid)} valid + {len(gen_shallow)} shallow generated "
          f"({len(gen_shallow)/max(1,n_src):.1f}× per rep, {len(targets)} severities)")

    # --- TEST: REAL no-reps (different takes, no leakage) ---
    Xte, yte = [], []
    for take in TEST_TAKES:
        reps, labels = rep_vectors(take)
        for rf, lab in zip(reps, labels):
            if lab in ("VALIDA", "INVALIDA_PROFUNDIDAD"):
                Xte.append(rf.to_list()); yte.append(int(lab == "INVALIDA_PROFUNDIDAD"))
    Xte, yte = np.array(Xte, np.float32), np.array(yte)

    # --- judge: depth threshold learned ONLY on generated data, plus logistic ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def scores(pred):
        tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum()); tn = int(((pred == 0) & (yte == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return (tp + tn) / len(yte), f1, (fn / (tp + fn) if tp + fn else 0.0)

    # threshold: midpoint between the class means on generated data
    thr = (Xtr[ytr == 0, DI].mean() + Xtr[ytr == 1, DI].mean()) / 2
    acc, f1, fa = scores((Xte[:, DI] < thr).astype(int))
    print(f"\nTRAINED ONLY ON GENERATED, TESTED ON REAL SHALLOW "
          f"(n={len(yte)}, shallow={int(yte.sum())}):")
    print(f"  RULE (threshold depth<{thr:+.3f} from generated): "
          f"acc {acc:.2f} | F1 {f1:.2f} | false-acc {fa:.1%}")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    acc, f1, fa = scores(clf.predict(sc.transform(Xte)))
    print(f"  ML (logistic):                        "
          f"acc {acc:.2f} | F1 {f1:.2f} | false-acc {fa:.1%}")

    print("\nMatch against the real data (mean depth):")
    print(f"  generated shallow {np.array(gen_shallow)[:,DI].mean():+.3f} vs "
          f"real shallow {Xte[yte==1,DI].mean():+.3f}")
    print(f"  generated valid   {np.array(gen_valid)[:,DI].mean():+.3f} vs "
          f"real valid   {Xte[yte==0,DI].mean():+.3f}")


if __name__ == "__main__":
    main()
