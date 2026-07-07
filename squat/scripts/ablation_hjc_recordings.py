"""F4b — HJC ablation BY VIEW on our own recordings (labeled frontal).

Reprocesses the recordings computing per-rep depth with and without the HJC
correction, and compares the depth judge in each view (lateral, frontal, 45°).
Answers: does the hip-center correction help in the FRONTAL view, where depth breaks
down? Threshold calibrated on lateral and applied to each view.
"""
import os

import numpy as np

from squat.application.rep_features import extract_rep_features
from squat.domain.features import SquatFeatureExtractor
from squat.domain.hjc import correct_hip_center
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader
from squat.scripts.cache_recordings import TAKES, detect_bottoms, labels_for

REC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "grabaciones")
PERSPECTIVES = ["lateral", "frontal", "45deg"]
TRUNCATED = {("45deg", "4"), ("45deg", "5")}


def judge(depth, is_shallow, thr):
    pred = depth < thr
    tp = int((pred & is_shallow).sum()); fp = int((pred & ~is_shallow).sum())
    fn = int((~pred & is_shallow).sum()); tn = int((~pred & ~is_shallow).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(depth)
    fa = fn / (tp + fn) if tp + fn else 0.0
    return dict(acc=acc, f1=f1, fa=fa)


def collect():
    reader = OpenCVVideoReader()
    est = MediaPipePoseEstimator(mode="image")
    fx = SquatFeatureExtractor()
    rows = {"base": [], "hjc": []}
    persp_all, lab_all = [], []
    for persp in PERSPECTIVES:
        for take in TAKES:
            if (persp, take) in TRUNCATED:
                continue
            path = os.path.join(REC_DIR, persp, f"{take}.mp4")
            if not os.path.exists(path):
                continue
            base_seq, hjc_seq = [], []
            last_b = last_h = None
            for frame in reader.frames(path):
                skel = est.estimate(frame)
                if skel is None:
                    if last_b is not None:
                        base_seq.append(last_b); hjc_seq.append(last_h)
                    continue
                last_b = fx.extract(skel).to_list()
                last_h = fx.extract(correct_hip_center(skel)).to_list()
                base_seq.append(last_b); hjc_seq.append(last_h)
            base_arr = np.array(base_seq, np.float32)
            hjc_arr = np.array(hjc_seq, np.float32)
            # bottoms from the base signal (same segmentation for both variants)
            from squat.domain.features import SquatFeatures
            bottoms = detect_bottoms([SquatFeatures(*r) for r in base_seq], reader.fps(path))
            labels = labels_for(take, len(bottoms))
            rb = extract_rep_features(base_arr, bottoms)
            rh = extract_rep_features(hjc_arr, bottoms)
            for a, b, lab in zip(rb, rh, labels):
                rows["base"].append(a.depth_at_bottom)
                rows["hjc"].append(b.depth_at_bottom)
                persp_all.append(persp); lab_all.append(lab)
            print(f"  {persp}/{take}: {len(bottoms)} reps")
    est.close()
    return (np.array(rows["base"]), np.array(rows["hjc"]),
            np.array(persp_all), np.array(lab_all))


def best_threshold(depth, is_shallow):
    best = (None, -1.0)
    for thr in np.quantile(depth, np.linspace(0.05, 0.95, 60)):
        m = judge(depth, is_shallow, thr)
        bal = m["acc"]
        if bal > best[1]:
            best = (float(thr), bal)
    return best[0]


def main():
    base, hjc, persp, y = collect()
    is_shallow = y == "INVALIDA_PROFUNDIDAD"
    keep = np.isin(y, ["VALIDA", "INVALIDA_PROFUNDIDAD"])
    base, hjc, persp, is_shallow = base[keep], hjc[keep], persp[keep], is_shallow[keep]

    print("\n== HJC ABLATION BY VIEW (recordings) — depth judge ==")
    for name, depth in [("NO HJC", base), ("WITH HJC", hjc)]:
        lat = persp == "lateral"
        thr = best_threshold(depth[lat], is_shallow[lat])  # calibrated on lateral
        print(f"\n{name} (lateral threshold depth<{thr:+.3f}):")
        for v in PERSPECTIVES:
            m = persp == v
            if m.sum() == 0:
                continue
            r = judge(depth[m], is_shallow[m], thr)
            print(f"  {v:8} acc {r['acc']:.2f} | F1 {r['f1']:.2f} | false-acc {r['fa']:5.1%}")

    print("\nMean depth by view and class (no -> with HJC):")
    for v in PERSPECTIVES:
        for lab, cls in [("VALIDA", "VALIDA"), ("SHALLOW", "INVALIDA_PROFUNDIDAD")]:
            m = (persp == v) & (is_shallow == (cls == "INVALIDA_PROFUNDIDAD"))
            if m.sum():
                print(f"  {v:8} {lab:8}: {base[m].mean():+.3f} -> {hjc[m].mean():+.3f}")


if __name__ == "__main__":
    main()
