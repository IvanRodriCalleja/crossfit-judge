"""Can depth be judged from the FRONT with 2D image geometry?

Depth used to be computed from 3D world landmarks, whose orientation MediaPipe
estimates poorly from the front. But the vertical height in the IMAGE (hip relative to
knee, in pixels) is observable from the front. This probe reprocesses the recordings
computing depth from 2D image landmarks and checks, per view, whether it separates the
VALIDA / INVALIDA_PROFUNDIDAD classes better than the 3D version.
"""
import math
import os

import numpy as np

from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader
from squat.scripts.cache_recordings import TAKES, labels_for
from scipy.signal import find_peaks

REC = os.path.join(os.path.dirname(__file__), "..", "data", "grabaciones")
VIEWS = ["lateral", "frontal", "45deg"]
TRUNCATED = {("45deg", "4"), ("45deg", "5")}
LH, RH, LK, RK = 23, 24, 25, 26


def depth2d(xy):
    """Depth in the image plane: (hip_y − knee_y) / thigh length, with image y (grows
    downward). Crosses 0 when the hip reaches the knee."""
    hip_y = (xy[LH][1] + xy[RH][1]) / 2
    knee_y = (xy[LK][1] + xy[RK][1]) / 2
    thigh = (math.hypot(xy[LH][0] - xy[LK][0], xy[LH][1] - xy[LK][1])
             + math.hypot(xy[RH][0] - xy[RK][0], xy[RH][1] - xy[RK][1])) / 2
    return (hip_y - knee_y) / thigh if thigh > 1e-6 else 0.0


def knee2d(xy):
    def ang(a, b, c):
        v1 = (xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
        v2 = (xy[c][0] - xy[b][0], xy[c][1] - xy[b][1])
        n = math.hypot(*v1) * math.hypot(*v2)
        return math.degrees(math.acos(max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/n)))) if n else 0
    return (ang(LH, LK, 27) + ang(RH, RK, 28)) / 2


def main():
    reader = OpenCVVideoReader()
    est = MediaPipePoseEstimator(mode="image")
    d2, knees, y, persp, takes = [], [], [], [], []
    for v in VIEWS:
        for take in TAKES:
            if (v, take) in TRUNCATED:
                continue
            path = os.path.join(REC, v, f"{take}.mp4")
            if not os.path.exists(path):
                continue
            seq_d, seq_k = [], []
            last = None
            for frame in reader.frames(path):
                res = est.estimate_with_image(frame)
                if res is None:
                    if last is not None:
                        seq_d.append(last[0]); seq_k.append(last[1])
                    continue
                _, xy = res
                last = (depth2d(xy), knee2d(xy))
                seq_d.append(last[0]); seq_k.append(last[1])
            arr_d, arr_k = np.array(seq_d), np.array(seq_k)
            ks = np.convolve(arr_k, np.ones(5)/5, mode="same") if len(arr_k) >= 5 else arr_k
            bottoms, _ = find_peaks(-ks, prominence=20.0, distance=15)
            labels = labels_for(take, len(bottoms))
            for i, b in enumerate(bottoms):
                lo = (bottoms[i-1]+b)//2 if i > 0 else max(0, b-45)
                hi = (b+bottoms[i+1])//2 if i < len(bottoms)-1 else min(len(arr_d), b+45)
                d2.append(float(arr_d[lo:hi].max()))
                knees.append(float(arr_k[lo:hi].min()))
                y.append(labels[i]); persp.append(v); takes.append(take)
            print(f"  {v}/{take}: {len(bottoms)} reps")
    est.close()
    d2 = np.array(d2); knees = np.array(knees); y = np.array(y)
    persp = np.array(persp); takes = np.array(takes)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "frontal_2d.npz")
    np.savez(out, depth2d=d2, knee=knees, y=y, perspective=persp, take=takes)
    print(f"saved {os.path.abspath(out)}")

    print("\n== 2D image depth: does it separate VALIDA from shallow per view? ==")
    print(f"{'view':8} {'VALIDA (mean±sd)':>22} {'SHALLOW (mean±sd)':>22} {'|Δ|':>6}")
    for v in VIEWS:
        va = d2[(persp == v) & (y == "VALIDA")]
        sh = d2[(persp == v) & (y == "INVALIDA_PROFUNDIDAD")]
        if len(va) and len(sh):
            print(f"{v:8} {va.mean():>10.3f}±{va.std():.2f}      "
                  f"{sh.mean():>10.3f}±{sh.std():.2f}   {abs(va.mean()-sh.mean()):.3f}")
    # Cohen's d per view (separability)
    print("\nSeparability (Cohen's d; >0.8 = large):")
    for v in VIEWS:
        va = d2[(persp == v) & (y == "VALIDA")]
        sh = d2[(persp == v) & (y == "INVALIDA_PROFUNDIDAD")]
        if len(va) > 1 and len(sh) > 1:
            sp = np.sqrt(((len(va)-1)*va.var(ddof=1)+(len(sh)-1)*sh.var(ddof=1))/(len(va)+len(sh)-2))
            dc = abs(va.mean()-sh.mean())/sp if sp > 0 else float("nan")
            print(f"  {v:8}: d = {dc:.2f}")


if __name__ == "__main__":
    main()
