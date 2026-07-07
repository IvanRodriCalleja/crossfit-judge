"""Cache pose + features per crop for the Fitness-AQA shallow subset.

For each labelled crop it runs MediaPipe and stores:
  - the 6 squat features (world landmarks, identical to the pipeline's),
  - a VIEW descriptor (2D horizontal separation of shoulders/hips normalised by torso
    length; ~0 in lateral view, larger from the front or the back),
  - the label (0 no error / 1 shallow depth), the subject and the official split.

Crops with no detectable pose are recorded (pose_ok=False) to report coverage.
Output: squat/data/cache/fitness_aqa/shallow.npz (gitignored).
"""
import math
import os
import time

import cv2
import numpy as np

from squat.domain.features import FEATURE_NAMES, SquatFeatureExtractor
from squat.domain.features_2d import features_2d
from squat.domain.hjc import correct_hip_center
from squat.domain.pose import Joint
from squat.infrastructure.datasets.fitness_aqa import FitnessAQAShallow
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "fitness_aqa")


def view_score(xy: dict[int, tuple[float, float]]) -> float:
    """Mean horizontal separation of shoulders and hips (2D) normalised by torso
    length. Small in lateral view (the two sides overlap), large from the front or
    the back."""
    def sep(a, b):
        return abs(xy[int(a)][0] - xy[int(b)][0])
    sh_mid = ((xy[11][0] + xy[12][0]) / 2, (xy[11][1] + xy[12][1]) / 2)
    hip_mid = ((xy[23][0] + xy[24][0]) / 2, (xy[23][1] + xy[24][1]) / 2)
    torso = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
    if torso <= 1e-6:
        return float("nan")
    return (sep(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
            + sep(Joint.LEFT_HIP, Joint.RIGHT_HIP)) / 2 / torso


def main():
    ds = FitnessAQAShallow()
    est = MediaPipePoseEstimator(mode="image")
    fx = SquatFeatureExtractor()
    crops = ds.all_crops()
    print(f"{len(crops)} labelled crops; extracting pose...")

    keys, subj, split, labels = [], [], [], []
    feats, feats_hjc, feats2d, views, pose_ok = [], [], [], [], []
    t0 = time.time()
    for i, c in enumerate(crops):
        img = cv2.imread(c.path)
        res = None if img is None else est.estimate_with_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        keys.append(c.key); subj.append(c.subject); split.append(c.split)
        labels.append(c.label)
        if res is None:
            feats.append([np.nan] * len(FEATURE_NAMES))
            feats_hjc.append([np.nan] * len(FEATURE_NAMES))
            feats2d.append([np.nan] * len(FEATURE_NAMES)); views.append(np.nan)
            pose_ok.append(False)
        else:
            skel, xy = res
            feats.append(fx.extract(skel).to_list())
            feats_hjc.append(fx.extract(correct_hip_center(skel)).to_list())
            feats2d.append(features_2d(xy))
            views.append(view_score(xy)); pose_ok.append(True)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(crops)}  ({time.time()-t0:.0f}s)")
    est.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "shallow.npz")
    np.savez(
        out,
        key=np.array(keys), subject=np.array(subj), split=np.array(split),
        label=np.array(labels, dtype=np.int64),
        features=np.array(feats, dtype=np.float32),
        features_hjc=np.array(feats_hjc, dtype=np.float32),
        features_2d=np.array(feats2d, dtype=np.float32),
        view=np.array(views, dtype=np.float32),
        pose_ok=np.array(pose_ok, dtype=bool),
        feature_names=np.array(FEATURE_NAMES),
    )
    ok = int(np.sum(pose_ok))
    print(f"\nsaved {out}")
    print(f"pose detected in {ok}/{len(crops)} ({ok/len(crops):.1%})")


if __name__ == "__main__":
    main()
