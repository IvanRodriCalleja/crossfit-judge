"""Cache our own labelled recordings: extract pose/features, detect each rep from the
signal and assign its validity label according to each take's script.

Script (per view, takes 1-5):
  1: alternating every ~5 s, starting VALIDA -> even=VALIDA, odd=INVALIDA_PROFUNDIDAD
  2: 20 VALIDA            3: 16 INVALIDA_LOCKOUT
  4: 25 INVALIDA_PROFUNDIDAD   5: 22 VALIDA
(the opening clap doesn't bend the knee, so it isn't counted as a rep)

Saves squat/data/cache/grabaciones/{persp}_{take}.npz: features (T,6),
bottoms (frame of each rep), labels (validity per rep) + manifest.json.

Usage:  ./.venv/bin/python squat/scripts/cache_recordings.py
"""
import json
import os

import numpy as np
from scipy.signal import find_peaks

from squat.application.extract_features import FeatureSequenceExtractor
from squat.domain.features import FEATURE_NAMES, SquatFeatureExtractor
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader

REC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "grabaciones")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones")
PERSPECTIVES = ["lateral", "frontal", "45deg"]
# take -> (mode, fixed_label, expected)
TAKES = {
    "1": ("alt", None, None),
    "2": ("fixed", "VALIDA", 20),
    "3": ("fixed", "INVALIDA_LOCKOUT", 16),
    "4": ("fixed", "INVALIDA_PROFUNDIDAD", 25),
    "5": ("fixed", "VALIDA", 22),
}


def detect_bottoms(feats, fps):
    knee = np.array([(f.knee_angle_left + f.knee_angle_right) / 2 for f in feats], float)
    if len(knee) >= 5:
        knee = np.convolve(knee, np.ones(5) / 5, mode="same")
    bottoms, _ = find_peaks(-knee, prominence=20.0, distance=15)
    return bottoms


def labels_for(take, n):
    mode, fixed, _ = TAKES[take]
    if mode == "fixed":
        return [fixed] * n
    # alternating: even -> VALIDA, odd -> INVALIDA_PROFUNDIDAD
    return ["VALIDA" if i % 2 == 0 else "INVALIDA_PROFUNDIDAD" for i in range(n)]


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    reader = OpenCVVideoReader()
    estimator = MediaPipePoseEstimator(mode="image")
    seq = FeatureSequenceExtractor(reader, estimator, SquatFeatureExtractor())

    manifest = []
    for persp in PERSPECTIVES:
        for take, (mode, fixed, expected) in TAKES.items():
            path = os.path.join(REC_DIR, persp, f"{take}.mp4")
            if not os.path.exists(path):
                continue
            fps = reader.fps(path)
            feats = seq.extract(path)
            arr = np.array([f.to_list() for f in feats], dtype=np.float32)
            bottoms = detect_bottoms(feats, fps)
            labels = labels_for(take, len(bottoms))
            key = f"{persp}_{take}"
            np.savez(os.path.join(CACHE_DIR, key + ".npz"),
                     features=arr, bottoms=bottoms.astype(np.int32),
                     labels=np.array(labels, dtype="<U24"))
            manifest.append({"key": key, "perspective": persp, "take": take,
                             "n_frames": int(arr.shape[0]), "count": int(len(bottoms)),
                             "expected": expected, "fps": fps, "source": "grabaciones"})
            tag = f" (expected {expected})" if expected else ""
            print(f"{key}: {len(bottoms)} reps{tag}  frames={arr.shape[0]}")
    estimator.close()

    with open(os.path.join(CACHE_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "labels":
                   ["VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"],
                   "items": manifest}, fh, indent=2)
    print(f"\ncached {len(manifest)} recordings in {os.path.abspath(CACHE_DIR)}")


if __name__ == "__main__":
    main()
