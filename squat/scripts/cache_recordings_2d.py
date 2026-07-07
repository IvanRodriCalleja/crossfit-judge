"""Cache the recordings with per-frame 2D IMAGE features (plus the per-rep validity
label). This is the substrate for the multi-view ML validator (F5): 2D depth also
separates the classes outside the lateral view.

Saves squat/data/cache/grabaciones_2d/{persp}_{take}.npz: features (T,6) in 2D,
bottoms, labels; + manifest.json. Labels follow each take's script (cache_recordings).
"""
import json
import os

import numpy as np
from scipy.signal import find_peaks

from squat.domain.features import FEATURE_NAMES
from squat.domain.features_2d import features_2d
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader
from squat.scripts.cache_recordings import TAKES, labels_for

REC = os.path.join(os.path.dirname(__file__), "..", "data", "grabaciones")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones_2d")
PERSPECTIVES = ["lateral", "frontal", "45deg"]


def detect_bottoms(feats2d, fps):
    knee = np.array([(f[0] + f[1]) / 2 for f in feats2d], float)  # cols 0,1 = knees
    if len(knee) >= 5:
        knee = np.convolve(knee, np.ones(5) / 5, mode="same")
    bottoms, _ = find_peaks(-knee, prominence=20.0, distance=15)
    return bottoms


def main():
    os.makedirs(OUT, exist_ok=True)
    reader = OpenCVVideoReader()
    est = MediaPipePoseEstimator(mode="image")
    manifest = []
    for persp in PERSPECTIVES:
        for take in TAKES:
            path = os.path.join(REC, persp, f"{take}.mp4")
            if not os.path.exists(path):
                continue
            seq, last = [], None
            for frame in reader.frames(path):
                res = est.estimate_with_image(frame)
                if res is None:
                    if last is not None:
                        seq.append(last)
                    continue
                _, xy = res
                last = features_2d(xy)
                seq.append(last)
            arr = np.array(seq, dtype=np.float32)
            bottoms = detect_bottoms(seq, reader.fps(path))
            labels = labels_for(take, len(bottoms))
            key = f"{persp}_{take}"
            np.savez(os.path.join(OUT, key + ".npz"),
                     features=arr, bottoms=bottoms.astype(np.int32),
                     labels=np.array(labels, dtype="<U24"))
            manifest.append({"key": key, "perspective": persp, "take": take,
                             "count": int(len(bottoms)), "source": "grabaciones_2d"})
            print(f"{key}: {len(bottoms)} reps  frames={arr.shape[0]}")
    est.close()
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "items": manifest}, fh, indent=2)
    print(f"\ncached {len(manifest)} recordings (2D) in {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
