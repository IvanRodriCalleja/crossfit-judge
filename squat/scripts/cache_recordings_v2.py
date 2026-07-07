"""Cache the NEW recordings (multi-session) driven by a config file instead of a
hardcoded script. Built for the protocol that breaks the class-take confound: each
take can mix classes (cyclic pattern) and there are several takes/sessions, so
validation can group BY TAKE/SESSION without leakage.

JSON config (see recordings_v2.example.json):
{
  "subject": "ivan",
  "root": "squat/data/grabaciones_v2",
  "filename": "{name}_{camera}.mp4",          # per-camera filename template
  "cameras": ["lateral", "frontal", "45deg"],
  "takes": [
    {"name": "mix1", "session": "s1", "expected": 24,
     "pattern": {"type": "cycle", "classes": ["VALIDA","INVALIDA_PROFUNDIDAD","INVALIDA_LOCKOUT"]}},
    {"name": "lockout1", "session": "s1", "expected": 15,
     "pattern": {"type": "fixed", "class": "INVALIDA_LOCKOUT"}}
  ]
}

Pattern types: "cycle" (classes[i % len]), "fixed" (one class), "alt" (even/odd,
compatible with the old recordings). Leakage-free CV group = (subject, session).

Usage:  ./.venv/bin/python squat/scripts/cache_recordings_v2.py <config.json>
"""
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

from squat.application.extract_features import FeatureSequenceExtractor
from squat.domain.features import FEATURE_NAMES, SquatFeatureExtractor
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones_v2")
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def detect_bottoms(feats, prominence=20.0, distance=15):
    knee = np.array([(f.knee_angle_left + f.knee_angle_right) / 2 for f in feats], float)
    if len(knee) >= 5:
        knee = np.convolve(knee, np.ones(5) / 5, mode="same")
    bottoms, _ = find_peaks(-knee, prominence=prominence, distance=distance)
    return bottoms


def pattern_labels(pattern, n):
    """Validity label for the n reps according to the take's pattern."""
    t = pattern["type"]
    if t == "fixed":
        return [pattern["class"]] * n
    if t == "cycle":
        cs = pattern["classes"]
        return [cs[i % len(cs)] for i in range(n)]
    if t == "alt":  # even=VALIDA, odd=INVALIDA_PROFUNDIDAD (old recordings)
        return ["VALIDA" if i % 2 == 0 else "INVALIDA_PROFUNDIDAD" for i in range(n)]
    raise ValueError(f"unknown pattern: {t}")


def main(config_path):
    cfg = json.load(open(config_path))
    subject = cfg.get("subject", "ivan")
    root = os.path.join(_ROOT, cfg["root"])
    fname = cfg.get("filename", "{name}_{camera}.mp4")
    cameras = cfg["cameras"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    reader = OpenCVVideoReader()
    estimator = MediaPipePoseEstimator(mode="image")
    seq = FeatureSequenceExtractor(reader, estimator, SquatFeatureExtractor())

    manifest, warnings = [], []
    for take in cfg["takes"]:
        for cam in cameras:
            path = os.path.join(root, fname.format(name=take["name"], camera=cam))
            if not os.path.exists(path):
                warnings.append(f"missing {path}")
                continue
            feats = seq.extract(path)
            arr = np.array([f.to_list() for f in feats], dtype=np.float32)
            bottoms = detect_bottoms(feats)
            n = len(bottoms)
            labels = pattern_labels(take["pattern"], n)
            key = f"{subject}_{take['name']}_{cam}"
            np.savez(os.path.join(CACHE_DIR, key + ".npz"),
                     features=arr, bottoms=bottoms.astype(np.int32),
                     labels=np.array(labels, dtype="<U24"))
            exp = take.get("expected")
            manifest.append({
                "key": key, "subject": subject, "session": take.get("session", "s1"),
                "take": take["name"], "perspective": cam,
                "group": f"{subject}:{take.get('session', 's1')}",
                "count": n, "expected": exp, "n_frames": int(arr.shape[0]),
                "pattern": take["pattern"], "source": "grabaciones_v2",
            })
            flag = "" if exp in (None, n) else f"  ⚠ EXPECTED {exp}"
            print(f"{key}: {n} reps{flag}  frames={arr.shape[0]}")
    estimator.close()

    with open(os.path.join(CACHE_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"feature_names": FEATURE_NAMES,
                   "labels": ["VALIDA", "INVALIDA_PROFUNDIDAD", "INVALIDA_LOCKOUT"],
                   "items": manifest}, fh, indent=2)
    print(f"\ncached {len(manifest)} recordings in {os.path.abspath(CACHE_DIR)}")
    if warnings:
        print("WARNINGS:"); [print(" ", w) for w in warnings]
    mism = [m for m in manifest if m["expected"] not in (None, m["count"])]
    if mism:
        print("\n⚠ Counts that do NOT match expected (check segmentation before "
              "trusting the labels):")
        for m in mism:
            print(f"  {m['key']}: detected {m['count']} vs expected {m['expected']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: cache_recordings_v2.py <config.json>")
    main(sys.argv[1])
