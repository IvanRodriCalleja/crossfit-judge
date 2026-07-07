"""Precompute and store a dataset's features (so pose extraction isn't repeated on
every training run). Supports several datasets and merges the manifest.

If a video has no rep annotation (e.g. Exercise Recognition), it is auto-annotated
with the signal baseline over its own features (weak label).

Usage:
  ./.venv/bin/python squat/scripts/cache_features.py repcount <repcount-a-path> [limit]
  ./.venv/bin/python squat/scripts/cache_features.py exrec    <exrec-path>      [limit]
"""
import json
import os
import sys

import numpy as np

from squat.application.extract_features import FeatureSequenceExtractor
from squat.domain.features import FEATURE_NAMES, SquatFeatureExtractor
from squat.infrastructure.counter.signal_counter import SignalRepCounter
from squat.infrastructure.datasets.exercise_recognition import ExerciseRecognitionDataset
from squat.infrastructure.datasets.repcount import RepCountDataset
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "repcount")
MANIFEST = os.path.join(CACHE_DIR, "manifest.json")


def build_dataset(name: str, root: str):
    if name == "repcount":
        return RepCountDataset(root)
    if name == "exrec":
        return ExerciseRecognitionDataset(root)
    raise ValueError(f"unknown dataset: {name}")


def load_manifest_items() -> dict:
    if os.path.exists(MANIFEST):
        return {it["key"]: it for it in json.load(open(MANIFEST))["items"]}
    return {}


def main(name: str, root: str, limit: int | None = None) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    samples = list(build_dataset(name, root).samples())
    if limit:
        samples = samples[:limit]

    reader = OpenCVVideoReader()
    extractor = SquatFeatureExtractor()
    estimator = MediaPipePoseEstimator(mode="image")
    seq_extractor = FeatureSequenceExtractor(reader, estimator, extractor)

    items = load_manifest_items()
    for i, s in enumerate(samples, 1):
        feats = seq_extractor.extract(s.video_path)
        arr = np.array([f.to_list() for f in feats], dtype=np.float32)
        fps = reader.fps(s.video_path)
        if s.reps:                                   # real annotation
            reps = np.array([[r.start, r.end] for r in s.reps], dtype=np.int32)
            count = s.count
        else:                                        # auto-annotation (baseline)
            res = SignalRepCounter(fps=fps).count(feats)
            reps = np.array([[b, b] for b in res.bottoms], dtype=np.int32)
            count = res.count
        key = f"{s.source}_{s.split}_{os.path.basename(s.video_path).rsplit('.', 1)[0]}"
        np.savez(os.path.join(CACHE_DIR, key + ".npz"), features=arr, reps=reps)
        items[key] = {"key": key, "split": s.split, "subject": s.subject,
                      "count": int(count), "n_frames": int(arr.shape[0]),
                      "fps": fps, "source": s.source}
        print(f"[{i}/{len(samples)}] {key}  frames={arr.shape[0]} reps={count}")
    estimator.close()

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "items": list(items.values())}, fh, indent=2)
    print(f"\nmanifest: {len(items)} videos total in {os.path.abspath(CACHE_DIR)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: cache_features.py <repcount|exrec> <path> [limit]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
