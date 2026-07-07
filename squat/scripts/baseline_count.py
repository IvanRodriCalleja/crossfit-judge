"""Signal-based counting baseline on RepCount-A: compares the predicted count with the
true one and reports MAE and OBO (fraction of videos with error <= 1).

Usage:  ./.venv/bin/python squat/scripts/baseline_count.py <repcount-a-path> [split] [limit]
"""
import sys

import numpy as np

from squat.application.extract_features import FeatureSequenceExtractor
from squat.domain.features import SquatFeatureExtractor
from squat.infrastructure.counter.signal_counter import SignalRepCounter
from squat.infrastructure.datasets.repcount import RepCountDataset
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader


def main(root: str, split: str = "test", limit: int | None = None) -> None:
    ds = RepCountDataset(root, splits=(split,))
    samples = list(ds.samples())
    if limit:
        samples = samples[:limit]

    reader = OpenCVVideoReader()
    extractor = SquatFeatureExtractor()
    errors, obo = [], 0

    for i, s in enumerate(samples, 1):
        fps = reader.fps(s.video_path)
        estimator = MediaPipePoseEstimator(mode="image")  # one per video (no cross-state)
        seq = FeatureSequenceExtractor(reader, estimator, extractor).extract(s.video_path)
        estimator.close()
        pred = SignalRepCounter(fps=fps).count(seq).count
        err = abs(pred - s.count)
        errors.append(err)
        obo += int(err <= 1)
        print(f"[{i}/{len(samples)}] {s.video_path.split('/')[-1]:16} pred={pred:2d} real={s.count:2d} |err|={err}")

    n = len(samples)
    print(f"\nSignal baseline — split={split}, n={n}")
    print(f"  MAE = {np.mean(errors):.2f}")
    print(f"  OBO = {obo / n:.1%}  (videos with |error| <= 1)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: baseline_count.py <repcount-a-path> [split] [limit]")
        sys.exit(1)
    root = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(root, split, limit)
