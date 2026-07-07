"""Analyze one of our recordings: extract pose, detect each rep from the knee-angle
signal and report the detected count (to check against the known total and tune the
segmentation before labelling).

Usage:  ./.venv/bin/python squat/scripts/analyze_recording.py <video> [expected] [prominence] [distance]
"""
import sys

import numpy as np
from scipy.signal import find_peaks

from squat.application.extract_features import FeatureSequenceExtractor
from squat.domain.features import SquatFeatureExtractor
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader


def main(path, expected=None, prominence=20.0, distance=15):
    reader = OpenCVVideoReader()
    fps = reader.fps(path)
    estimator = MediaPipePoseEstimator(mode="image")
    feats = FeatureSequenceExtractor(reader, estimator, SquatFeatureExtractor()).extract(path)
    estimator.close()

    knee = np.array([(f.knee_angle_left + f.knee_angle_right) / 2 for f in feats], float)
    if len(knee) >= 5:
        knee = np.convolve(knee, np.ones(5) / 5, mode="same")
    bottoms, _ = find_peaks(-knee, prominence=prominence, distance=distance)

    spac = np.diff(bottoms) / fps if len(bottoms) > 1 else np.array([])
    print(f"{path.split('/')[-2]}/{path.split('/')[-1]} | {len(feats)}f @ {fps:.0f}fps "
          f"({len(feats)/fps:.0f}s)")
    print(f"  detected: {len(bottoms)}" + (f"  (expected: {expected})" if expected else ""))
    if len(spac):
        print(f"  rep spacing: mean {spac.mean():.1f}s  min {spac.min():.1f}s  max {spac.max():.1f}s")
    lo, hi = knee.min(), knee.max()
    print(f"  knee angle: {lo:.0f}°-{hi:.0f}°  (deeper = smaller)")


if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else None,
         float(sys.argv[3]) if len(sys.argv) > 3 else 20.0,
         int(sys.argv[4]) if len(sys.argv) > 4 else 15)
