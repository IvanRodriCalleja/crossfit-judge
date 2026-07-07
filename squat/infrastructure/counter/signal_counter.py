"""Signal-based rep counter (baseline, no machine learning).

Each squat is a valley in the knee angle (standing ~180deg, bottom ~70-90deg).
Those valleys are found with `find_peaks` on the inverted signal, using a minimum
prominence (rejects noise) and a minimum spacing between reps. Serves as a
reference to compare against the model and to auto-annotate videos.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from squat.domain.ports import RepCounter
from squat.domain.sequence import RepResult


class SignalRepCounter(RepCounter):
    def __init__(self, prominence: float = 25.0, min_distance_s: float = 0.4,
                 fps: float = 30.0, smooth: int = 5):
        self.prominence = prominence
        self.min_distance_s = min_distance_s
        self.fps = fps if fps > 0 else 30.0
        self.smooth = smooth

    def count(self, features: list) -> RepResult:
        if not features:
            return RepResult(0, [])
        knee = np.array(
            [(f.knee_angle_left + f.knee_angle_right) / 2 for f in features],
            dtype=float,
        )
        if self.smooth > 1 and len(knee) >= self.smooth:
            kernel = np.ones(self.smooth) / self.smooth
            knee = np.convolve(knee, kernel, mode="same")
        distance = max(1, int(self.min_distance_s * self.fps))
        bottoms, _ = find_peaks(-knee, prominence=self.prominence, distance=distance)
        return RepResult(count=int(len(bottoms)), bottoms=bottoms.tolist())
