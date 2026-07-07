"""Density target for counting. Each rep is a Gaussian centred on its midpoint; the
model learns to reproduce that curve and the rep count is recovered by integrating
it.

The Gaussians have height 1 (unnormalised) and we divide by their analytic integral
(sigma·sqrt(2π)) to recover the count, so the peaks give a strong training gradient
(the usual density-counting approach)."""
from __future__ import annotations

import math

import numpy as np


def rep_centers(reps) -> list[int]:
    """Centre (mid-frame) of each rep from (start, end) pairs."""
    return [(int(s) + int(e)) // 2 for s, e in reps]


def build_density_target(centers: list[int], n_frames: int, sigma: float = 8.0) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float32)
    density = np.zeros(n_frames, dtype=np.float32)
    for c in centers:
        density += np.exp(-0.5 * ((t - c) / sigma) ** 2)
    return density


def density_to_count(density: np.ndarray, sigma: float = 8.0) -> float:
    return float(np.sum(density)) / (sigma * math.sqrt(2.0 * math.pi))


def count_peaks(density: np.ndarray, height: float = 0.5, min_distance: int = 20) -> int:
    """Count reps as peaks of the predicted density. More robust than the integral
    against somewhat wide peaks or a raised baseline."""
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(density, height=height, distance=min_distance)
    return int(len(peaks))
