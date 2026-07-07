"""Per-rep features for REP/NO-REP validation.

The two rulebook criteria expressed as geometric measures over each rep's window
(between the midpoints to the neighbouring reps):

- Depth ("breaking parallel"): at the bottom the hip must drop below the knee.
  Measured by `depth_at_bottom` (max of the hip-knee depth signal in the window;
  ~-1 standing, crosses ~0 when reaching parallel).
- Lockout (hip extension at the top): at the end the hip must extend almost fully.
  Measured by `hip_ext_at_top` (max hip angle in the window; ~180° means full
  extension).
`knee_min` (min knee angle) is added as an auxiliary depth measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Column indices per squat.domain.features.FEATURE_NAMES:
# [knee_l, knee_r, hip_l, hip_r, hip_knee_depth, visibility]
_KNEE = (0, 1)
_HIP = (2, 3)
_DEPTH = 4


@dataclass(frozen=True)
class RepFeatures:
    depth_at_bottom: float   # max depth at the bottom (crosses ~0 when breaking parallel)
    knee_min: float          # min knee angle (smaller = deeper)
    hip_ext_at_top: float    # max hip extension (~180° = full lockout)

    def to_list(self) -> list[float]:
        return [self.depth_at_bottom, self.knee_min, self.hip_ext_at_top]


FEATURE_NAMES = ["depth_at_bottom", "knee_min", "hip_ext_at_top"]


def extract_rep_features(feats: np.ndarray, bottoms) -> list[RepFeatures]:
    knee = (feats[:, _KNEE[0]] + feats[:, _KNEE[1]]) / 2.0
    hip = (feats[:, _HIP[0]] + feats[:, _HIP[1]]) / 2.0
    depth = feats[:, _DEPTH]
    T = len(feats)
    bottoms = list(int(b) for b in bottoms)

    out = []
    for i, b in enumerate(bottoms):
        lo = (bottoms[i - 1] + b) // 2 if i > 0 else max(0, b - 45)
        hi = (b + bottoms[i + 1]) // 2 if i < len(bottoms) - 1 else min(T, b + 45)
        lo, hi = max(0, lo), min(T, max(hi, lo + 1))
        out.append(RepFeatures(
            depth_at_bottom=float(depth[lo:hi].max()),
            knee_min=float(knee[lo:hi].min()),
            hip_ext_at_top=float(hip[lo:hi].max()),
        ))
    return out
