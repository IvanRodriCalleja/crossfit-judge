"""Rich per-rep features for the ML validator.

Where the rule-based judge uses 3 measures (depth, min knee, hip extension), here
each rep is described by a wider vector: signal statistics over the window (min, max,
mean, range), values at key instants (bottom and highest point) and left-right
asymmetry. The point is to give the model information a fixed threshold ignores, so
it can learn to judge more robustly than a rule, even across different views.
"""
from __future__ import annotations

import numpy as np

# base feature columns: [knee_l, knee_r, hip_l, hip_r, depth, visibility]
_KL, _KR, _HL, _HR, _DEPTH, _VIS = 0, 1, 2, 3, 4, 5

RICH_NAMES = [
    "depth_max", "depth_min", "depth_mean", "depth_range",
    "knee_min", "knee_max", "knee_mean", "knee_at_bottom",
    "hip_max", "hip_min", "hip_mean", "hip_at_bottom",
    "knee_lr_asym", "hip_lr_asym", "vis_mean", "n_frames_norm",
]


def rich_features(window: np.ndarray) -> np.ndarray:
    """Rich feature vector from a rep window (T, 6)."""
    knee = (window[:, _KL] + window[:, _KR]) / 2.0
    hip = (window[:, _HL] + window[:, _HR]) / 2.0
    depth = window[:, _DEPTH]
    bottom = int(np.argmax(depth))  # deepest frame
    return np.array([
        depth.max(), depth.min(), depth.mean(), depth.max() - depth.min(),
        knee.min(), knee.max(), knee.mean(), knee[bottom],
        hip.max(), hip.min(), hip.mean(), hip[bottom],
        float(np.mean(np.abs(window[:, _KL] - window[:, _KR]))),
        float(np.mean(np.abs(window[:, _HL] - window[:, _HR]))),
        float(window[:, _VIS].mean()),
        min(len(window) / 90.0, 2.0),  # normalised duration (~cadence), clipped
    ], dtype=np.float32)
