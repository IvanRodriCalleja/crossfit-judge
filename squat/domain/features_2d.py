"""Kinematic features in the IMAGE PLANE (normalized 2D landmarks).

Same semantics and column order as the 3D version (features.py), but computed from
the 2D image position instead of the world landmarks. MediaPipe misjudges 3D
orientation away from the lateral view, whereas vertical height in the image is
observable from any angle, so 2D depth still separates reps seen from the front.

Layout (matches FEATURE_NAMES): [knee_l, knee_r, hip_l, hip_r, hip_knee_depth, vis].
"""
from __future__ import annotations

import math

from .pose import Joint


def _angle2d(a, b, c) -> float:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n <= 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def features_2d(xy: dict[int, tuple[float, float]]) -> list[float]:
    """The 6 image-plane features from normalized 2D landmarks {index: (x, y)};
    y grows downward."""
    def p(j):
        return xy[int(j)]
    knee_l = _angle2d(p(Joint.LEFT_HIP), p(Joint.LEFT_KNEE), p(Joint.LEFT_ANKLE))
    knee_r = _angle2d(p(Joint.RIGHT_HIP), p(Joint.RIGHT_KNEE), p(Joint.RIGHT_ANKLE))
    hip_l = _angle2d(p(Joint.LEFT_SHOULDER), p(Joint.LEFT_HIP), p(Joint.LEFT_KNEE))
    hip_r = _angle2d(p(Joint.RIGHT_SHOULDER), p(Joint.RIGHT_HIP), p(Joint.RIGHT_KNEE))
    hip_y = (p(Joint.LEFT_HIP)[1] + p(Joint.RIGHT_HIP)[1]) / 2
    knee_y = (p(Joint.LEFT_KNEE)[1] + p(Joint.RIGHT_KNEE)[1]) / 2
    thigh = (math.hypot(p(Joint.LEFT_HIP)[0] - p(Joint.LEFT_KNEE)[0],
                        p(Joint.LEFT_HIP)[1] - p(Joint.LEFT_KNEE)[1])
             + math.hypot(p(Joint.RIGHT_HIP)[0] - p(Joint.RIGHT_KNEE)[0],
                          p(Joint.RIGHT_HIP)[1] - p(Joint.RIGHT_KNEE)[1])) / 2
    depth = (hip_y - knee_y) / thigh if thigh > 1e-6 else 0.0
    return [knee_l, knee_r, hip_l, hip_r, depth, 1.0]
