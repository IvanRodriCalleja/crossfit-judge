"""Squat kinematic features derived from the pose: turns the skeleton into
meaningful quantities (angles and depth) rather than feeding raw points to the model.

Initial set of 6 features; refined through experimentation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .pose import Joint, Landmark, Skeleton
from .ports import FeatureExtractor

FEATURE_NAMES = [
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "hip_knee_depth",
    "visibility",
]


def _angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Angle in degrees at vertex b, between segments b->a and b->c."""
    bax, bay, baz = a.x - b.x, a.y - b.y, a.z - b.z
    bcx, bcy, bcz = c.x - b.x, c.y - b.y, c.z - b.z
    dot = bax * bcx + bay * bcy + baz * bcz
    na = math.sqrt(bax * bax + bay * bay + baz * baz)
    nc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if na == 0 or nc == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cos))


@dataclass(frozen=True)
class SquatFeatures:
    knee_angle_left: float
    knee_angle_right: float
    hip_angle_left: float
    hip_angle_right: float
    hip_knee_depth: float   # vertical hip-to-knee, normalized by thigh length
    visibility: float

    def to_list(self) -> list[float]:
        return [
            self.knee_angle_left, self.knee_angle_right,
            self.hip_angle_left, self.hip_angle_right,
            self.hip_knee_depth, self.visibility,
        ]


class SquatFeatureExtractor(FeatureExtractor):
    """Extracts the 6 squat features from a skeleton."""

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def extract(self, s: Skeleton) -> SquatFeatures:
        J = Joint
        knee_l = _angle(s.joint(J.LEFT_HIP), s.joint(J.LEFT_KNEE), s.joint(J.LEFT_ANKLE))
        knee_r = _angle(s.joint(J.RIGHT_HIP), s.joint(J.RIGHT_KNEE), s.joint(J.RIGHT_ANKLE))
        hip_l = _angle(s.joint(J.LEFT_SHOULDER), s.joint(J.LEFT_HIP), s.joint(J.LEFT_KNEE))
        hip_r = _angle(s.joint(J.RIGHT_SHOULDER), s.joint(J.RIGHT_HIP), s.joint(J.RIGHT_KNEE))

        # Depth: vertical (y) hip-to-knee gap, normalized by thigh length so it is
        # independent of athlete size.
        hip_y = (s.joint(J.LEFT_HIP).y + s.joint(J.RIGHT_HIP).y) / 2
        knee_y = (s.joint(J.LEFT_KNEE).y + s.joint(J.RIGHT_KNEE).y) / 2
        thigh = (
            s.joint(J.LEFT_HIP).distance_to(s.joint(J.LEFT_KNEE))
            + s.joint(J.RIGHT_HIP).distance_to(s.joint(J.RIGHT_KNEE))
        ) / 2
        depth = (hip_y - knee_y) / thigh if thigh > 0 else 0.0

        legs = [J.LEFT_HIP, J.RIGHT_HIP, J.LEFT_KNEE, J.RIGHT_KNEE, J.LEFT_ANKLE, J.RIGHT_ANKLE]
        visibility = sum(s.joint(j).visibility for j in legs) / len(legs)

        return SquatFeatures(knee_l, knee_r, hip_l, hip_r, depth, visibility)
