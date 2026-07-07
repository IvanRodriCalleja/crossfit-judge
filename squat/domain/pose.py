"""Body pose in a frame. Pure domain (no external dependencies): it only describes
what a pose is, not how it is obtained."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class Joint(IntEnum):
    """Joints used by the squat analysis, with their index in MediaPipe BlazePose's
    33 points."""
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class Landmark:
    """A body point in 3D coordinates (meters, body frame) plus its estimated
    visibility [0, 1]."""
    x: float
    y: float
    z: float
    visibility: float

    def distance_to(self, other: "Landmark") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True)
class Skeleton:
    """The landmarks of a frame, keyed by their MediaPipe index."""
    landmarks: dict[int, Landmark]

    def joint(self, j: Joint) -> Landmark:
        return self.landmarks[int(j)]

    def has(self, j: Joint) -> bool:
        return int(j) in self.landmarks
