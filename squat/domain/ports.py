"""Domain interfaces (ports). The application depends on these abstractions, not on
concrete implementations, so MediaPipe, the video reader or the model can be swapped
without touching the rest (dependency inversion, SOLID)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np

from .pose import Skeleton
from .sequence import RepResult, RepSample


class VideoReader(ABC):
    """Reads a video as a sequence of RGB frames."""

    @abstractmethod
    def fps(self, path: str) -> float: ...

    @abstractmethod
    def frames(self, path: str) -> Iterator[np.ndarray]: ...


class PoseEstimator(ABC):
    """Estimates the body pose in a frame."""

    @abstractmethod
    def estimate(self, frame_rgb: np.ndarray) -> Optional[Skeleton]:
        """Return the detected skeleton, or None if no one is detected."""


class FeatureExtractor(ABC):
    """Turns a skeleton into the feature vector fed to the model."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]: ...

    @abstractmethod
    def extract(self, skeleton: Skeleton):
        """Return the frame's features (an object with .to_list())."""


class RepDataset(ABC):
    """Source of videos with their annotated reps. Each dataset (RepCount, Exercise
    Recognition, MM-Fit, own recordings) is an adapter of this interface, so the rest
    of the system does not depend on its format."""

    @abstractmethod
    def samples(self) -> Iterator[RepSample]: ...


class RepCounter(ABC):
    """Counts reps from a video's feature sequence. Two interchangeable
    implementations: a signal-based one (baseline) and one based on the temporal
    model (TCN)."""

    @abstractmethod
    def count(self, features: list) -> RepResult: ...
