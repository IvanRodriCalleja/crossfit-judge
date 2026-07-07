"""Use case: extract the feature sequence from a video.

Orchestrates video reader + pose estimator + feature extractor through their
interfaces (it knows nothing of MediaPipe or OpenCV). On frames with no pose
detected it carries the last valid features forward to keep the sequence intact."""
from __future__ import annotations

from squat.domain.ports import FeatureExtractor, PoseEstimator, VideoReader


class FeatureSequenceExtractor:
    def __init__(self, reader: VideoReader, estimator: PoseEstimator,
                 extractor: FeatureExtractor):
        self._reader = reader
        self._estimator = estimator
        self._extractor = extractor

    def extract(self, video_path: str) -> list:
        features = []
        last = None
        for frame in self._reader.frames(video_path):
            skeleton = self._estimator.estimate(frame)
            if skeleton is None:
                if last is not None:
                    features.append(last)
                continue
            last = self._extractor.extract(skeleton)
            features.append(last)
        return features
