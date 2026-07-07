"""Pose estimator with MediaPipe BlazePose (Tasks API). PoseEstimator adapter.

Uses the world landmarks (3D in meters, origin at the hip center), which are the
right ones for computing angles and depth regardless of where the athlete is in
the frame. Runs in VIDEO mode to exploit temporal tracking.
"""
from __future__ import annotations

import os
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from squat.domain.pose import Landmark, Skeleton

from squat.domain.ports import PoseEstimator

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "models", "pose_landmarker_full.task"
)


class MediaPipePoseEstimator(PoseEstimator):
    def __init__(self, model_path: str | None = None, mode: str = "image",
                 fps: float = 30.0):
        """mode='image': each frame is processed independently (batch, offline).
        mode='video': processed as a stream with temporal tracking (real-time)."""
        model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing pose model: {model_path}")
        self._mode = mode
        running_mode = (vision.RunningMode.VIDEO if mode == "video"
                        else vision.RunningMode.IMAGE)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._dt_ms = 1000.0 / (fps if fps > 0 else 30.0)
        self._t_ms = 0.0

    def estimate(self, frame_rgb: np.ndarray) -> Optional[Skeleton]:
        result = self._detect(frame_rgb)
        if not result.pose_world_landmarks:
            return None
        return self._world_skeleton(result)

    def estimate_with_image(
        self, frame_rgb: np.ndarray
    ) -> Optional[tuple[Skeleton, dict[int, tuple[float, float]]]]:
        """Like estimate(), but also returns the normalized 2D landmarks (x, y in
        [0, 1] of the crop). The 2D ones are reliable on a single frame and serve
        to estimate the camera angle (lateral vs frontal/rear), which the 3D world
        landmarks estimate poorly without temporal context."""
        result = self._detect(frame_rgb)
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None
        image_xy = {i: (lm.x, lm.y) for i, lm in enumerate(result.pose_landmarks[0])}
        return self._world_skeleton(result), image_xy

    def _detect(self, frame_rgb: np.ndarray):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        if self._mode == "video":
            self._t_ms += self._dt_ms
            return self._landmarker.detect_for_video(image, int(self._t_ms))
        return self._landmarker.detect(image)

    @staticmethod
    def _world_skeleton(result) -> Skeleton:
        lms = result.pose_world_landmarks[0]  # first detected person
        landmarks = {
            i: Landmark(lm.x, lm.y, lm.z, lm.visibility) for i, lm in enumerate(lms)
        }
        return Skeleton(landmarks)

    def close(self) -> None:
        self._landmarker.close()
