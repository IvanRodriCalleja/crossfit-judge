"""OpenCV video reader. Concrete VideoReader adapter."""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from squat.domain.ports import VideoReader


class OpenCVVideoReader(VideoReader):
    def fps(self, path: str) -> float:
        cap = cv2.VideoCapture(path)
        value = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(value) if value and value > 0 else 30.0

    def frames(self, path: str) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
