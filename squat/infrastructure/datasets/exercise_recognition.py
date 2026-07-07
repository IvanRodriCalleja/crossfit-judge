"""Exercise Recognition dataset adapter (Riccio, 2024). Implements RepDataset.

Provides real squat videos, but WITHOUT rep annotations. That's why its samples
come out with `reps=[]`: the annotation is generated later, automatically (via
the signal baseline) when caching. Used only for training (weak labels);
evaluation stays on RepCount, with ground-truth labels.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

from squat.domain.ports import RepDataset
from squat.domain.sequence import RepSample


class ExerciseRecognitionDataset(RepDataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = root
        self.split = split

    def samples(self) -> Iterator[RepSample]:
        pattern = os.path.join(self.root, "*", "squat", "*.mp4")
        for path in sorted(glob.glob(pattern)):
            yield RepSample(
                video_path=path,
                reps=[],                 # no labels: auto-annotated when caching
                split=self.split,
                subject=None,
                source="exrec",
            )
