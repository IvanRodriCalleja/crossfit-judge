"""RepCount-A dataset adapter. Implements RepDataset.

RepCount-A is a rep-counting dataset. Each split has a CSV with, per video: the
exercise type, the rep count and the start/end frames of each rep (columns L1,
L2, ... in pairs). Here we keep only the squat videos and turn each row into a
domain RepSample.
"""
from __future__ import annotations

import csv
import os
from typing import Iterator

from squat.domain.ports import RepDataset
from squat.domain.sequence import RepBoundary, RepSample

# The squat label shows up as "squat" and, due to a typo in the original
# dataset, also as "squant".
SQUAT_LABELS = ("squat", "squant")


class RepCountDataset(RepDataset):
    def __init__(self, root: str, splits=("train", "valid", "test"),
                 labels=SQUAT_LABELS):
        self.root = root
        self.splits = splits
        self.labels = labels

    def _rows(self, split: str):
        csv_path = os.path.join(self.root, "annotation", f"{split}.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)

    @staticmethod
    def _reps_from_row(row: dict) -> list[RepBoundary]:
        # Collect the non-empty L1, L2, ... values and pair them (start, end).
        vals = []
        i = 1
        while f"L{i}" in row:
            v = row[f"L{i}"]
            if v not in (None, ""):
                vals.append(int(float(v)))
            i += 1
        return [RepBoundary(vals[j], vals[j + 1]) for j in range(0, len(vals) - 1, 2)]

    def samples(self) -> Iterator[RepSample]:
        for split in self.splits:
            for row in self._rows(split):
                if (row.get("type") or "").strip().lower() not in self.labels:
                    continue
                name = (row.get("name") or "").strip()
                video_path = os.path.join(self.root, "video", split, name)
                if not os.path.exists(video_path):
                    continue
                reps = self._reps_from_row(row)
                subject = name.split("_")[0] if name.startswith("stu") else None
                yield RepSample(
                    video_path=video_path,
                    reps=reps,
                    split=split,
                    subject=subject,
                    source="repcount",
                )
