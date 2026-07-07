"""Fitness-AQA dataset adapter (Parmar et al., 2022), squat subset with the
shallow-depth label.

Unlike the in-house recordings (a single subject), this is a public MULTI-SUBJECT
benchmark: 623 athletes with official subject-disjoint splits. It directly targets
the single-subject limitation in depth validation.

On-disk layout (already extracted) under Shallow_Squat_Error_Dataset/:
  crops_unaligned/<key>.jpg   crop of the rep's bottom frame
  labels_shallow_depth.json   {key: 0 no error | 1 shallow depth}
  splits/{train,val,test}_ids.json   key lists per split

The key has the form <subject>_<view>_<frame>; the first field identifies the
athlete and is used to check there's no leakage across splits.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

_DEFAULT_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "fitness-aqa", "Squat",
    "Labeled_Dataset", "Shallow_Squat_Error_Dataset",
)


@dataclass(frozen=True)
class ShallowCrop:
    key: str
    path: str
    label: int      # 0 = no error (enough depth), 1 = shallow depth
    subject: str    # athlete id (first field of the key)
    split: str      # "train" | "val" | "test"


class FitnessAQAShallow:
    """Reads the labeled shallow-depth crops with their official splits."""

    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or _DEFAULT_ROOT)
        self._labels = json.load(open(os.path.join(self.root, "labels_shallow_depth.json")))
        self._splits = {
            name: json.load(open(os.path.join(self.root, "splits", f"{name}_ids.json")))
            for name in ("train", "val", "test")
        }

    def crops(self, split: str) -> list[ShallowCrop]:
        out = []
        for key in self._splits[split]:
            if key not in self._labels:
                continue
            path = os.path.join(self.root, "crops_unaligned", f"{key}.jpg")
            if not os.path.exists(path):
                continue
            out.append(ShallowCrop(
                key=key, path=path, label=int(self._labels[key]),
                subject=key.split("_")[0], split=split,
            ))
        return out

    def all_crops(self) -> list[ShallowCrop]:
        return [c for s in ("train", "val", "test") for c in self.crops(s)]
