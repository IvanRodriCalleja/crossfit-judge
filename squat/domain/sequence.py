"""Domain types for a video's annotated reps. Pure, no external dependencies."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepBoundary:
    """A rep delimited by its start and end frames."""
    start: int
    end: int

    @property
    def middle(self) -> int:
        return (self.start + self.end) // 2


@dataclass(frozen=True)
class RepSample:
    """A video with its annotated reps. Common unit across all datasets."""
    video_path: str
    reps: list[RepBoundary]
    split: str                  # 'train' | 'valid' | 'test'
    subject: str | None = None  # for subject-disjoint splits (no leakage)
    source: str = ""            # origin dataset (repcount, exrec, ...)

    @property
    def count(self) -> int:
        return len(self.reps)


@dataclass(frozen=True)
class RepResult:
    """Rep-counting result: how many, and the bottom frame of each."""
    count: int
    bottoms: list[int]
