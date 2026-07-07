"""Inspect a rep dataset: how many squat videos per split, total reps and one example.

Usage:  ./.venv/bin/python squat/scripts/inspect_dataset.py <repcount-a-path>
"""
import sys
from collections import Counter

from squat.infrastructure.datasets.repcount import RepCountDataset


def main(root: str) -> None:
    ds = RepCountDataset(root)
    samples = list(ds.samples())

    by_split = Counter(s.split for s in samples)
    total_reps = sum(s.count for s in samples)
    print(f"dataset: RepCount-A (squat) in {root}")
    print(f"squat videos: {len(samples)} | by split: {dict(by_split)}")
    print(f"total annotated reps: {total_reps}")
    if samples:
        s = samples[0]
        print("\nexample:")
        print(f"  video  : {s.video_path.split('/')[-1]} (split={s.split}, subject={s.subject})")
        print(f"  reps   : {s.count}")
        print(f"  bounds : {[(r.start, r.end) for r in s.reps][:5]}{' ...' if s.count > 5 else ''}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: inspect_dataset.py <repcount-a-path>")
        sys.exit(1)
    main(sys.argv[1])
