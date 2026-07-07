"""Evaluate the TCN model (trained on RepCount + exrec) on our own CrossFit
recordings, to see how it generalizes to real phone video.

Known true counts per take (every rep counts, valid or not):
  1≈98 (alternating)  2=20  3=16  4=25  5=22
The 45° views of takes 4 and 5 are truncated, so they're dropped from the aggregate.
"""
import json
import os
import sys

import numpy as np
import torch

from squat.application.density import count_peaks
from squat.infrastructure.model.tcn import TCN

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "grabaciones")
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "squat", "tcn_counter.pt")
TRUE = {"1": 98, "2": 20, "3": 16, "4": 25, "5": 22}
TRUNCATED = {("45deg", "4"), ("45deg", "5")}


def main(ckpt_path=CKPT):
    ck = torch.load(ckpt_path, weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = TCN(n_features=ck["n_features"], channels=ck["channels"], n_blocks=ck["n_blocks"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    items = json.load(open(os.path.join(CACHE, "manifest.json")))["items"]
    per_persp = {}
    print(f"{'recording':16} {'pred':>5} {'true':>5} {'|err|':>6}")
    for it in sorted(items, key=lambda x: (x["perspective"], x["take"])):
        feats = np.load(os.path.join(CACHE, it["key"] + ".npz"))["features"]
        x = torch.from_numpy(((feats - mean) / std).T[None]).float()
        with torch.no_grad():
            d = torch.sigmoid(model(x)).squeeze(0).numpy()
        pred = count_peaks(d)
        true = TRUE[it["take"]]
        trunc = (it["perspective"], it["take"]) in TRUNCATED
        err = abs(pred - true)
        print(f"{it['key']:16} {pred:5d} {true:5d} {err:6d}" + ("  (truncated, excluded)" if trunc else ""))
        if not trunc:
            per_persp.setdefault(it["perspective"], []).append(err)

    print("\nBy perspective (truncated excluded):")
    for persp, errs in per_persp.items():
        obo = sum(e <= 1 for e in errs) / len(errs)
        print(f"  {persp:8} n={len(errs)}  MAE={np.mean(errs):.2f}  OBO={obo:.0%}")
    all_errs = [e for v in per_persp.values() for e in v]
    print(f"  TOTAL    n={len(all_errs)}  MAE={np.mean(all_errs):.2f}  "
          f"OBO={sum(e <= 1 for e in all_errs)/len(all_errs):.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CKPT)
