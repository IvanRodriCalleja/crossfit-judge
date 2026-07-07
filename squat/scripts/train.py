"""Train the counting TCN on the cached RepCount features.

- Splits: RepCount's own (train/valid/test).
- Normalisation: per-feature z-score with train statistics.
- Loss: BCE-with-logits on the density, weighted (peaks weigh ~11x the background).
- Metrics: MAE and OBO (|rounded count - true| <= 1).
Saves the best model (by validation MAE) in checkpoints/squat/.

Usage:  ./.venv/bin/python squat/scripts/train.py [epochs]
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from squat.application.density import count_peaks
from squat.infrastructure.model.dataset import SquatCacheDataset
from squat.infrastructure.model.tcn import TCN

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "repcount")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "squat")
SIGMA = 8.0


def feature_stats(items):
    feats = [np.load(os.path.join(CACHE, it["key"] + ".npz"))["features"] for it in items]
    allf = np.concatenate(feats, axis=0).astype(np.float32)
    return allf.mean(0), allf.std(0) + 1e-6


def evaluate(model, dataset, device):
    model.eval()
    errors, obo = [], 0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _, count = dataset[i]
            logits = model(x.unsqueeze(0).to(device)).squeeze(0)
            density = torch.sigmoid(logits).cpu().numpy()   # density = sigmoid(logits)
            pc = count_peaks(density)
            errors.append(abs(pc - count))
            obo += int(abs(pc - count) <= 1)
    return float(np.mean(errors)), obo / len(dataset)


def main(epochs: int = 40, include_exrec: bool = False, ckpt_name: str = "tcn_counter.pt"):
    torch.manual_seed(42)
    np.random.seed(42)
    items = json.load(open(os.path.join(CACHE, "manifest.json")))["items"]

    def is_repcount(it):
        return it["key"].startswith("repcount")

    # RepCount (REAL label) for train/valid/test. exrec (weak label) is only added to
    # training if include_exrec (ablation); never to valid/test.
    train = [it for it in items if is_repcount(it) and it["split"] == "train"]
    if include_exrec:
        train += [it for it in items if it["key"].startswith("exrec")]
    valid = [it for it in items if is_repcount(it) and it["split"] == "valid"]
    test = [it for it in items if is_repcount(it) and it["split"] == "test"]

    mean, std = feature_stats(train)
    tr = SquatCacheDataset(CACHE, train, SIGMA, chunk=256, samples_per_video=20,
                           mean=mean, std=std)
    va = SquatCacheDataset(CACHE, valid, SIGMA, chunk=None, mean=mean, std=std)
    te = SquatCacheDataset(CACHE, test, SIGMA, chunk=None, mean=mean, std=std)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TCN(n_features=mean.shape[0], channels=64, n_blocks=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(tr, batch_size=16, shuffle=True)
    print(f"device={device} | train={len(tr)} valid={len(va)} test={len(te)} | "
          f"campo receptivo={model.receptive_field()} frames")

    os.makedirs(CKPT_DIR, exist_ok=True)
    history = []
    best_mae = math.inf
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            weight = 1.0 + 10.0 * y           # peaks weigh ~11x more than the background
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val_mae, val_obo = evaluate(model, va, device)
        flag = ""
        train_loss = total / len(loader)
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_mae": val_mae, "val_obo": val_obo})
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "sigma": SIGMA, "channels": 64, "n_blocks": 5,
                        "n_features": int(mean.shape[0])},
                       os.path.join(CKPT_DIR, ckpt_name))
            flag = "  <- best"
        print(f"epoch {ep:3d} | loss {train_loss:.4f} | "
              f"valid MAE {val_mae:.2f} OBO {val_obo:.0%}{flag}")

    hist_name = ckpt_name.replace(".pt", "_history.json")
    with open(os.path.join(CKPT_DIR, hist_name), "w") as fh:
        json.dump(history, fh)

    # final evaluation with the best model
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_mae, test_obo = evaluate(model, te, device)
    tag = "RepCount+exrec" if include_exrec else "RepCount only"
    print(f"\nTEST ({tag}) | MAE {test_mae:.2f} | OBO {test_obo:.0%}  "
          f"(signal baseline: MAE 1.78 / OBO 61%)")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    exrec = len(sys.argv) > 2 and sys.argv[2] == "exrec"
    main(epochs, include_exrec=exrec,
         ckpt_name="tcn_counter_exrec.pt" if exrec else "tcn_counter.pt")
