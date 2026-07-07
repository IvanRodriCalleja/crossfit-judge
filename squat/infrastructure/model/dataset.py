"""PyTorch dataset that reads the cached features and builds the density target.
For training it crops fixed-length windows (so they can be batched); for
validation/test it returns the full sequence."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from squat.application.density import build_density_target, rep_centers


class SquatCacheDataset(Dataset):
    def __init__(self, cache_dir: str, items: list[dict], sigma: float = 8.0,
                 chunk: int | None = None, samples_per_video: int = 1,
                 mean=None, std=None):
        self.sigma = sigma
        self.chunk = chunk
        # In training (fixed chunk) each video yields several windows per epoch;
        # in evaluation (full sequence) just one.
        self.samples_per_video = samples_per_video if chunk else 1
        self.mean = mean
        self.std = std
        self.data = []
        for it in items:
            npz = np.load(os.path.join(cache_dir, it["key"] + ".npz"))
            feats = npz["features"].astype(np.float32)        # (T, F)
            centers = rep_centers(npz["reps"])
            self.data.append((feats, centers, int(it["count"])))

    def __len__(self) -> int:
        return len(self.data) * self.samples_per_video

    def _normalize(self, feats: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            return (feats - self.mean) / self.std
        return feats

    def __getitem__(self, i: int):
        feats, centers, count = self.data[i % len(self.data)]
        n = feats.shape[0]
        density = build_density_target(centers, n, self.sigma)
        feats = self._normalize(feats)

        if self.chunk and n > self.chunk:
            start = np.random.randint(0, n - self.chunk + 1)
            feats = feats[start:start + self.chunk]
            density = density[start:start + self.chunk]
        elif self.chunk and n < self.chunk:
            pad = self.chunk - n
            feats = np.concatenate([feats, np.repeat(feats[-1:], pad, axis=0)], axis=0)
            density = np.concatenate([density, np.zeros(pad, dtype=np.float32)])

        x = torch.from_numpy(np.ascontiguousarray(feats.T)).float()  # (F, T)
        y = torch.from_numpy(density).float()                         # (T,)
        return x, y, float(count)
