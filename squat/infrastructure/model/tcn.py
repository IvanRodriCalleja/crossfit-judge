"""Temporal convolutional network (TCN) for rep-density regression.

Dilated 1D convolutions: a few layers already cover a receptive field of several
seconds (enough for one rep) while keeping the parameter count low and inference
real-time. Input (B, F, T) -> output (B, T) with the per-frame density (>= 0)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # 'same' padding for odd kernel
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.down is None else self.down(x)
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return self.act(y + res)


class TCN(nn.Module):
    def __init__(self, n_features: int = 6, channels: int = 64, n_blocks: int = 5,
                 kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        c_in = n_features
        for i in range(n_blocks):
            blocks.append(TemporalBlock(c_in, channels, kernel, dilation=2 ** i, dropout=dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F, T)
        y = self.tcn(x)
        logits = self.head(y)        # (B, 1, T)
        return logits.squeeze(1)     # (B, T) logits; density is sigmoid(logits)

    def receptive_field(self, kernel: int = 3, n_blocks: int = 5) -> int:
        return 1 + 2 * sum((kernel - 1) * 2 ** i for i in range(n_blocks))
