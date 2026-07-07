"""Counterfactual generation of no-reps by trimming the trajectory.

A valid squat already contains, in its intermediate frames, the poses of a no-rep.
The hip travels 180°->bottom->180°, so the frame on the way up where the hip sits at
165° is a real skeleton of that body at 165° of extension. Taking that point as the
"top" of the rep yields a genuine lockout failure with an exact label, without
editing any joints (every point is measured by MediaPipe, so the real variety is
preserved).

- lockout_fail: trims the ascent once the hip reaches a target < the real extension
  -> hip_ext_at_top = target; depth and knee stay real.
- shallow_fail: drops the frames below a target depth -> depth_at_bottom = target
  (never breaks parallel); the top (lockout) is kept.

Only for augmenting training with multi-subject data. Validation always uses real
reps (see plan.md).
"""
from __future__ import annotations

import numpy as np

from squat.application.rep_features import (_DEPTH, _HIP, _KNEE, RepFeatures,
                                            extract_rep_features)


def _signals(window: np.ndarray):
    knee = (window[:, _KNEE[0]] + window[:, _KNEE[1]]) / 2.0
    hip = (window[:, _HIP[0]] + window[:, _HIP[1]]) / 2.0
    depth = window[:, _DEPTH]
    return knee, hip, depth


def real_features(window: np.ndarray) -> RepFeatures:
    """Real features of the full window (the baseline valid rep)."""
    knee, hip, depth = _signals(window)
    return RepFeatures(float(depth.max()), float(knee.min()), float(hip.max()))


def lockout_fail(window: np.ndarray, target_hip_ext: float) -> RepFeatures | None:
    """The rep never fully extends the hip: drop the frames with hip above
    `target_hip_ext` (< real extension) on BOTH sides (the rep starts and ends
    standing, hip ~180°). What remains is the bottom region bounded by the target
    -> hip_ext_at_top = target; depth and knee stay real."""
    knee, hip, depth = _signals(window)
    if hip.max() <= target_hip_ext:
        return None
    keep = hip <= target_hip_ext               # bottom region, without the high extension
    if keep.sum() < 2:
        return None
    w = window[keep]
    knee_t, hip_t, depth_t = _signals(w)
    return RepFeatures(float(depth_t.max()), float(knee_t.min()), float(hip_t.max()))


def shallow_fail(window: np.ndarray, target_depth: float) -> RepFeatures | None:
    """Drop the frames below `target_depth` (< real depth): the rep only descends
    that far and comes back, keeping the lockout. None if not applicable."""
    knee, hip, depth = _signals(window)
    if depth.max() <= target_depth:
        return None                            # already shallower than the target
    keep = depth <= target_depth               # keep descent+ascent up to the target
    if keep.sum() < 2 or not keep[0]:
        return None
    w = window[keep]
    knee_t, hip_t, depth_t = _signals(w)
    return RepFeatures(float(depth_t.max()), float(knee_t.min()), float(hip_t.max()))


def rep_windows(features: np.ndarray, reps: np.ndarray) -> list[np.ndarray]:
    """Window (frames) of each rep from its annotated (start, end) bounds
    (RepCount cache format)."""
    out = []
    T = len(features)
    for s, e in reps:
        s, e = max(0, int(s)), min(T - 1, int(e))
        w = features[s:e + 1]
        if len(w) >= 3:  # minimum descent+ascent window
            out.append(w)
    return out


def rep_windows_from_bottoms(features: np.ndarray, bottoms) -> list[np.ndarray]:
    """Window of each rep bounded by the midpoints to the neighbouring reps (same
    convention as extract_rep_features), for the recordings cache, which stores the
    bottoms instead of (start, end) bounds."""
    T = len(features)
    bottoms = [int(b) for b in bottoms]
    out = []
    for i, b in enumerate(bottoms):
        lo = (bottoms[i - 1] + b) // 2 if i > 0 else max(0, b - 45)
        hi = (b + bottoms[i + 1]) // 2 if i < len(bottoms) - 1 else min(T, b + 45)
        lo, hi = max(0, lo), min(T, max(hi, lo + 1))
        if hi - lo >= 3:
            out.append(features[lo:hi])
    return out


def lockout_window(window: np.ndarray, target_hip_ext: float) -> np.ndarray | None:
    """Like lockout_fail, but returns the frame sub-window (for models that consume
    the sequence or rich features), not just the aggregate measures."""
    _, hip, _ = _signals(window)
    if hip.max() <= target_hip_ext:
        return None
    keep = hip <= target_hip_ext
    return window[keep] if keep.sum() >= 2 else None


def shallow_window(window: np.ndarray, target_depth: float) -> np.ndarray | None:
    """Like shallow_fail, but returns the frame sub-window."""
    _, _, depth = _signals(window)
    if depth.max() <= target_depth:
        return None
    keep = depth <= target_depth
    return window[keep] if (keep.sum() >= 2 and keep[0]) else None


def shallow_window_frac(window: np.ndarray, frac: float) -> np.ndarray | None:
    """Depth no-rep by cutting the descent at a fraction of the real range (frac<1:
    doesn't reach the bottom). Relative to each rep, so it holds for any view/scale,
    unlike an absolute threshold that depends on the view."""
    _, _, depth = _signals(window)
    top, bottom = float(depth.min()), float(depth.max())
    if bottom - top < 1e-3:
        return None
    target = top + frac * (bottom - top)   # partial depth reached
    keep = depth <= target
    return window[keep] if (keep.sum() >= 2 and keep[0]) else None


def multi_severity(window: np.ndarray, kind: str, targets) -> list[RepFeatures]:
    """Several no-reps from one valid rep by varying the stop point (failure
    severity). Turns N reps into N×len(targets) samples and covers the severity
    continuum, borderline cases included. kind: 'lockout' | 'shallow'."""
    fn = lockout_fail if kind == "lockout" else shallow_fail
    return [r for t in targets if (r := fn(window, t)) is not None]
