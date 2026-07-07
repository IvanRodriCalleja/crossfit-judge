"""Sync the lateral and frontal views using the movement itself.

Both cameras record the same reps at once, so the knee-angle signal is the same curve
with a constant offset. That offset is estimated by cross-correlation (in time, to
tolerate different fps) and any lateral frame is mapped to its frontal counterpart.
Signals are cached to avoid recomputing.

Usage:  python squat/scripts/sync_views.py <take> <frame_lat_1> [frame_lat_2 ...]
"""
import os
import sys

import numpy as np

from squat.scripts.make_pose_grid import feats_per_frame

REC = os.path.join(os.path.dirname(__file__), "..", "data", "grabaciones")
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "sync")


def knee_signal(view, take):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{view}_{take}_knee.npy")
    if os.path.exists(f):
        return np.load(f)
    knee, _ = feats_per_frame(os.path.join(REC, view, f"{take}.mp4"))
    np.save(f, knee)
    return knee


def fps(view, take):
    from squat.infrastructure.video.opencv_reader import OpenCVVideoReader
    return OpenCVVideoReader().fps(os.path.join(REC, view, f"{take}.mp4"))


def resample(knee, fps_v, dt):
    t = np.arange(len(knee)) / fps_v
    grid = np.arange(0, t[-1], dt)
    s = np.interp(grid, t[~np.isnan(knee)], knee[~np.isnan(knee)])
    return grid, s - np.nanmean(s)


def main(take, lat_frames):
    fl, ff = fps("lateral", take), fps("frontal", take)
    kl, kf = knee_signal("lateral", take), knee_signal("frontal", take)
    dt = 1 / 60.0
    _, sl = resample(kl, fl, dt)
    _, sf = resample(kf, ff, dt)
    # cross-correlation: shift (in dt samples) that aligns frontal with lateral
    corr = np.correlate(sl, sf, mode="full")
    lag = np.argmax(corr) - (len(sf) - 1)
    shift_s = lag * dt   # t_frontal ≈ t_lateral - shift_s
    print(f"take {take}: fps lat {fl:.2f} / fro {ff:.2f} | estimated offset {shift_s:+.2f} s")
    print("lateral -> frontal map:")
    fro_frames = []
    for flat in lat_frames:
        t_lat = flat / fl
        t_fro = t_lat - shift_s
        f_fro = int(round(t_fro * ff))
        fro_frames.append(f_fro)
        print(f"  lateral f{flat} ({t_lat:.2f}s) -> frontal f{f_fro} ({t_fro:.2f}s)")
    return fro_frames


if __name__ == "__main__":
    take = sys.argv[1]
    lat_frames = [int(x) for x in sys.argv[2:]]
    main(take, lat_frames)
