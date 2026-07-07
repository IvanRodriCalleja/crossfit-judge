"""Quick check of the pose + features pipeline on a squat video.

Extracts pose frame by frame, computes the squat features and summarises the signal.
The knee-angle oscillation should track the reps.

Usage:  ./.venv/bin/python squat/scripts/test_pose.py <video>
"""
import sys

from squat.domain.features import SquatFeatureExtractor
from squat.infrastructure.pose.mediapipe_estimator import MediaPipePoseEstimator
from squat.infrastructure.video.opencv_reader import OpenCVVideoReader


def _sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // 50)
    return "".join(
        blocks[min(7, max(0, int((v - lo) / rng * 7)))] for v in values[::step]
    )


def main(path: str) -> None:
    reader = OpenCVVideoReader()
    extractor = SquatFeatureExtractor()

    fps = reader.fps(path)
    estimator = MediaPipePoseEstimator(fps=fps)
    knee, depth = [], []
    n_frames = n_detected = 0
    for frame in reader.frames(path):
        n_frames += 1
        skeleton = estimator.estimate(frame)
        if skeleton is None:
            continue
        n_detected += 1
        f = extractor.extract(skeleton)
        knee.append((f.knee_angle_left + f.knee_angle_right) / 2)
        depth.append(f.hip_knee_depth)
    estimator.close()

    print(f"video: {path}")
    print(f"fps: {fps:.1f} | frames: {n_frames} | with pose detected: {n_detected}")
    if knee:
        print(f"knee angle (L/R mean): min={min(knee):.0f}°  max={max(knee):.0f}°")
        print(f"hip-knee depth:        min={min(depth):.2f}  max={max(depth):.2f}")
        print("knee over time: ", _sparkline(knee))
        print("depth over time:", _sparkline(depth))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: test_pose.py <video>")
        sys.exit(1)
    main(sys.argv[1])
