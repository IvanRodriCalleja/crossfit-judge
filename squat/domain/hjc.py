"""Hip joint center (HJC) correction.

MediaPipe's hip point sits above the true center of the femoral joint. Since depth is
measured by hip height relative to the knee, that offset adds a systematic bias: the
squat looks shallower than it is. The correction shifts each hip point toward the
estimated HJC before computing features.

We use the regression model of Bell et al. (1989), which places the HJC relative to
the pelvic markers as fractions of pelvis width (PW, the hip-to-hip distance):
0.30·PW inferior, 0.19·PW posterior, and 0.36·PW medial from the midpoint (0.14·PW
medial from each marker). The shift is applied in the pelvis's own frame — axes set by
hips and shoulders — so it is independent of the global camera orientation. The
dominant term for depth is the inferior one (0.30·PW).
"""
from __future__ import annotations

import numpy as np

from .pose import Joint, Landmark, Skeleton

# Fractions of pelvis width (Bell et al., 1989).
_INFERIOR = 0.30
_POSTERIOR = 0.19
_MEDIAL = 0.14  # 0.36 from center − 0.50 from marker = 0.14 toward center


def _vec(lm: Landmark) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def correct_hip_center(skel: Skeleton) -> Skeleton:
    """Return a skeleton with the hip points shifted to the estimated HJC.
    If the required landmarks are missing, return the skeleton unchanged."""
    needed = [Joint.LEFT_HIP, Joint.RIGHT_HIP, Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER]
    if not all(skel.has(j) for j in needed):
        return skel

    hip_l, hip_r = _vec(skel.joint(Joint.LEFT_HIP)), _vec(skel.joint(Joint.RIGHT_HIP))
    sh_mid = (_vec(skel.joint(Joint.LEFT_SHOULDER)) + _vec(skel.joint(Joint.RIGHT_SHOULDER))) / 2
    hip_mid = (hip_l + hip_r) / 2

    pw = float(np.linalg.norm(hip_r - hip_l))
    if pw < 1e-6:
        return skel

    # Pelvic frame: medio-lateral (left->right), superior (hip->shoulders), antero-post.
    ml = _unit(hip_r - hip_l)
    up = _unit(sh_mid - hip_mid)
    ap = _unit(np.cross(up, ml))  # points to the front of the body

    def hjc(hip_v: np.ndarray, medial_sign: float) -> np.ndarray:
        return (hip_v
                - _INFERIOR * pw * up          # down toward the joint center
                - _POSTERIOR * pw * ap         # slightly backward
                + _MEDIAL * pw * medial_sign * ml)  # toward the pelvis center

    hjc_l = hjc(hip_l, +1.0)  # left hip: medial = rightward (+ml)
    hjc_r = hjc(hip_r, -1.0)  # right hip: medial = leftward (−ml)

    lm = dict(skel.landmarks)
    ol, orr = skel.joint(Joint.LEFT_HIP), skel.joint(Joint.RIGHT_HIP)
    lm[int(Joint.LEFT_HIP)] = Landmark(*hjc_l, ol.visibility)
    lm[int(Joint.RIGHT_HIP)] = Landmark(*hjc_r, orr.visibility)
    return Skeleton(lm)
