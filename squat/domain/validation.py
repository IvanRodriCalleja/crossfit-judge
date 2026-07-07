"""REP/NO-REP validation by geometric rules (explainable baseline).

Maps the rulebook criteria directly to thresholds on the per-rep measures. It is the
baseline the learned model is compared against, and also the explanation layer: it
returns the verdict and the reason.
"""
from __future__ import annotations

from dataclasses import dataclass

from squat.application.rep_features import RepFeatures

VALIDA = "VALIDA"
INVALIDA_PROFUNDIDAD = "INVALIDA_PROFUNDIDAD"
INVALIDA_LOCKOUT = "INVALIDA_LOCKOUT"


@dataclass
class RuleBasedValidator:
    # Thresholds calibrated from the observed separation between classes.
    depth_threshold: float = -0.15    # min depth_at_bottom to accept breaking parallel
    lockout_threshold: float = 163.0  # min hip_ext_at_top (degrees) to accept lockout

    def judge(self, rf: RepFeatures) -> tuple[str, str]:
        """Return (verdict, reason)."""
        if rf.depth_at_bottom < self.depth_threshold:
            return INVALIDA_PROFUNDIDAD, "did not break parallel (hip above the knee)"
        if rf.hip_ext_at_top < self.lockout_threshold:
            return INVALIDA_LOCKOUT, "hip not fully extended at the top"
        return VALIDA, "meets depth and hip extension"
