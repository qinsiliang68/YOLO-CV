"""SCTSR v4: State-Conditional Tail-Safe Replay infrastructure.

This package is an isolated implementation overlay for qinsiliang68/YOLO-CV
main@a70ba60485dd32c2f8b4268b8f28ea2d3549f42f.  It deliberately does not
start formal training, generate assignments, open blind/test data, or claim
method effectiveness.
"""
from .arm_spec import ArmId, SctsrArmSpec, default_phase1_arms
from .errors import ErrorCode, SctsrError
from .rate_spec import ReplayRateSpec

__all__ = [
    "ArmId",
    "ErrorCode",
    "ReplayRateSpec",
    "SctsrArmSpec",
    "SctsrError",
    "default_phase1_arms",
]

__version__ = "4.0.0"
