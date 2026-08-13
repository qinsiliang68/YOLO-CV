from __future__ import annotations

from enum import Enum

from .errors import ErrorCode,SctsrError


class FaultKind(str,Enum):
    KILL="KILL";OOM="OOM";DISK_FULL="DISK_FULL";CORRUPT_RECEIPT="CORRUPT_RECEIPT";HALF_WRITTEN_JSON="HALF_WRITTEN_JSON";HALF_WRITTEN_PARQUET="HALF_WRITTEN_PARQUET"


def inject_fault(kind:FaultKind)->None:
    if kind is FaultKind.KILL:raise RuntimeError("SYNTHETIC_KILL_INJECTION")
    if kind is FaultKind.OOM:raise SctsrError(ErrorCode.OOM_FIXED_CONTRACT_ABORT,"Synthetic OOM injection",recoverable=True)
    if kind is FaultKind.DISK_FULL:raise SctsrError(ErrorCode.DISK_SPACE_PRECHECK_FAILED,"Synthetic disk-full injection",recoverable=False)
    if kind is FaultKind.CORRUPT_RECEIPT:raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH,"Synthetic corrupt receipt injection")
    if kind in {FaultKind.HALF_WRITTEN_JSON,FaultKind.HALF_WRITTEN_PARQUET}:raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,"Synthetic half-write injection")
    raise ValueError(kind)
