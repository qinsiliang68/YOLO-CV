from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .branch_lineage import BranchLineage
from .checkpointing import load_checkpoint
from .contracts import require_synthetic_or_authorized
from .errors import ErrorCode, SctsrError
from .serialization import sha256_file


def authorize_execution(*, execution_mode: str, release_authorization: str | Path | None) -> None:
    require_synthetic_or_authorized(execution_mode, release_authorization)


def validate_parent_for_branch(
    *, lineage: BranchLineage, parent_checkpoint: str | Path, arm_id: str,
    training_seed: int, source_digest: str, contract_digest: str,
) -> Mapping[str, Any]:
    parent = Path(parent_checkpoint)
    observed_sha = sha256_file(parent)
    lineage.validate(parent_sha=observed_sha, training_seed=training_seed, arm_id=arm_id, source_digest=source_digest, contract_digest=contract_digest)
    before = observed_sha
    payload = load_checkpoint(parent, expected_sha256=lineage.parent_checkpoint_sha256, expected_epoch=120)
    after = sha256_file(parent)
    if before != after:
        raise SctsrError(ErrorCode.CHILD_MUTATED_PARENT, "Reading child parent checkpoint changed its SHA")
    return payload


def reject_forbidden_data_role(role: str, *, purpose: str) -> None:
    if role == "test":
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Blind/test access is forbidden before final freeze")
    if role == "val_op" and purpose in {"method_selection", "checkpoint_selection", "stop_selection", "threshold_selection"}:
        raise SctsrError(ErrorCode.VAL_OP_SELECTION_FORBIDDEN, "val_op may not select method/checkpoint/stop/threshold")
