from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import load_checkpoint
from .common_parent import CommonParentSpec, validate_parent_checkpoint
from .errors import ErrorCode, SctsrError
from .serialization import sha256_file, stable_digest


@dataclass(frozen=True, slots=True)
class BranchLineage:
    logical_run_id: str
    parent_id: str
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    training_seed: int
    arm_id: str
    child_source_tree_digest: str
    child_contract_digest: str
    created_at_utc: str
    parent_checkpoint_epoch: int = 120
    lineage_digest: str = ""

    @classmethod
    def create(
        cls,
        *,
        logical_run_id: str,
        parent_id: str,
        parent_checkpoint_path: str,
        parent_checkpoint_sha256: str,
        training_seed: int,
        arm_id: str,
        child_source_tree_digest: str,
        child_contract_digest: str,
        created_at_utc: str | None = None,
    ) -> "BranchLineage":
        item = cls(
            logical_run_id=logical_run_id,
            parent_id=parent_id,
            parent_checkpoint_path=parent_checkpoint_path,
            parent_checkpoint_sha256=parent_checkpoint_sha256.upper(),
            training_seed=training_seed,
            arm_id=arm_id,
            child_source_tree_digest=child_source_tree_digest,
            child_contract_digest=child_contract_digest,
            created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        )
        return replace(item, lineage_digest=item.compute_digest())

    def compute_digest(self) -> str:
        return stable_digest({name: getattr(self, name) for name in self.__dataclass_fields__ if name != "lineage_digest"})

    def validate(self, *, parent_sha: str, training_seed: int, arm_id: str, source_digest: str, contract_digest: str) -> None:
        for field in ("parent_checkpoint_sha256", "child_source_tree_digest", "child_contract_digest"):
            value = str(getattr(self, field))
            if len(value) != 64 or value.upper() != value or any(char not in "0123456789ABCDEF" for char in value):
                raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Lineage identity is not a canonical SHA-256", failing_field=field, observed=value)
        if not self.logical_run_id.strip() or not self.parent_id.strip() or not self.arm_id.strip():
            raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Lineage run, parent and arm IDs must be non-empty")
        if Path(self.parent_checkpoint_path).name.lower() == "best.pt":
            raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "Lineage may not bind best.pt")
        try:
            created = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Lineage timestamp is invalid", observed=self.created_at_utc) from exc
        if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
            raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Lineage timestamp must be UTC", observed=self.created_at_utc)
        expected = {
            "parent_checkpoint_sha256": parent_sha.upper(),
            "training_seed": training_seed,
            "arm_id": arm_id,
            "child_source_tree_digest": source_digest,
            "child_contract_digest": contract_digest,
            "parent_checkpoint_epoch": 120,
            "lineage_digest": self.compute_digest(),
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Branch lineage mismatch", failing_field=field, observed=getattr(self, field), expected=value)


def validate_branch_parent(
    lineage: BranchLineage,
    *,
    parent_spec: CommonParentSpec,
    expected_arm_id: str,
    expected_child_source_digest: str,
    expected_child_contract_digest: str,
) -> dict[str, Any]:
    """Validate a child launch against the actual immutable E120 parent bytes.

    A caller cannot launch from a naked checkpoint path: it must supply a
    digest-bound :class:`BranchLineage` and the registered parent contract.
    The file is hashed before and after deserialisation so validation itself
    also proves that the parent was not modified.
    """

    if lineage.parent_id != parent_spec.parent_id:
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "Lineage references a different registered parent",
            failing_field="parent_id",
            observed=lineage.parent_id,
            expected=parent_spec.parent_id,
        )
    parent_path = Path(lineage.parent_checkpoint_path)
    if not parent_path.is_file():
        raise SctsrError(
            ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
            "Lineage parent checkpoint does not exist",
            artifact_path=str(parent_path),
        )
    before_sha = sha256_file(parent_path)
    if before_sha != lineage.parent_checkpoint_sha256:
        raise SctsrError(
            ErrorCode.PARENT_SHA_MISMATCH,
            "Lineage does not bind the current parent checkpoint bytes",
            artifact_path=str(parent_path),
            observed=before_sha,
            expected=lineage.parent_checkpoint_sha256,
        )
    lineage.validate(
        parent_sha=before_sha,
        training_seed=parent_spec.training_seed,
        arm_id=expected_arm_id,
        source_digest=expected_child_source_digest,
        contract_digest=expected_child_contract_digest,
    )
    payload = load_checkpoint(parent_path, expected_sha256=before_sha, expected_epoch=120)
    validate_parent_checkpoint(payload, parent_spec)
    after_sha = sha256_file(parent_path)
    if after_sha != before_sha:
        raise SctsrError(
            ErrorCode.CHILD_MUTATED_PARENT,
            "Parent checkpoint changed while the child lineage was validated",
            artifact_path=str(parent_path),
            observed=after_sha,
            expected=before_sha,
        )
    return payload
