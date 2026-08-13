from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, stable_digest


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and value.upper() == value and all(char in "0123456789ABCDEF" for char in value)


@dataclass(frozen=True, slots=True)
class LogicalArtifactEntry:
    logical_run_id: str
    logical_epoch: int
    physical_owner_type: str
    physical_run_id: str
    artifact_relative_path: str
    artifact_sha256: str
    checkpoint_sha256: str
    source_tree_digest: str
    lineage_digest: str


class LogicalArtifactIndex:
    def __init__(self, entries: list[LogicalArtifactEntry] | None = None) -> None:
        self.entries = list(entries or [])

    def add(self, entry: LogicalArtifactEntry) -> None:
        key = (entry.logical_run_id, entry.logical_epoch)
        if any((x.logical_run_id, x.logical_epoch) == key for x in self.entries):
            raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Duplicate logical run/epoch entry", observed=key)
        self.entries.append(entry)

    @property
    def digest(self) -> str:
        return stable_digest([asdict(x) for x in sorted(self.entries, key=lambda y: (y.logical_run_id, y.logical_epoch))])

    def validate(self, *, require_complete_timeline: bool = False, logical_run_id: str | None = None) -> None:
        keys = [(entry.logical_run_id, entry.logical_epoch) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical artifact index contains duplicate run/epoch keys")
        for entry in self.entries:
            expected = "PARENT" if entry.logical_epoch <= 120 else "CHILD"
            if entry.physical_owner_type != expected:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical epoch points to wrong physical owner", observed=asdict(entry), expected=expected)
            if not 1 <= entry.logical_epoch <= 200:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical epoch is outside E1-E200")
            path = Path(entry.artifact_relative_path)
            if path.is_absolute() or path.drive or ".." in path.parts or entry.artifact_relative_path in {"", "."}:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Artifact path escapes experiment root")
            for field in ("artifact_sha256", "checkpoint_sha256", "source_tree_digest"):
                if not _is_sha256(str(getattr(entry, field))):
                    raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical artifact identity is not a canonical SHA-256", failing_field=field, observed=getattr(entry, field))
            if entry.physical_owner_type == "CHILD":
                if not _is_sha256(entry.lineage_digest):
                    raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Child logical artifact lacks a canonical lineage digest", observed=entry.lineage_digest)
            elif entry.lineage_digest != "NOT_APPLICABLE_PARENT" and not _is_sha256(entry.lineage_digest):
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Parent logical artifact has an invalid lineage marker", observed=entry.lineage_digest)
        if require_complete_timeline:
            if not logical_run_id:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Complete-timeline validation requires a logical run ID")
            observed = sorted(entry.logical_epoch for entry in self.entries if entry.logical_run_id == logical_run_id)
            expected = list(range(1, 201))
            if observed != expected:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical artifact index does not cover exactly E1-E200", observed=observed, expected=expected)
            foreign = sorted({entry.logical_run_id for entry in self.entries if entry.logical_run_id != logical_run_id})
            if foreign:
                raise SctsrError(ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH, "Logical artifact index mixes unrelated runs", observed=foreign)

    def write(self, path: str | Path) -> None:
        self.validate()
        atomic_write_json(path, {"schema_version": "stage1.sctsr.logical_artifact_index.v1", "digest": self.digest, "entries": [asdict(x) for x in sorted(self.entries, key=lambda y: (y.logical_run_id, y.logical_epoch))]})

    def validate_child_tree(self, child_root: str | Path) -> None:
        """Reject child-owned files or directories that masquerade as E1-E120.

        Historical epochs are represented only by parent-owned index entries;
        copying them into a child tree destroys artifact identity even when the
        bytes happen to match.
        """

        root = Path(child_root)
        if not root.exists():
            return
        epoch_token = re.compile(r"^(?:epoch[_-]?|e)(\d{1,4})(?:\D.*)?$", re.IGNORECASE)
        forged: list[str] = []
        for path in root.rglob("*"):
            for component in path.relative_to(root).parts:
                match = epoch_token.match(component)
                if match and 1 <= int(match.group(1)) <= 120:
                    forged.append(path.relative_to(root).as_posix())
                    break
        if forged:
            raise SctsrError(
                ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH,
                "Child tree contains physical artifacts for parent-owned E1-E120",
                observed=sorted(set(forged)),
                expected="E1-E120 represented only by PARENT index entries",
            )
