"""Fail-closed validation for the Stage1 core-method anchor contract."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


class AnchorContractError(RuntimeError):
    """Raised when an anchor contract cannot be used reproducibly."""


REQUIRED_COLUMNS = {
    "anchor_id",
    "family",
    "rq_ids",
    "canonical_title",
    "canonical_authors",
    "year",
    "queue_id",
    "canonical_work_id",
    "current_status",
    "inclusion_basis",
    "evidence_role",
    "primary_url",
    "full_text_url",
}
ALLOWED_FAMILIES = {
    "TRAINING_DYNAMICS",
    "REDUCIBLE_LEARNABILITY",
    "DIRECTION_INFLUENCE",
    "LABEL_RELIABILITY",
    "DIVERSITY_COVERAGE",
    "TIMING_BUDGET_RANDOM",
    "FN95_STATE_DEPENDENCE",
}
ALLOWED_STATUSES = {
    "BROAD_V2_ELIGIBLE",
    "DISCOVERED_NOT_BROAD_SCREENED",
    "EXTERNAL_PRIMARY_IDENTITY_VERIFIED_PENDING_SCREEN",
}
RQ_IDS = {f"RQ{index}" for index in range(1, 9)}
PLACEHOLDER = re.compile(r"(?:\b(?:todo|tbd|unknown)\b|待补|待确认|同上)", re.I)


@dataclass(frozen=True)
class AnchorContractValidation:
    status: str
    anchor_count: int
    status_counts: dict[str, int]
    contract_sha256: str
    formal_broad_increment: int = 0
    formal_screened_increment: int = 0
    formal_deep_increment: int = 0


def _read_contract_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise AnchorContractError(f"anchor contract missing columns: {missing}")
        return [dict(row) for row in reader]


def _read_broad_membership(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        raise AnchorContractError(f"frozen BROAD membership missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted({"queue_id", "canonical_work_id"} - fields)
        if missing:
            raise AnchorContractError(
                f"frozen BROAD membership missing columns: {missing}"
            )
        pairs = {
            (row["queue_id"].strip(), row["canonical_work_id"].strip())
            for row in reader
        }
    if not pairs:
        raise AnchorContractError("frozen BROAD membership is empty")
    return pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def derive_canonical_work_id(*, title: str, authors: str, year: str | int) -> str:
    """Derive the work ID with the same identity rule as BROAD staging."""

    payload = "|".join(
        (
            _normalize_identity(title),
            _normalize_identity(authors),
            str(year).strip(),
        )
    )
    return "CW" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20].upper()


def _require_text(row: dict[str, str], field: str, anchor_id: str) -> str:
    value = row[field].strip()
    if not value or PLACEHOLDER.search(value):
        raise AnchorContractError(f"{anchor_id} invalid {field}")
    return value


def validate_anchor_contract(
    path: Path,
    *,
    expected_count: int,
    broad_membership_path: Path,
) -> AnchorContractValidation:
    """Validate identity, source and state fields without granting reading credit."""

    source = Path(path)
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not source.is_file():
        raise AnchorContractError(f"anchor contract missing: {source}")
    rows = _read_contract_rows(source)
    broad_membership = _read_broad_membership(Path(broad_membership_path))

    if len(rows) != expected_count:
        raise AnchorContractError(
            f"anchor count {len(rows)} does not equal expected {expected_count}"
        )
    expected_ids = [f"A{index:03d}" for index in range(1, expected_count + 1)]
    observed_ids = [row["anchor_id"].strip() for row in rows]
    if observed_ids != expected_ids:
        raise AnchorContractError(
            f"anchor_id sequence mismatch: expected {expected_ids[0]}..{expected_ids[-1]}"
        )

    unique_fields = ("anchor_id", "queue_id", "canonical_work_id", "canonical_title")
    for field in unique_fields:
        values = [row[field].strip().casefold() for row in rows]
        if len(values) != len(set(values)):
            raise AnchorContractError(f"duplicate {field}")

    status_counts = {status: 0 for status in sorted(ALLOWED_STATUSES)}
    for row in rows:
        anchor_id = row["anchor_id"].strip()
        for field in (
            "canonical_title",
            "canonical_authors",
            "inclusion_basis",
            "evidence_role",
        ):
            _require_text(row, field, anchor_id)
        try:
            year = int(row["year"])
        except ValueError as exc:
            raise AnchorContractError(f"{anchor_id} invalid year") from exc
        if not 1900 <= year <= 2100:
            raise AnchorContractError(f"{anchor_id} year outside allowed range")

        family = row["family"].strip()
        if family not in ALLOWED_FAMILIES:
            raise AnchorContractError(f"{anchor_id} invalid family: {family}")
        rqs = {part.strip() for part in row["rq_ids"].split(";") if part.strip()}
        if not rqs or not rqs.issubset(RQ_IDS):
            raise AnchorContractError(f"{anchor_id} invalid rq_ids: {row['rq_ids']}")
        if not re.fullmatch(r"(?:RG|AX)[A-F0-9]{16}", row["queue_id"].strip()):
            raise AnchorContractError(f"{anchor_id} invalid queue_id")
        if not re.fullmatch(r"CW[A-F0-9]{20}", row["canonical_work_id"].strip()):
            raise AnchorContractError(f"{anchor_id} invalid canonical_work_id")
        observed_work_id = row["canonical_work_id"].strip()

        status = row["current_status"].strip()
        if status not in ALLOWED_STATUSES:
            raise AnchorContractError(f"{anchor_id} invalid current_status: {status}")
        if status == "BROAD_V2_ELIGIBLE":
            membership = (row["queue_id"].strip(), observed_work_id)
            if membership not in broad_membership:
                raise AnchorContractError(
                    f"{anchor_id} frozen BROAD membership mismatch: "
                    f"queue_id={membership[0]}, canonical_work_id={membership[1]}"
                )
        else:
            expected_work_id = derive_canonical_work_id(
                title=row["canonical_title"],
                authors=row["canonical_authors"],
                year=year,
            )
            if observed_work_id != expected_work_id:
                raise AnchorContractError(
                    f"{anchor_id} canonical_work_id mismatch: "
                    f"expected {expected_work_id}, observed {observed_work_id}"
                )
        status_counts[status] += 1
        for field in ("primary_url", "full_text_url"):
            value = row[field].strip()
            if not value.startswith("https://"):
                raise AnchorContractError(f"{anchor_id} invalid {field}: {value}")

    return AnchorContractValidation(
        status="PASS",
        anchor_count=len(rows),
        status_counts={key: value for key, value in status_counts.items() if value},
        contract_sha256=_sha256(source),
    )


def build_anchor_source_expected_rows(
    path: Path,
    *,
    broad_membership_path: Path,
) -> tuple[dict[str, str], ...]:
    """Derive the expected supplemental-source identities from a valid contract."""

    source = Path(path)
    if not source.is_file():
        raise AnchorContractError(f"anchor contract missing: {source}")
    rows = _read_contract_rows(source)
    validate_anchor_contract(
        source,
        expected_count=len(rows),
        broad_membership_path=broad_membership_path,
    )
    selected = [
        {
            "paper_id": row["queue_id"].strip(),
            "title": row["canonical_title"].strip(),
        }
        for row in rows
        if row["current_status"].strip() != "BROAD_V2_ELIGIBLE"
    ]
    return tuple(selected)


def load_validated_anchor_work_ids(
    path: Path,
    *,
    expected_count: int,
    broad_membership_path: Path,
) -> tuple[str, ...]:
    """Return frozen anchor identities only after the complete contract passes."""

    source = Path(path)
    validate_anchor_contract(
        source,
        expected_count=expected_count,
        broad_membership_path=broad_membership_path,
    )
    return tuple(
        row["canonical_work_id"].strip() for row in _read_contract_rows(source)
    )
