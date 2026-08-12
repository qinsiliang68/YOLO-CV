"""Build a self-contained, validated BROAD-500 staging corpus.

The staging corpus is deliberately not the formal corpus.  It proves that the
500 broad records, notes, and source identities are internally consistent while
leaving the formal 500/300/100 registry untouched until deeper reading passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

from .literature_evidence_v2 import (
    TierCounts,
    validate_corpus,
    validate_source_acquisitions,
)
from .literature_tier_freeze_v2 import (
    BroadCandidate,
    ExplicitMerge,
    TierSelectionPolicy,
    canonicalize_candidates,
    select_broad_candidates,
)


REGISTRY_FIELDS = (
    "paper_id",
    "tier",
    "canonical_work_id",
    "title",
    "authors",
    "year",
    "venue",
    "primary_url",
    "doi",
    "arxiv_id",
    "openreview_id",
    "note_path",
    "source_path",
    "source_sha256",
    "source_bytes",
)
SELECTION_FIELDS = (
    "paper_id",
    "queue_id",
    "canonical_work_id",
    "quota_rq",
    "secondary_rqs",
    "directness",
    "relevance_class",
    "relation",
    "selection_phase",
    "tie_break_key",
    "original_source_path",
    "source_sha256",
    "source_bytes",
    "merged_queue_ids",
)
SOURCE_FIELDS = (
    "paper_id",
    "artifact_role",
    "path",
    "url",
    "retrieved_at",
    "http_status",
    "content_type",
    "bytes",
    "sha256",
    "retrieval_method",
    "source_authority",
)
BUILD_INPUT_FIELDS = (
    "input_role",
    "root_scope",
    "path",
    "bytes",
    "sha256",
)


class BroadStagingError(RuntimeError):
    """Raised when broad staging cannot be built without weakening evidence."""


@dataclass(frozen=True)
class ManualBroadEvidence:
    candidate: BroadCandidate
    queue: Mapping[str, str]
    decision: Mapping[str, str]
    source: Mapping[str, str]
    receipt_ledger: Mapping[str, Any]


@dataclass(frozen=True)
class BroadStagingResult:
    status: str
    output_root: Path
    selected_count: int
    reserve_count: int
    quota_counts: Mapping[str, int]
    formal_broad_increment: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _canonical_work_id(title: str, authors: str, year: str) -> str:
    payload = "|".join(
        (_normalize_identity(title), _normalize_identity(authors), year.strip())
    )
    return "CW" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20].upper()


def _read_csv(path: Path, *, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise BroadStagingError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required) - fields)
        if missing:
            raise BroadStagingError(f"{path.name} missing columns: {missing}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_source(root: Path, raw: str, queue_id: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        raise BroadStagingError(f"{queue_id} source path must be corpus-relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BroadStagingError(f"{queue_id} source path escapes corpus root") from exc
    return resolved


def _latest_validation_csv(batch_root: Path, batch_number: int) -> Path:
    stem = f"source_validation_{batch_number:03d}"
    versioned = sorted(
        batch_root.glob(f"{stem}_v*.csv"),
        key=lambda path: int(re.search(r"_v(\d+)\.csv$", path.name).group(1)),
    )
    return versioned[-1] if versioned else batch_root / f"{stem}.csv"


def _load_receipt(root: Path, source: Mapping[str, str], queue_id: str) -> Mapping[str, Any]:
    receipt_path = _safe_source(root, source["receipt_path"], queue_id)
    if not receipt_path.is_file():
        raise BroadStagingError(f"{queue_id} source receipt missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BroadStagingError(f"{queue_id} source receipt is invalid JSON") from exc
    ledger = receipt.get("ledger_row")
    if not isinstance(ledger, Mapping):
        raise BroadStagingError(f"{queue_id} source receipt lacks ledger_row")
    required = {
        "url",
        "retrieved_at",
        "http_status",
        "content_type",
        "retrieval_method",
        "source_authority",
    }
    missing = sorted(required - set(ledger))
    if missing:
        raise BroadStagingError(f"{queue_id} receipt ledger missing fields: {missing}")
    return ledger


def _load_merges(path: Path) -> tuple[ExplicitMerge, ...]:
    rows = _read_csv(
        path,
        required=("alias_queue_id", "canonical_queue_id", "evidence"),
    )
    merges: list[ExplicitMerge] = []
    for row in rows:
        alias = row["alias_queue_id"].strip()
        canonical = row["canonical_queue_id"].strip()
        if alias.startswith("NOT_APPLICABLE_WITH_REASON:"):
            continue
        merges.append(
            ExplicitMerge(
                alias_queue_id=alias,
                canonical_queue_id=canonical,
                evidence=row["evidence"].strip(),
                merge_basis=(row.get("merge_basis") or "EXACT_METADATA").strip(),
                shared_identity=(row.get("shared_identity") or "").strip(),
            )
        )
    return tuple(merges)


def load_manual_broad_evidence(
    corpus_root: str | Path,
    *,
    batch_numbers: Iterable[int],
) -> tuple[ManualBroadEvidence, ...]:
    """Load and independently recheck every eligible manual broad decision."""

    root = Path(corpus_root).resolve()
    batch_root = root / "discovery" / "manual_screen_batches_v2"
    decision_root = root / "discovery" / "manual_screen_decisions_v2"
    evidence: list[ManualBroadEvidence] = []
    seen_ids: set[str] = set()
    for batch_number in batch_numbers:
        queue_rows = _read_csv(
            batch_root / f"review_input_{batch_number:03d}.csv",
            required=(
                "queue_id",
                "title",
                "authors",
                "year",
                "venue",
                "primary_url",
                "doi",
                "candidate_version_ids",
            ),
        )
        decision_rows = _read_csv(
            decision_root / f"batch_{batch_number:03d}.csv",
            required=(
                "queue_id",
                "decision",
                "canonical_title",
                "primary_url_checked",
                "source_authority",
                "checked_at",
                "reading_scope",
                "direct_rq_ids",
                "relevance_class",
                "problem_summary_zh",
                "method_overview_zh",
                "conclusion_summary_zh",
                "critical_review_zh",
                "stage1_transfer_zh",
                "cannot_infer_zh",
            ),
        )
        source_rows = _read_csv(
            _latest_validation_csv(batch_root, batch_number),
            required=(
                "paper_id",
                "title",
                "path",
                "bytes",
                "sha256",
                "source_format",
                "source_authority",
                "source_url",
                "receipt_path",
            ),
        )
        queue_by_id = {row["queue_id"]: row for row in queue_rows}
        source_by_id = {row["paper_id"]: row for row in source_rows}
        if len(queue_by_id) != len(queue_rows):
            raise BroadStagingError(f"batch {batch_number:03d} has duplicate queue IDs")
        for decision in decision_rows:
            if decision["decision"] != "ELIGIBLE_BROAD":
                continue
            queue_id = decision["queue_id"]
            if queue_id in seen_ids:
                raise BroadStagingError(f"duplicate eligible queue ID across batches: {queue_id}")
            seen_ids.add(queue_id)
            queue = queue_by_id.get(queue_id)
            source = source_by_id.get(queue_id)
            if queue is None:
                raise BroadStagingError(f"{queue_id} decision has no queue identity")
            if source is None:
                raise BroadStagingError(f"{queue_id} eligible decision has no verified source")
            if _normalize_identity(decision["canonical_title"]) != _normalize_identity(queue["title"]):
                raise BroadStagingError(f"{queue_id} decision title does not match queue")
            if _normalize_identity(source["title"]) != _normalize_identity(queue["title"]):
                raise BroadStagingError(f"{queue_id} source title does not match queue")
            if decision["source_authority"] != source["source_authority"]:
                raise BroadStagingError(f"{queue_id} source authority disagrees with verified source")
            source_path = _safe_source(root, source["path"], queue_id)
            if not source_path.is_file():
                raise BroadStagingError(f"{queue_id} source file missing: {source_path}")
            actual_bytes = source_path.stat().st_size
            try:
                expected_bytes = int(source["bytes"])
            except ValueError as exc:
                raise BroadStagingError(f"{queue_id} source bytes are invalid") from exc
            if actual_bytes != expected_bytes:
                raise BroadStagingError(
                    f"{queue_id} source byte mismatch: expected {expected_bytes}, observed {actual_bytes}"
                )
            actual_sha = _sha256(source_path)
            if actual_sha != source["sha256"].upper():
                raise BroadStagingError(
                    f"{queue_id} source SHA mismatch: expected {source['sha256']}, observed {actual_sha}"
                )
            receipt_ledger = _load_receipt(root, source, queue_id)
            rqs = tuple(sorted({value for value in decision["direct_rq_ids"].split(";") if value}))
            candidate = BroadCandidate(
                queue_id=queue_id,
                canonical_work_id=_canonical_work_id(
                    queue["title"], queue["authors"], queue["year"]
                ),
                title=queue["title"],
                authors=queue["authors"],
                year=int(queue["year"]),
                direct_rqs=rqs,
                relevance_class=decision["relevance_class"],
                doi=queue["doi"],
                effect_relation=_effect_relation(decision["relevance_class"]),
            )
            evidence.append(
                ManualBroadEvidence(
                    candidate=candidate,
                    queue=queue,
                    decision=decision,
                    source=source,
                    receipt_ledger=receipt_ledger,
                )
            )
    return tuple(evidence)


def _relation(relevance_class: str) -> str:
    if relevance_class == "STRICT_CONTROL_NEGATIVE":
        return "REFUTED"
    if relevance_class == "DIRECT_INTERVENTION":
        return "SUPPORTED"
    return "MIXED"


def _effect_relation(relevance_class: str) -> str:
    if relevance_class == "STRICT_CONTROL_NEGATIVE":
        return "NULL_NEGATIVE"
    if relevance_class == "DIRECT_INTERVENTION":
        return "SUPPORTED"
    if relevance_class == "TARGET_METRIC":
        return "METHOD_ONLY"
    return "MIXED"


def _external_id(values: Sequence[str], pattern: str, missing_reason: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for value in values:
        match = regex.search(value)
        if match:
            return match.group(1).removesuffix(".pdf")
    return f"NOT_APPLICABLE_WITH_REASON:{missing_reason}"


def _formal_metadata_value(value: str, field: str) -> str:
    clean = value.strip()
    if clean in {"NOT_REPORTED_BY_SOURCE", "NOT_REPORTED_BY_PAPER"}:
        return f"NOT_APPLICABLE_WITH_REASON:{field} not reported by verified primary metadata"
    return clean


def _metadata(
    *,
    paper_id: str,
    selected: Any,
    evidence: ManualBroadEvidence,
    staged_source: Path,
    output_root: Path,
    merged_versions: Sequence[str],
) -> dict[str, Any]:
    queue = evidence.queue
    decision = evidence.decision
    source = evidence.source
    title = queue["title"]
    formal_authors = _formal_metadata_value(queue["authors"], "authors")
    formal_venue = _formal_metadata_value(queue["venue"], "venue")
    formal_doi = _formal_metadata_value(queue["doi"], "DOI")
    relation = _relation(decision["relevance_class"])
    summary = (
        f"问题：{decision['problem_summary_zh']} 方法：{decision['method_overview_zh']} "
        f"结论：{decision['conclusion_summary_zh']}"
    )
    critical = (
        f"对该研究的证据边界判断：{decision['critical_review_zh']} "
        f"反向约束：{decision['cannot_infer_zh']}"
    )
    relevance = (
        f"该研究直接映射到 {','.join(selected.candidate.direct_rqs)}；"
        f"其可迁移链条是：{decision['stage1_transfer_zh']} "
        "最终 utility 仍须由固定累计曝光下的真实配对 replay 干预确认。"
    )
    judgment = (
        f"关系判定为 {relation}：原文摘要层结论是 {decision['conclusion_summary_zh']} "
        f"但 Stage1 约束是 {decision['critical_review_zh']}"
    )
    boundary = (
        f"允许迁移：{decision['stage1_transfer_zh']} "
        f"禁止外推：{decision['cannot_infer_zh']}"
    )
    source_kind = (
        "PRIMARY_FULL_TEXT_PDF"
        if source["source_format"].upper() == "PDF"
        else "OFFICIAL_LANDING_HTML"
    )
    source_rel = staged_source.relative_to(output_root).as_posix()
    primary_values = (
        decision["primary_url_checked"],
        queue["primary_url"],
        queue["doi"],
        source["source_url"],
    )
    return {
        "schema_version": "2.0",
        "paper_id": paper_id,
        "tier": "BROAD",
        "identity": {
            "canonical_work_id": selected.candidate.canonical_work_id,
            "title": title,
            "authors": [value.strip() for value in formal_authors.split(";") if value.strip()],
            "year": int(queue["year"]),
            "venue": formal_venue,
            "primary_url": decision["primary_url_checked"],
            "doi": formal_doi,
            "arxiv_id": _external_id(
                primary_values,
                r"(?:arxiv[\.:/]|arxiv\.)(\d{4}\.\d{4,5}(?:v\d+)?)",
                "no arXiv identity found in verified metadata",
            ),
            "openreview_id": _external_id(
                primary_values,
                r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_-]+)",
                "not an OpenReview work",
            ),
            "merged_versions": list(merged_versions),
        },
        "source_artifact": {
            "path": source_rel,
            "kind": source_kind,
            "bytes": int(source["bytes"]),
            "sha256": source["sha256"].upper(),
        },
        "rq_ids": list(selected.candidate.direct_rqs),
        "relation": relation,
        "reading": {
            "read_at": decision["checked_at"],
            "scopes": [value for value in decision["reading_scope"].split(";") if value],
            "sections_checked": [
                "Primary title and identity",
                "Primary abstract",
                "Research problem",
                "Method overview",
                "Conclusion",
            ],
            "summary_zh": summary,
            "critical_review_zh": critical,
            "direct_relevance_chain": relevance,
            "supported_or_refuted": judgment,
            "transferable_mechanisms": [decision["stage1_transfer_zh"]],
            "unsupported_inferences": [decision["cannot_infer_zh"]],
            "stage1_boundary": boundary,
        },
        "screened": "NOT_ASSESSED_AT_BROAD_LEVEL",
        "deep": "NOT_ASSESSED_AT_BROAD_LEVEL",
    }


def _write_note(path: Path, metadata: Mapping[str, Any]) -> None:
    reading = metadata["reading"]
    text = (
        "<!-- STAGE1_EVIDENCE_V2 -->\n"
        "```json\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        + f"# {metadata['paper_id']} - {metadata['identity']['title']}\n\n"
        + "## 独立中文摘要\n\n"
        + reading["summary_zh"]
        + "\n\n## 批判性小综述\n\n"
        + reading["critical_review_zh"]
        + "\n\n## Stage1 直接相关链\n\n"
        + reading["direct_relevance_chain"]
        + "\n\n## 迁移边界\n\n"
        + reading["stage1_boundary"]
        + "\n"
    )
    path.write_text(text, encoding="utf-8")


def _root_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _input_manifest_row(
    *,
    role: str,
    root_scope: str,
    path: Path,
    relative_to: Path,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BroadStagingError(f"build input missing: {resolved}")
    try:
        relative = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError as exc:
        raise BroadStagingError(
            f"build input {resolved} is outside declared {root_scope} root"
        ) from exc
    return {
        "input_role": role,
        "root_scope": root_scope,
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _build_input_manifest(
    root: Path,
    *,
    batch_numbers: Sequence[int],
    evidence_rows: Sequence[ManualBroadEvidence],
    merge_ledger_path: Path,
    policy_source_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    batch_root = root / "discovery" / "manual_screen_batches_v2"
    decision_root = root / "discovery" / "manual_screen_decisions_v2"
    inputs: list[tuple[str, str, Path, Path]] = [
        (
            "CANONICAL_MERGES",
            "CORPUS",
            merge_ledger_path,
            root,
        ),
        ("BUILDER_CODE", "REPOSITORY_CODE", Path(__file__), repo_root),
        (
            "SELECTION_CODE",
            "REPOSITORY_CODE",
            repo_root / "stage1_dynamic_replay_v3" / "literature_tier_freeze_v2.py",
            repo_root,
        ),
        (
            "VALIDATOR_CODE",
            "REPOSITORY_CODE",
            repo_root / "stage1_dynamic_replay_v3" / "literature_evidence_v2.py",
            repo_root,
        ),
    ]
    for policy_source in policy_source_paths:
        inputs.append(
            (
                "SELECTION_POLICY_SOURCE",
                "CORPUS",
                policy_source,
                root,
            )
        )
    for batch_number in batch_numbers:
        inputs.extend(
            (
                (
                    "MANUAL_REVIEW_INPUT",
                    "CORPUS",
                    batch_root / f"review_input_{batch_number:03d}.csv",
                    root,
                ),
                (
                    "MANUAL_REVIEW_DECISION",
                    "CORPUS",
                    decision_root / f"batch_{batch_number:03d}.csv",
                    root,
                ),
                (
                    "SOURCE_VALIDATION",
                    "CORPUS",
                    _latest_validation_csv(batch_root, batch_number),
                    root,
                ),
            )
        )
    for evidence in evidence_rows:
        queue_id = evidence.candidate.queue_id
        inputs.extend(
            (
                (
                    "VERIFIED_PRIMARY_SOURCE",
                    "CORPUS",
                    _safe_source(root, evidence.source["path"], queue_id),
                    root,
                ),
                (
                    "SOURCE_RECEIPT",
                    "CORPUS",
                    _safe_source(root, evidence.source["receipt_path"], queue_id),
                    root,
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for role, scope, path, relative_to in inputs:
        row = _input_manifest_row(
            role=role,
            root_scope=scope,
            path=path,
            relative_to=relative_to,
        )
        identity = (role, scope, str(row["path"]))
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["root_scope"], row["path"], row["input_role"]))


def build_broad_staging(
    corpus_root: str | Path,
    *,
    batch_numbers: Iterable[int],
    policy: TierSelectionPolicy = TierSelectionPolicy(),
    merge_ledger_path: Path = Path("discovery/CANONICAL_MERGES_v2.csv"),
    policy_source_paths: Sequence[Path] = (),
    output_relative: Path = Path("staging/broad_freeze_v2"),
    replace_existing: bool = False,
) -> BroadStagingResult:
    """Build and atomically publish a broad-only staging corpus."""

    root = Path(corpus_root).resolve()
    frozen_batch_numbers = tuple(batch_numbers)
    if not frozen_batch_numbers:
        raise BroadStagingError("at least one manual screening batch is required")
    if output_relative.is_absolute() or ".." in output_relative.parts:
        raise BroadStagingError("output_relative must stay inside the corpus root")
    output = (root / output_relative).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise BroadStagingError("output path escapes corpus root") from exc
    if output.exists() and not replace_existing:
        raise BroadStagingError(f"staging output already exists: {output}")

    merge_ledger = Path(merge_ledger_path)
    if not merge_ledger.is_absolute():
        merge_ledger = root / merge_ledger
    merge_ledger = merge_ledger.resolve()
    try:
        merge_ledger.relative_to(root)
    except ValueError as exc:
        raise BroadStagingError("merge ledger path escapes corpus root") from exc

    evidence_rows = load_manual_broad_evidence(
        root,
        batch_numbers=frozen_batch_numbers,
    )
    evidence_by_queue = {item.candidate.queue_id: item for item in evidence_rows}
    merges = _load_merges(merge_ledger)
    canonical = canonicalize_candidates(
        (item.candidate for item in evidence_rows),
        merges=merges,
    )
    selected_result = select_broad_candidates(canonical, policy)

    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise BroadStagingError(f"stale staging temp exists: {temp}")
    temp.mkdir(parents=True)
    try:
        (temp / "notes").mkdir()
        (temp / "sources").mkdir()
        registry_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        acquisition_rows: list[dict[str, Any]] = []
        for index, selected in enumerate(selected_result.selected, start=1):
            paper_id = f"P{index:04d}"
            candidate = selected.candidate
            evidence = evidence_by_queue[candidate.queue_id]
            source = evidence.source
            original = _safe_source(root, source["path"], candidate.queue_id)
            suffix = ".pdf" if source["source_format"].upper() == "PDF" else ".html"
            staged_source = temp / "sources" / f"{paper_id}{suffix}"
            os.link(original, staged_source)
            merged_versions = [
                value
                for value in evidence.queue["candidate_version_ids"].split(";")
                if value
            ]
            merged_versions.append(candidate.queue_id)
            for alias_id in candidate.merged_queue_ids:
                merged_versions.append(alias_id)
                alias_evidence = evidence_by_queue[alias_id]
                merged_versions.extend(
                    value
                    for value in alias_evidence.queue["candidate_version_ids"].split(";")
                    if value
                )
            metadata = _metadata(
                paper_id=paper_id,
                selected=selected,
                evidence=evidence,
                staged_source=staged_source,
                output_root=temp,
                merged_versions=tuple(dict.fromkeys(merged_versions)),
            )
            note = temp / "notes" / f"{paper_id}.md"
            _write_note(note, metadata)
            identity = metadata["identity"]
            artifact = metadata["source_artifact"]
            registry_rows.append(
                {
                    "paper_id": paper_id,
                    "tier": "BROAD",
                    "canonical_work_id": identity["canonical_work_id"],
                    "title": identity["title"],
                    "authors": "; ".join(identity["authors"]),
                    "year": identity["year"],
                    "venue": identity["venue"],
                    "primary_url": identity["primary_url"],
                    "doi": identity["doi"],
                    "arxiv_id": identity["arxiv_id"],
                    "openreview_id": identity["openreview_id"],
                    "note_path": f"notes/{paper_id}.md",
                    "source_path": artifact["path"],
                    "source_sha256": artifact["sha256"],
                    "source_bytes": artifact["bytes"],
                }
            )
            secondary = [rq for rq in candidate.direct_rqs if rq != selected.quota_rq]
            selection_rows.append(
                {
                    "paper_id": paper_id,
                    "queue_id": candidate.queue_id,
                    "canonical_work_id": candidate.canonical_work_id,
                    "quota_rq": selected.quota_rq,
                    "secondary_rqs": ";".join(secondary)
                    or "NOT_APPLICABLE_WITH_REASON:no secondary RQ",
                    "directness": selected.directness,
                    "relevance_class": candidate.relevance_class,
                    "relation": metadata["relation"],
                    "selection_phase": selected.selection_phase,
                    "tie_break_key": selected.tie_break_key,
                    "original_source_path": source["path"],
                    "source_sha256": source["sha256"].upper(),
                    "source_bytes": source["bytes"],
                    "merged_queue_ids": ";".join(candidate.merged_queue_ids)
                    or "NOT_APPLICABLE_WITH_REASON:no merged queue aliases",
                }
            )
            receipt = evidence.receipt_ledger
            acquisition_rows.append(
                {
                    "paper_id": paper_id,
                    "artifact_role": "BROAD_SOURCE",
                    "path": artifact["path"],
                    "url": receipt["url"],
                    "retrieved_at": receipt["retrieved_at"],
                    "http_status": receipt["http_status"],
                    "content_type": receipt["content_type"],
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                    "retrieval_method": receipt["retrieval_method"],
                    "source_authority": receipt["source_authority"],
                }
            )

        reserve_rows = [
            {
                "queue_id": candidate.queue_id,
                "canonical_work_id": candidate.canonical_work_id,
                "title": candidate.title,
                "direct_rqs": ";".join(candidate.direct_rqs),
                "directness": candidate.directness,
                "relevance_class": candidate.relevance_class,
                "reserve_reason": "LOWER_LEXICOGRAPHIC_PRIORITY_AFTER_EXACT_BROAD_TOTAL",
            }
            for candidate in selected_result.reserves
        ]
        _write_csv(temp / "CANONICAL_WORKS.csv", registry_rows, REGISTRY_FIELDS)
        _write_csv(temp / "BROAD_500.csv", selection_rows, SELECTION_FIELDS)
        _write_csv(
            temp / "BROAD_RESERVES.csv",
            reserve_rows,
            (
                "queue_id",
                "canonical_work_id",
                "title",
                "direct_rqs",
                "directness",
                "relevance_class",
                "reserve_reason",
            ),
        )
        _write_csv(temp / "SOURCE_ACQUISITION.csv", acquisition_rows, SOURCE_FIELDS)
        input_manifest_rows = _build_input_manifest(
            root,
            batch_numbers=frozen_batch_numbers,
            evidence_rows=evidence_rows,
            merge_ledger_path=merge_ledger,
            policy_source_paths=tuple(Path(path) for path in policy_source_paths),
        )
        input_manifest_path = temp / "BUILD_INPUT_MANIFEST.csv"
        _write_csv(
            input_manifest_path,
            input_manifest_rows,
            BUILD_INPUT_FIELDS,
        )

        report = validate_corpus(
            temp,
            expected=TierCounts(policy.total, 0, 0),
            inspect_pdf_pages=False,
        )
        validate_source_acquisitions(temp, report.papers)
        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "selected_count": len(selected_result.selected),
            "reserve_count": len(selected_result.reserves),
            "quota_counts": dict(selected_result.quota_counts),
            "formal_broad_increment": 0,
            "formal_registry_published": False,
            "formal_training_started": False,
            "engineering_gate_generated": False,
            "blind_holdout_opened": False,
            "build_input_count": len(input_manifest_rows),
            "build_input_manifest_sha256": _sha256(input_manifest_path),
            "selection_policy": {
                "total": policy.total,
                "minimum_per_rq": policy.minimum_per_rq,
                "maximum_per_rq": policy.maximum_per_rq,
                "maximum_transfer": policy.maximum_transfer,
                "minimum_counterevidence_per_rq": (
                    policy.minimum_counterevidence_per_rq
                ),
                "mandatory_canonical_work_ids": list(
                    policy.mandatory_canonical_work_ids
                ),
                "tier_label": policy.tier_label,
                "frozen_seed": policy.frozen_seed,
            },
            "root_digest_before_receipt": _root_digest(temp),
        }
        (temp / "FREEZE_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise BroadStagingError(f"stale staging backup exists: {backup}")
            output.rename(backup)
            try:
                temp.rename(output)
            except Exception:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            temp.rename(output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    return BroadStagingResult(
        status="PASS",
        output_root=output,
        selected_count=len(selected_result.selected),
        reserve_count=len(selected_result.reserves),
        quota_counts=dict(selected_result.quota_counts),
    )
