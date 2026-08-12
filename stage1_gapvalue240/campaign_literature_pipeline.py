"""Atomic, traceable publication of the campaign literature review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from .campaign_literature import (
    _atomic_csv,
    _atomic_text,
    _sha256,
    assert_literature_evidence_matrix,
    publish_literature_review,
)
from .campaign_literature_registry import build_literature_matrix_from_candidates


class LiteraturePipelineError(RuntimeError):
    """Raised when a literature snapshot cannot be published safely."""


def _atomic_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _method_note(
    *,
    candidate_count: int,
    matrix: pd.DataFrame,
    exclusion_count: int,
) -> str:
    depths = matrix["screening_depth"].value_counts()
    return "\n".join(
        [
            "# Literature review method",
            "",
            "## Scope and reading levels",
            "",
            f"- OpenAlex shortlisted metadata records preserved: {candidate_count}",
            f"- Included primary papers: {len(matrix)}",
            f"- Abstract screens: {int(depths.get('ABSTRACT_SCREEN', 0))}",
            f"- Method-level reads: {int(depths.get('METHOD_READ', 0))}",
            f"- Deep or near-full reads: {int(depths.get('DEEP_READ', 0))}",
            f"- Logged exclusions: {exclusion_count}",
            "",
            "`ABSTRACT_SCREEN` means title and primary abstract were checked for scope. ",
            "`METHOD_READ` means the primary abstract exposed both a method mechanism and an ",
            "experimental scope; it does not claim a cover-to-cover read. `DEEP_READ` is the ",
            "manually curated mechanism set for which method, experiments, and limitations or ",
            "scope boundaries were inspected in the primary paper.",
            "",
            "## Selection rules",
            "",
            "1. Prefer conference or journal proceedings, DOI landing pages, OpenReview, or arXiv primary records.",
            "2. Deduplicate normalized titles and let a manual deep-read record supersede its metadata candidate.",
            "3. Exclude surveys, obvious domain collisions, unsupported hosts, and records without a mechanism-relevant title.",
            "4. Balance the retained abstract pool across training dynamics, subset selection, attribution, noisy labels, replay, optimization variance, and operational-tail evaluation.",
            "5. Use literature to define falsifiable measurements and boundaries, not to claim that an uncollected quantity was observed.",
            "",
            "## Interpretation boundary",
            "",
            "The matrix is a decision evidence map, not a meta-analysis. Paper counts do not vote on the correct Stage1 mechanism. ",
            "The ranked directions prioritize direct agreement with observed same-selection seed reversals, missing-field closure, ",
            "feasibility before 2026-09-10, and ability to falsify the proposed mechanism.",
            "",
        ]
    )


def _query_log(candidates: pd.DataFrame) -> pd.DataFrame:
    needed = {"category_query", "query", "openalex_id"}
    if not needed.issubset(candidates.columns):
        return pd.DataFrame(
            [{"category_query": "UNRECORDED", "query": "UNRECORDED", "shortlisted_records": len(candidates)}]
        )
    return (
        candidates.groupby(["category_query", "query"], dropna=False)["openalex_id"]
        .nunique()
        .rename("shortlisted_records")
        .reset_index()
        .sort_values(["category_query", "query"], kind="stable")
        .reset_index(drop=True)
    )


def publish_campaign_literature(
    candidate_source: str | Path,
    output_dir: str | Path,
    *,
    campaign_id: str,
    core_records: Sequence[Mapping[str, Any]],
    target_count: int = 155,
    method_target: int = 55,
    min_screened: int = 150,
    min_method: int = 50,
    min_deep: int = 20,
    discovery_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish discovery, screening, and reading evidence."""

    source = Path(candidate_source).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise LiteraturePipelineError(f"candidate source does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise LiteraturePipelineError(f"refusing to overwrite non-empty output: {output}")

    candidates = pd.read_csv(source)
    build = build_literature_matrix_from_candidates(
        candidates,
        target_count=target_count,
        method_target=method_target,
        core_records=core_records,
    )
    counts = assert_literature_evidence_matrix(
        build.matrix,
        min_screened=min_screened,
        min_method=min_method,
        min_deep=min_deep,
    )
    query_log = _query_log(candidates)
    stated_counts = {
        "raw_results": int((discovery_counts or {}).get("raw_results", len(candidates))),
        "deduplicated": int((discovery_counts or {}).get("deduplicated", len(candidates))),
        "shortlisted": int((discovery_counts or {}).get("shortlisted", len(candidates))),
    }
    if stated_counts["shortlisted"] != len(candidates):
        raise LiteraturePipelineError(
            f"shortlisted count {stated_counts['shortlisted']} does not match candidate rows {len(candidates)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{os.getpid()}.tmpdir"
    if staging.exists():
        raise LiteraturePipelineError(f"staging path already exists: {staging}")
    try:
        discovery = staging / "discovery"
        discovery.mkdir(parents=True, exist_ok=False)
        snapshot = discovery / "OPENALEX_CANDIDATES.csv"
        _atomic_bytes(source.read_bytes(), snapshot)
        _atomic_csv(query_log, discovery / "DISCOVERY_QUERY_LOG.csv")
        provenance = {
            "campaign_id": campaign_id,
            "source": "OpenAlex API high-recall discovery followed by local screening",
            "source_path_at_build": str(source),
            "source_sha256": _sha256(source),
            "snapshot_sha256": _sha256(snapshot),
            "candidate_rows": len(candidates),
            "unique_queries": len(query_log),
            "discovery_counts": stated_counts,
            "built_at": "2026-08-07",
        }
        _atomic_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            discovery / "DISCOVERY_PROVENANCE.json",
        )
        _atomic_csv(build.exclusions, staging / "SCREENING_EXCLUSIONS.csv")
        _atomic_text(
            _method_note(
                candidate_count=len(candidates),
                matrix=build.matrix,
                exclusion_count=len(build.exclusions),
            ),
            staging / "LITERATURE_REVIEW_METHOD.md",
        )
        review_validation = publish_literature_review(
            build.matrix,
            staging,
            campaign_id=campaign_id,
            min_screened=min_screened,
            min_method=min_method,
            min_deep=min_deep,
        )
        artifacts = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "relative_path": path.relative_to(staging).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
        receipt = {
            "status": "complete",
            "campaign_id": campaign_id,
            "counts": counts,
            "candidate_rows": len(candidates),
            "exclusion_rows": len(build.exclusions),
            "review_validation": review_validation,
            "artifacts": artifacts,
        }
        _atomic_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            staging / "LITERATURE_BUILD_RECEIPT.json",
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = ["LiteraturePipelineError", "publish_campaign_literature"]
