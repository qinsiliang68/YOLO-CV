"""Evidence contracts and synthesis for the dynamic replay literature review."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd


class LiteratureReviewError(RuntimeError):
    """Raised when literature evidence is too small, duplicated, or unverifiable."""


REQUIRED_COLUMNS = (
    "evidence_id",
    "title",
    "authors",
    "year",
    "venue",
    "primary_url",
    "doi",
    "topic",
    "direction_id",
    "screening_depth",
    "reading_basis",
    "sections_checked",
    "method_family",
    "measured_quantity",
    "selection_unit",
    "training_stage_dependency",
    "distinguishes_beneficial_harmful",
    "required_fields",
    "stage1_testability",
    "stage1_implication",
    "method_summary",
    "claim_boundary",
    "evidence_relation",
    "primary_source_verified",
    "verified_at",
    "abstract",
)

ALLOWED_DEPTHS = frozenset({"ABSTRACT_SCREEN", "METHOD_READ", "DEEP_READ"})
ALLOWED_RELATIONS = frozenset({"SUPPORTS", "CAUTIONS", "CONTRADICTS", "CONTEXT"})
PRIMARY_HOSTS = frozenset(
    {
        "doi.org",
        "dx.doi.org",
        "arxiv.org",
        "openreview.net",
        "proceedings.mlr.press",
        "proceedings.neurips.cc",
        "openaccess.thecvf.com",
        "aclanthology.org",
        "jmlr.org",
        "www.jmlr.org",
        "ojs.aaai.org",
        "dl.acm.org",
        "ieeexplore.ieee.org",
        "link.springer.com",
        "academic.oup.com",
        "proceedings.mlsys.org",
    }
)

DIRECTION_ORDER = (
    "D1_CONDITIONAL_VALUE",
    "D2_DYNAMIC_REPLAY",
    "D3_WEAK_DEFECT_GUARD",
    "D5_REALIZED_EXPOSURE",
    "D6_DIVERSITY_COVERAGE",
    "D4_GRADIENT_PILOT",
    "D7_OPERATIONAL_TAIL",
    "D8_STATIC_SCORE_BASELINES",
)

DIRECTION_LABELS = {
    "D1_CONDITIONAL_VALUE": "Estimate a value distribution conditional on seed and training state",
    "D2_DYNAMIC_REPLAY": "Control replay dose over training time",
    "D3_WEAK_DEFECT_GUARD": "Protect the weak-defect tail while lowering difficult-normal scores",
    "D5_REALIZED_EXPOSURE": "Measure sampler, batch, augmentation, and optimizer realization",
    "D6_DIVERSITY_COVERAGE": "Constrain redundancy and preserve coverage",
    "D4_GRADIENT_PILOT": "Pilot target-aligned last-layer gradients at key checkpoints",
    "D7_OPERATIONAL_TAIL": "Evaluate the preregistered high-recall raw-score frontier",
    "D8_STATIC_SCORE_BASELINES": "Retain static scores only as baselines or candidate-pool filters",
}


def _normalize_title(value: str) -> str:
    decoded = html.unescape(str(value)).lower()
    return re.sub(r"[^a-z0-9]+", " ", decoded).strip()


def _primary_url_valid(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    if parsed.scheme != "https" or parsed.netloc.lower() not in PRIMARY_HOSTS:
        return False
    lower_path = parsed.path.lower()
    return "/search" not in lower_path and "google." not in parsed.netloc.lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def assert_literature_evidence_matrix(
    matrix: pd.DataFrame,
    *,
    min_screened: int = 150,
    min_method: int = 50,
    min_deep: int = 20,
) -> dict[str, int]:
    """Validate review depth, provenance, and one-paper-one-row identity."""

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(matrix.columns))
    if missing_columns:
        raise LiteratureReviewError(f"missing columns: {missing_columns}")
    normalized = matrix["title"].fillna("").map(_normalize_title)
    duplicate_ids = int(matrix["evidence_id"].fillna("").duplicated().sum())
    duplicate_titles = int(normalized.duplicated().sum())
    invalid_urls = int(
        (~matrix["primary_url"].fillna("").map(_primary_url_valid)).sum()
        + (~matrix["primary_source_verified"].map(_truthy)).sum()
    )
    required = list(REQUIRED_COLUMNS)
    optional_blank = {"doi"}
    required_nonblank = [column for column in required if column not in optional_blank]
    missing_metadata = int(
        matrix[required_nonblank]
        .apply(lambda column: column.isna() | column.astype(str).str.strip().eq(""))
        .any(axis=1)
        .sum()
    )
    invalid_depth = ~matrix["screening_depth"].isin(ALLOWED_DEPTHS)
    invalid_relation = ~matrix["evidence_relation"].isin(ALLOWED_RELATIONS)
    missing_metadata += int(invalid_depth.sum() + invalid_relation.sum())

    invalid_reading = 0
    for row in matrix.itertuples(index=False):
        depth = str(row.screening_depth)
        sections = {item.strip().upper() for item in str(row.sections_checked).split(";")}
        if depth == "METHOD_READ":
            if not any("METHOD" in item for item in sections) or not any(
                "EXPERIMENT" in item for item in sections
            ):
                invalid_reading += 1
        elif depth == "DEEP_READ":
            needed = ("METHOD", "EXPERIMENT", "LIMITATION")
            if any(not any(token in item for item in sections) for token in needed):
                invalid_reading += 1

    counts = {
        "screened": len(matrix),
        "method_read": int(matrix["screening_depth"].eq("METHOD_READ").sum()),
        "deep_read": int(matrix["screening_depth"].eq("DEEP_READ").sum()),
        "duplicate_evidence_ids": duplicate_ids,
        "duplicate_normalized_titles": duplicate_titles,
        "invalid_primary_urls": invalid_urls,
        "missing_required_metadata": missing_metadata,
        "invalid_reading_evidence": invalid_reading,
    }
    errors = []
    if counts["screened"] < min_screened:
        errors.append(f"screened={counts['screened']}<{min_screened}")
    if counts["method_read"] < min_method:
        errors.append(f"method_read={counts['method_read']}<{min_method}")
    if counts["deep_read"] < min_deep:
        errors.append(f"deep_read={counts['deep_read']}<{min_deep}")
    for key in (
        "duplicate_evidence_ids",
        "duplicate_normalized_titles",
        "invalid_primary_urls",
        "missing_required_metadata",
        "invalid_reading_evidence",
    ):
        if counts[key]:
            errors.append(f"{key}={counts[key]}")
    if errors:
        raise LiteratureReviewError("Literature evidence failed gates: " + ", ".join(errors))
    return counts


def build_research_synthesis(matrix: pd.DataFrame) -> str:
    """Render the mechanism-led literature conclusion without blended weights."""

    rows = []
    for direction in DIRECTION_ORDER:
        group = matrix[matrix["direction_id"] == direction]
        rows.append(
            {
                "direction_id": direction,
                "label": DIRECTION_LABELS[direction],
                "papers": len(group),
                "deep": int(group["screening_depth"].eq("DEEP_READ").sum()),
                "method": int(group["screening_depth"].eq("METHOD_READ").sum()),
                "supports": int(group["evidence_relation"].eq("SUPPORTS").sum()),
                "cautions": int(group["evidence_relation"].eq("CAUTIONS").sum()),
            }
        )
    counts = matrix["screening_depth"].value_counts()
    lines = [
        "# Stage1 literature synthesis: conditional sample value",
        "",
        "## Review scope",
        "",
        f"- Screened primary papers: {len(matrix)}",
        f"- Method-level reads: {int(counts.get('METHOD_READ', 0))}",
        f"- Deep or near-full reads: {int(counts.get('DEEP_READ', 0))}",
        "",
        "## Main conclusion",
        "",
        "Sample value is not a seed-invariant scalar under the available evidence. The more defensible object is ",
        "a conditional effect distribution: `V(selection | theta_t, replay schedule, realized exposure, seed, context)`. ",
        "Gradient magnitude, confidence, loss, and forgetting are useful candidate signals, but none alone establishes ",
        "that replay will improve the protected operational tail.",
        "",
        "The most direct warning comes from training-data-attribution work showing that initialization and SGD batch ",
        "composition can overwhelm individual attribution estimates. This matches the observed Stage1 same-selection ",
        "seed reversals and makes cross-seed process measurement the first investigation, not a tenth static ranking.",
        "",
        "## Ranked directions",
        "",
        "The order below is a decision order based on causal relevance, current evidence gaps, feasibility before ",
        "2026-09-10, and ability to falsify the current mechanism. It is not a weighted score.",
        "",
        "| Rank | Direction | Decision | Papers | Deep | Method | Supports | Cautions |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{row['direction_id']}` | {row['label']} | {row['papers']} | "
            f"{row['deep']} | {row['method']} | {row['supports']} | {row['cautions']} |"
        )
    lines.extend(
        [
            "",
            "## What changes in the experiment",
            "",
            "1. Use paired unseen seeds and add `NR_NO_REPLAY`; estimate arm effects and success probability, not a best run.",
            "2. Hold the replay selection fixed while changing only its epoch schedule, so schedule is identifiable.",
            "3. Record realized per-sample exposure and role-separated losses; configured weights are not enough.",
            "4. Freeze difficult-normal and weak-defect probes and export raw trajectories at epochs 120, 140, 150, 160, 180, 200.",
            "5. Pilot last-layer gradient norm, target alignment, and gradient outlier status on a bounded subset.",
            "6. Add diversity only after the candidate signal is defined; diversity cannot repair a harmful target direction.",
            "7. Freeze checkpoint and threshold rules before opening a blind holdout.",
            "",
            "## Stop rules",
            "",
            "- Do not scale gradient collection if the pilot cannot reproduce values or predict paired tail changes.",
            "- Do not expand from 14 to 22 or 30 seeds until the 84-run matrix is complete and storage/retry capacity is proven.",
            "- Do not promote a static score that fails same-selection, paired-seed replication.",
            "- Do not use blind data to tune replay selection, schedule, guard composition, checkpoint, or threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_literature_review(
    matrix: pd.DataFrame,
    output_dir: str | Path,
    *,
    campaign_id: str,
    min_screened: int = 150,
    min_method: int = 50,
    min_deep: int = 20,
) -> dict[str, Any]:
    """Atomically publish the matrix, reading log, synthesis, and validation."""

    counts = assert_literature_evidence_matrix(
        matrix,
        min_screened=min_screened,
        min_method=min_method,
        min_deep=min_deep,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "LITERATURE_EVIDENCE_MATRIX.csv"
    reading_path = output / "READING_LOG.csv"
    synthesis_path = output / "RESEARCH_SYNTHESIS.md"
    validation_path = output / "LITERATURE_VALIDATION.json"
    ordered = matrix.sort_values(
        ["screening_depth", "topic", "year", "title"],
        ascending=[False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    reading = ordered[
        ordered["screening_depth"].isin(["METHOD_READ", "DEEP_READ"])
    ][
        [
            "evidence_id",
            "title",
            "screening_depth",
            "reading_basis",
            "sections_checked",
            "method_summary",
            "claim_boundary",
            "stage1_implication",
            "primary_url",
        ]
    ].reset_index(drop=True)
    _atomic_csv(ordered, matrix_path)
    _atomic_csv(reading, reading_path)
    _atomic_text(build_research_synthesis(ordered), synthesis_path)
    artifacts = []
    for path in (matrix_path, reading_path, synthesis_path):
        artifacts.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    validation = {
        "status": "complete",
        "campaign_id": campaign_id,
        "counts": counts,
        "artifacts": artifacts,
    }
    _atomic_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        validation_path,
    )
    return validation
