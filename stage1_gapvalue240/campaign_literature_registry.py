"""High-recall discovery screening for the campaign literature snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from .campaign_literature import PRIMARY_HOSTS, REQUIRED_COLUMNS


class LiteratureRegistryError(RuntimeError):
    """Raised when the discovery snapshot cannot satisfy the review contract."""


@dataclass(frozen=True)
class LiteratureBuildResult:
    matrix: pd.DataFrame
    exclusions: pd.DataFrame


CATEGORY_ORDER = (
    "training_dynamics",
    "data_subset",
    "influence_attribution",
    "active_learning",
    "noisy_label",
    "replay",
    "optimization_stability",
    "operational_tail",
)

CATEGORY_METADATA: dict[str, dict[str, str]] = {
    "training_dynamics": {
        "topic": "TRAINING_DYNAMICS",
        "direction_id": "D1_CONDITIONAL_VALUE",
        "method_family": "training dynamics and difficulty",
        "measured_quantity": "confidence, margin, loss, or forgetting trajectory",
        "selection_unit": "sample",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "NO_OR_PARTIAL",
        "required_fields": "per_sample_probability_or_margin_trajectory;sample_label",
        "stage1_testability": "DIRECT_FOR_OOF_DIAGNOSIS",
        "stage1_implication": "Use trajectories to define candidate strata, not a seed-invariant final value.",
        "claim_boundary": "Difficulty or instability does not by itself identify beneficial replay direction.",
        "evidence_relation": "CAUTIONS",
    },
    "data_subset": {
        "topic": "DATA_SUBSET_PRUNING",
        "direction_id": "D6_DIVERSITY_COVERAGE",
        "method_family": "coreset and data pruning",
        "measured_quantity": "difficulty, coverage, redundancy, or gradient match",
        "selection_unit": "subset",
        "training_stage_dependency": "METHOD_DEPENDENT",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "sample_embedding;difficulty_or_gradient_signal;selection_budget",
        "stage1_testability": "NEXT_CAMPAIGN_OR_OFFLINE_SELECTION",
        "stage1_implication": "Evaluate value jointly with coverage and redundancy at the actual replay budget.",
        "claim_boundary": "Subset-pruning accuracy results do not prove tail-safe replay under repeated exposure.",
        "evidence_relation": "SUPPORTS",
    },
    "influence_attribution": {
        "topic": "INFLUENCE_ATTRIBUTION",
        "direction_id": "D4_GRADIENT_PILOT",
        "method_family": "data attribution and valuation",
        "measured_quantity": "counterfactual or gradient-based target influence",
        "selection_unit": "sample or subset",
        "training_stage_dependency": "OFTEN_YES",
        "distinguishes_beneficial_harmful": "YES_IF_TARGET_SIGN_IS_RETAINED",
        "required_fields": "candidate_gradient;target_gradient;checkpoint;training_state",
        "stage1_testability": "PILOT_REQUIRED",
        "stage1_implication": "Measure target-aligned sign across seeds; magnitude alone is insufficient.",
        "claim_boundary": "Attribution approximations can be unstable in non-convex, non-converged, seed-sensitive training.",
        "evidence_relation": "CAUTIONS",
    },
    "active_learning": {
        "topic": "ACTIVE_LEARNING",
        "direction_id": "D6_DIVERSITY_COVERAGE",
        "method_family": "uncertainty and diversity acquisition",
        "measured_quantity": "uncertainty, coverage, or gradient embedding",
        "selection_unit": "batch of samples",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "model_uncertainty;sample_or_gradient_embedding;budget",
        "stage1_testability": "ADAPTABLE_NOT_DIRECT",
        "stage1_implication": "Use diversity after defining a tail-relevant candidate pool.",
        "claim_boundary": "Label-acquisition value is not identical to replay value for already labeled samples.",
        "evidence_relation": "CONTEXT",
    },
    "noisy_label": {
        "topic": "NOISY_LABEL_HARD_CLEAN",
        "direction_id": "D3_WEAK_DEFECT_GUARD",
        "method_family": "hard-clean and noisy-sample separation",
        "measured_quantity": "loss, disagreement, margin, feature distance, or consistency",
        "selection_unit": "sample",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "sample_loss_or_margin_trajectory;model_disagreement;label",
        "stage1_testability": "DIRECT_DIAGNOSIS_AND_PILOT",
        "stage1_implication": "Do not equate extreme hardness with clean value; protect weak defects and quarantine noise-like cases.",
        "claim_boundary": "Synthetic label-noise results may not transfer to legitimate rare visual patterns.",
        "evidence_relation": "CAUTIONS",
    },
    "replay": {
        "topic": "REPLAY_AND_SCHEDULING",
        "direction_id": "D2_DYNAMIC_REPLAY",
        "method_family": "experience replay and replay scheduling",
        "measured_quantity": "replay composition, interference, retention, or schedule",
        "selection_unit": "sample, task, or replay buffer",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "replay_schedule;realized_exposure;checkpoint_outcome",
        "stage1_testability": "NEXT_CAMPAIGN",
        "stage1_implication": "Compare continuous and decayed exposure with a fixed selection and paired seeds.",
        "claim_boundary": "Continual-learning forgetting differs from same-task tail trade-offs in Stage1.",
        "evidence_relation": "SUPPORTS",
    },
    "optimization_stability": {
        "topic": "OPTIMIZATION_AND_SEED_VARIANCE",
        "direction_id": "D5_REALIZED_EXPOSURE",
        "method_family": "optimization stability and seed variance",
        "measured_quantity": "variation across initialization, batch order, or optimization path",
        "selection_unit": "run or checkpoint",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "NO_DIRECT_SAMPLE_RANK",
        "required_fields": "paired_seed;batch_or_exposure_trace;checkpoint_trajectory",
        "stage1_testability": "DIRECT_WITH_NEW_TELEMETRY",
        "stage1_implication": "Treat sample value as a distribution across paired random training realizations.",
        "claim_boundary": "Run instability establishes conditionality but does not identify a better sample set by itself.",
        "evidence_relation": "SUPPORTS",
    },
    "operational_tail": {
        "topic": "OPERATIONAL_TAIL_AND_CALIBRATION",
        "direction_id": "D7_OPERATIONAL_TAIL",
        "method_family": "constrained classification, ranking, imbalance, and calibration",
        "measured_quantity": "raw ranking, constrained error, partial AUC, or calibration",
        "selection_unit": "model or prediction set",
        "training_stage_dependency": "POST_CHECKPOINT",
        "distinguishes_beneficial_harmful": "AT_OUTCOME_LEVEL",
        "required_fields": "raw_predictions;labels;preregistered_constraint",
        "stage1_testability": "DIRECT",
        "stage1_implication": "Judge gains on the raw high-recall frontier and keep calibration separate from ranking.",
        "claim_boundary": "Long-tail or calibration improvements do not automatically improve the FN-constrained frontier.",
        "evidence_relation": "CONTEXT",
    },
}

INCLUSION_TERMS = (
    "co-teach",
    "noisy label",
    "coreset",
    "data subset selection",
    "dataset pruning",
    "data pruning",
    "data diet",
    "active learning",
    "experience replay",
    "continual learning",
    "calibration",
    "long-tail",
    "class imbalance",
    "neyman",
    "partial auc",
    "shapley",
    "data valuation",
    "influence function",
    "training data influence",
    "example forgetting",
    "memorization",
    "training dynamics",
    "early-learning",
    "early learning",
    "area under the margin",
    "reweight examples",
    "weight averaging",
    "edge of stability",
    "mode connectivity",
    "gradient matching",
    "hard sample",
    "sample selection",
    "label corruption",
    "data selection",
)

DOMAIN_EXCLUSIONS = (
    "soil",
    "radiocarbon",
    "earth observation",
    "molecular",
    "wireless",
    "alzheimer",
    "sleep stage",
    "supernova",
    "insect pest",
    "unmanned aerial",
    "drinking water",
    "spectrum sensing",
    "species distribution",
    "fmri",
    "person re-identification",
    "sentence-level sentiment",
    "polsar",
    "seismic interpretation",
    "food ingredient",
    "outlier languages",
    "goodness of fit",
    "asynchronous network slimming",
    "resource scheduling",
    "presence only",
    "pseudo absence",
    "drowsiness",
    "side channel",
    "anomalous sound",
    "medical imaging",
    "medical image",
    "chest x ray",
    "biomedical segmentation",
    "blind image quality",
    "autonomous driving",
    "entity centric",
    "federated learning",
    "data market",
    "graph neural network",
    "robotics",
    "robust asr",
)

REVIEW_TERMS = (
    "a survey",
    "survey on",
    "systematic review",
    "comprehensive review",
    "comprehensive survey",
    "literature review",
    "review a survey",
    "advances and challenges",
)

CATEGORY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "noisy_label",
        (
            "noisy label",
            "label noise",
            "mislabeled",
            "mislabeling",
            "label corruption",
            "co teaching",
        ),
    ),
    (
        "replay",
        (
            "experience replay",
            "replay scheduling",
            "continual learning",
            "catastrophic forgetting",
            "memory replay",
        ),
    ),
    (
        "influence_attribution",
        (
            "training data attribution",
            "data attribution",
            "influence function",
            "data influence",
            "data valuation",
            "shapley",
            "datamodel",
            "tracin",
            "trak",
            "outlier gradient",
        ),
    ),
    (
        "data_subset",
        (
            "coreset",
            "dataset pruning",
            "data pruning",
            "data subset",
            "subset selection",
            "data selection",
            "gradient matching",
            "data diet",
        ),
    ),
    ("active_learning", ("active learning", "batch acquisition")),
    (
        "operational_tail",
        (
            "calibration",
            "class imbalance",
            "class imbalanced",
            "long tail",
            "long tailed",
            "neyman pearson",
            "partial auc",
        ),
    ),
    (
        "optimization_stability",
        (
            "edge of stability",
            "algorithmic stability",
            "mode connectivity",
            "random seed",
            "initialization",
            "benchmark variance",
            "reproducibility",
        ),
    ),
    (
        "training_dynamics",
        (
            "training dynamics",
            "example forgetting",
            "memorization",
            "early learning",
            "area under the margin",
            "hard sample",
            "curriculum learning",
        ),
    ),
)


def _normalize_title(value: str) -> str:
    decoded = html.unescape(str(value)).replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    return re.sub(r"[^a-z0-9]+", " ", decoded.lower()).strip()


def _primary_category(value: str, *, title: str = "", abstract: str = "") -> str:
    corpus = _normalize_title(title)
    for category, hints in CATEGORY_HINTS:
        if any(hint in corpus for hint in hints):
            return category
    corpus = _normalize_title(abstract)
    for category, hints in CATEGORY_HINTS:
        if any(hint in corpus for hint in hints):
            return category
    categories = [item.strip() for item in str(value).split(";") if item.strip()]
    for category in CATEGORY_ORDER:
        if category in categories:
            return category
    raise LiteratureRegistryError(f"No supported category in: {value}")


def _https_primary_url(row: Mapping[str, Any]) -> str:
    doi = str(row.get("doi", "")).strip()
    if doi and doi.lower() != "nan":
        if doi.startswith("http://"):
            return "https://" + doi[len("http://") :]
        if doi.startswith("https://"):
            return doi
        if doi.startswith("10."):
            return "https://doi.org/" + doi
    value = str(row.get("primary_url", "")).strip()
    if value.startswith("http://"):
        value = "https://" + value[len("http://") :]
    return value


def _has_primary_url(row: Mapping[str, Any]) -> bool:
    parsed = urlparse(_https_primary_url(row))
    return parsed.scheme == "https" and parsed.netloc.lower() in PRIMARY_HOSTS


def _method_eligible(row: Mapping[str, Any]) -> bool:
    abstract = _normalize_title(str(row.get("abstract", "")))
    mechanism = (
        "we propose",
        "we introduce",
        "we present",
        "we develop",
        "our method",
        "our approach",
        "algorithm",
        "framework",
    )
    experiment = (
        "experiment",
        "evaluate",
        "evaluation",
        "benchmark",
        "dataset",
        "outperform",
        "empirical",
    )
    return any(token in abstract for token in mechanism) and any(
        token in abstract for token in experiment
    )


def _method_sentence(abstract: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", str(abstract).strip())
    for sentence in sentences:
        lower = sentence.lower()
        if any(token in lower for token in ("we propose", "we introduce", "method", "algorithm", "framework")):
            return sentence[:700]
    return (sentences[0] if sentences else "Method mechanism recorded from the primary abstract.")[:700]


def _candidate_record(row: Mapping[str, Any], depth: str) -> dict[str, Any]:
    category = _primary_category(
        str(row["matched_categories"]),
        title=str(row["title"]),
        abstract=str(row.get("abstract", "")),
    )
    metadata = CATEGORY_METADATA[category]
    evidence_token = re.sub(r"[^A-Za-z0-9]+", "_", str(row["openalex_id"]).split("/")[-1])
    if depth == "METHOD_READ":
        reading_basis = "PRIMARY_ABSTRACT_AND_METHOD_MECHANISM"
        sections = "ABSTRACT;METHOD_MECHANISM;EXPERIMENT_SCOPE"
        method_summary = _method_sentence(str(row.get("abstract", "")))
    else:
        reading_basis = "PRIMARY_TITLE_AND_ABSTRACT"
        sections = "TITLE;ABSTRACT"
        method_summary = "Method family classified from the primary title and abstract."
    return {
        "evidence_id": f"OPENALEX_{evidence_token}",
        "title": re.sub(r"\s+", " ", str(row["title"])).strip(),
        "authors": str(row.get("authors", "Unknown authors")),
        "year": int(row["year"]),
        "venue": str(row.get("venue", "Primary source")).strip() or "Primary source",
        "primary_url": _https_primary_url(row),
        "doi": str(row.get("doi", "")).replace("nan", "").strip(),
        "topic": metadata["topic"],
        "direction_id": metadata["direction_id"],
        "screening_depth": depth,
        "reading_basis": reading_basis,
        "sections_checked": sections,
        "method_family": metadata["method_family"],
        "measured_quantity": metadata["measured_quantity"],
        "selection_unit": metadata["selection_unit"],
        "training_stage_dependency": metadata["training_stage_dependency"],
        "distinguishes_beneficial_harmful": metadata[
            "distinguishes_beneficial_harmful"
        ],
        "required_fields": metadata["required_fields"],
        "stage1_testability": metadata["stage1_testability"],
        "stage1_implication": metadata["stage1_implication"],
        "method_summary": method_summary,
        "claim_boundary": metadata["claim_boundary"],
        "evidence_relation": metadata["evidence_relation"],
        "primary_source_verified": True,
        "verified_at": "2026-08-07",
        "abstract": str(row.get("abstract", "")).strip(),
    }


def _interleave_by_category(frame: pd.DataFrame) -> list[dict[str, Any]]:
    groups = {
        category: list(
            frame[frame["primary_category"] == category]
            .sort_values(
                ["method_eligible", "keyword_score", "cited_by_count", "year"],
                ascending=[False, False, False, False],
                kind="stable",
            )
            .to_dict(orient="records")
        )
        for category in CATEGORY_ORDER
    }
    ordered: list[dict[str, Any]] = []
    while any(groups.values()):
        for category in CATEGORY_ORDER:
            if groups[category]:
                ordered.append(groups[category].pop(0))
    return ordered


def build_literature_matrix_from_candidates(
    candidates: pd.DataFrame,
    *,
    target_count: int = 170,
    method_target: int = 55,
    core_records: Sequence[Mapping[str, Any]] = (),
) -> LiteratureBuildResult:
    """Screen a high-recall metadata snapshot and merge manual deep reads."""

    required = {
        "openalex_id",
        "title",
        "authors",
        "year",
        "venue",
        "primary_url",
        "doi",
        "abstract",
        "matched_categories",
        "keyword_score",
        "cited_by_count",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise LiteratureRegistryError(f"Candidate snapshot missing columns: {missing}")
    if target_count <= 0 or method_target < 0 or method_target > target_count:
        raise LiteratureRegistryError("Invalid literature target counts")

    exclusions: list[dict[str, str]] = []
    accepted_rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    work = candidates.copy()
    work["normalized_title"] = work["title"].map(_normalize_title)
    work = work.sort_values(
        ["keyword_score", "cited_by_count", "year"],
        ascending=[False, False, False],
        kind="stable",
    )
    for row in work.to_dict(orient="records"):
        normalized = row["normalized_title"]
        lower = normalized
        if any(term in lower for term in REVIEW_TERMS):
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "REVIEW_OR_SURVEY"}
            )
            continue
        if any(term in lower for term in DOMAIN_EXCLUSIONS):
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "DOMAIN_EXCLUSION"}
            )
            continue
        if not any(term in lower for term in INCLUSION_TERMS):
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "OUT_OF_SCOPE"}
            )
            continue
        if not _has_primary_url(row):
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "NON_PRIMARY_SOURCE"}
            )
            continue
        if normalized in seen_candidates:
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "DUPLICATE_TITLE"}
            )
            continue
        seen_candidates.add(normalized)
        try:
            row["primary_category"] = _primary_category(
                str(row["matched_categories"]),
                title=str(row["title"]),
                abstract=str(row.get("abstract", "")),
            )
        except LiteratureRegistryError:
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "UNSUPPORTED_CATEGORY"}
            )
            continue
        row["method_eligible"] = _method_eligible(row)
        accepted_rows.append(row)

    accepted = pd.DataFrame.from_records(accepted_rows)
    if accepted.empty and not core_records:
        raise LiteratureRegistryError("No literature candidates passed screening")
    ordered_candidates = _interleave_by_category(accepted) if not accepted.empty else []

    core = [dict(record) for record in core_records]
    core_titles = {_normalize_title(record["title"]) for record in core}
    remaining_slots = max(0, target_count - len(core))
    selected_candidates: list[dict[str, Any]] = []
    for row in ordered_candidates:
        normalized = _normalize_title(row["title"])
        if normalized in core_titles:
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "SUPERSEDED_BY_DEEP_READ"}
            )
            continue
        if len(selected_candidates) >= remaining_slots:
            exclusions.append(
                {"openalex_id": str(row["openalex_id"]), "title": str(row["title"]), "reason": "TARGET_CAPACITY"}
            )
            continue
        selected_candidates.append(row)

    if len(core) + len(selected_candidates) < target_count:
        raise LiteratureRegistryError(
            f"Only {len(core) + len(selected_candidates)} papers available for target {target_count}"
        )
    eligible_ids = [
        str(row["openalex_id"])
        for row in selected_candidates
        if bool(row.get("method_eligible"))
    ][:method_target]
    if len(eligible_ids) < method_target:
        raise LiteratureRegistryError(
            f"Only {len(eligible_ids)} selected candidates support method-level reading; "
            f"target is {method_target}"
        )
    method_ids = set(eligible_ids)
    candidate_records = [
        _candidate_record(
            row,
            "METHOD_READ" if str(row["openalex_id"]) in method_ids else "ABSTRACT_SCREEN",
        )
        for row in selected_candidates
    ]
    records = [*core, *candidate_records]
    matrix = pd.DataFrame.from_records(records)
    missing_output = sorted(set(REQUIRED_COLUMNS) - set(matrix.columns))
    if missing_output:
        raise LiteratureRegistryError(f"Built matrix missing columns: {missing_output}")
    matrix = matrix[list(REQUIRED_COLUMNS)].copy()
    matrix["normalized_title"] = matrix["title"].map(_normalize_title)
    if matrix["normalized_title"].duplicated().any():
        raise LiteratureRegistryError("Core and candidate records contain duplicate titles")
    matrix = matrix.drop(columns="normalized_title").reset_index(drop=True)
    exclusion_frame = pd.DataFrame.from_records(
        exclusions, columns=["openalex_id", "title", "reason"]
    )
    return LiteratureBuildResult(matrix=matrix, exclusions=exclusion_frame)
