from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_literature import (
    LiteratureReviewError,
    assert_literature_evidence_matrix,
    build_research_synthesis,
    publish_literature_review,
)
from stage1_gapvalue240.campaign_literature_registry import (
    build_literature_matrix_from_candidates,
)
from stage1_gapvalue240.campaign_literature_core import build_deep_read_records


def _row(
    evidence_id: str,
    title: str,
    depth: str,
    direction: str,
    relation: str = "SUPPORTS",
) -> dict[str, object]:
    deep = depth == "DEEP_READ"
    method = depth in {"METHOD_READ", "DEEP_READ"}
    return {
        "evidence_id": evidence_id,
        "title": title,
        "authors": "A. Author; B. Author",
        "year": 2024,
        "venue": "Primary Proceedings",
        "primary_url": f"https://proceedings.mlr.press/{evidence_id}.html",
        "doi": "",
        "topic": "TRAINING_DYNAMICS",
        "direction_id": direction,
        "screening_depth": depth,
        "reading_basis": (
            "FULL_PRIMARY_PAPER"
            if deep
            else "PRIMARY_METHOD_AND_EXPERIMENT_DESCRIPTION"
            if method
            else "PRIMARY_TITLE_AND_ABSTRACT"
        ),
        "sections_checked": (
            "ABSTRACT;METHOD;EXPERIMENTS;LIMITATIONS"
            if deep
            else "ABSTRACT;METHOD_MECHANISM;EXPERIMENT_SCOPE"
            if method
            else "TITLE;ABSTRACT"
        ),
        "method_family": "training dynamics",
        "measured_quantity": "sample trajectory",
        "selection_unit": "sample",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "per_sample_probability_trajectory",
        "stage1_testability": "DIRECT_OR_NEXT_CAMPAIGN",
        "stage1_implication": "Record process trajectories across paired seeds.",
        "method_summary": "Uses trajectory variation rather than one endpoint.",
        "claim_boundary": "Does not establish a seed-invariant scalar value.",
        "evidence_relation": relation,
        "primary_source_verified": True,
        "verified_at": "2026-08-07",
        "abstract": "A primary abstract describing the method and experiment.",
    }


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("deep1", "Deep paper", "DEEP_READ", "D1_CONDITIONAL_VALUE"),
            _row("method1", "Method paper", "METHOD_READ", "D2_DYNAMIC_REPLAY"),
            _row("method2", "Guard paper", "METHOD_READ", "D3_WEAK_DEFECT_GUARD"),
            _row("abs1", "Abstract paper", "ABSTRACT_SCREEN", "D4_GRADIENT_PILOT"),
            _row(
                "abs2",
                "Caution paper",
                "ABSTRACT_SCREEN",
                "D1_CONDITIONAL_VALUE",
                relation="CAUTIONS",
            ),
        ]
    )


def test_literature_matrix_enforces_depth_and_primary_source_contract() -> None:
    matrix = _matrix()

    gates = assert_literature_evidence_matrix(
        matrix, min_screened=5, min_method=2, min_deep=1
    )

    assert gates == {
        "screened": 5,
        "method_read": 2,
        "deep_read": 1,
        "duplicate_evidence_ids": 0,
        "duplicate_normalized_titles": 0,
        "invalid_primary_urls": 0,
        "missing_required_metadata": 0,
        "invalid_reading_evidence": 0,
    }


def test_literature_matrix_rejects_duplicate_preprint_and_search_url() -> None:
    matrix = pd.concat(
        [
            _matrix(),
            pd.DataFrame(
                [
                    {
                        **_row(
                            "duplicate",
                            "Deep Paper!",
                            "ABSTRACT_SCREEN",
                            "D1_CONDITIONAL_VALUE",
                        ),
                        "primary_url": "https://google.com/search?q=paper",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        LiteratureReviewError,
        match="duplicate_normalized_titles=1.*invalid_primary_urls=1",
    ):
        assert_literature_evidence_matrix(
            matrix, min_screened=5, min_method=2, min_deep=1
        )


def test_deep_read_requires_method_experiment_and_limitation_evidence() -> None:
    matrix = _matrix()
    matrix.loc[matrix["evidence_id"] == "deep1", "sections_checked"] = "ABSTRACT"

    with pytest.raises(LiteratureReviewError, match="invalid_reading_evidence=1"):
        assert_literature_evidence_matrix(
            matrix, min_screened=5, min_method=2, min_deep=1
        )


def test_synthesis_ranks_conditional_value_before_static_gradient_magnitude() -> None:
    synthesis = build_research_synthesis(_matrix())

    assert "D1_CONDITIONAL_VALUE" in synthesis
    assert "D4_GRADIENT_PILOT" in synthesis
    assert synthesis.index("D1_CONDITIONAL_VALUE") < synthesis.index(
        "D4_GRADIENT_PILOT"
    )
    assert "not a seed-invariant scalar" in synthesis


def test_publish_writes_matrix_reading_log_synthesis_and_validation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "02_literature"

    result = publish_literature_review(
        _matrix(),
        output,
        campaign_id="dynamic_replay_budget_efficiency_20260807",
        min_screened=5,
        min_method=2,
        min_deep=1,
    )

    assert {path.name for path in output.iterdir()} == {
        "LITERATURE_EVIDENCE_MATRIX.csv",
        "READING_LOG.csv",
        "RESEARCH_SYNTHESIS.md",
        "LITERATURE_VALIDATION.json",
    }
    assert result["status"] == "complete"
    reading = pd.read_csv(output / "READING_LOG.csv")
    assert len(reading) == 3
    validation = json.loads(
        (output / "LITERATURE_VALIDATION.json").read_text(encoding="utf-8")
    )
    assert validation["counts"]["deep_read"] == 1


def test_candidate_screening_deduplicates_and_keeps_exclusion_reasons() -> None:
    candidates = pd.DataFrame(
        [
            {
                "openalex_id": "W1",
                "title": "Dataset Pruning with Training Dynamics",
                "authors": "A. Author",
                "year": 2024,
                "venue": "Primary Venue",
                "primary_url": "http://arxiv.org/abs/2401.00001",
                "doi": "https://doi.org/10.1000/one",
                "abstract": "We propose a dataset pruning method based on training dynamics and evaluate it.",
                "matched_categories": "data_subset;training_dynamics",
                "keyword_score": 5,
                "cited_by_count": 20,
            },
            {
                "openalex_id": "W2",
                "title": "Dataset pruning with training-dynamics!",
                "authors": "A. Author",
                "year": 2023,
                "venue": "Preprint",
                "primary_url": "http://arxiv.org/abs/2301.00001",
                "doi": "",
                "abstract": "Duplicate preprint of the same method.",
                "matched_categories": "data_subset",
                "keyword_score": 4,
                "cited_by_count": 2,
            },
            {
                "openalex_id": "W3",
                "title": "Experience Replay for Continual Learning",
                "authors": "B. Author",
                "year": 2019,
                "venue": "Primary Venue",
                "primary_url": "https://doi.org/10.1000/two",
                "doi": "https://doi.org/10.1000/two",
                "abstract": "We evaluate experience replay and its sample composition over training.",
                "matched_categories": "replay",
                "keyword_score": 4,
                "cited_by_count": 100,
            },
            {
                "openalex_id": "W4",
                "title": "Soil Data Pruning for Tropical Agriculture",
                "authors": "C. Author",
                "year": 2020,
                "venue": "Agriculture",
                "primary_url": "https://doi.org/10.1000/soil",
                "doi": "https://doi.org/10.1000/soil",
                "abstract": "A soil mapping paper.",
                "matched_categories": "data_subset",
                "keyword_score": 3,
                "cited_by_count": 10,
            },
        ]
    )

    result = build_literature_matrix_from_candidates(
        candidates,
        target_count=2,
        method_target=1,
        core_records=(),
    )

    assert len(result.matrix) == 2
    assert result.matrix["screening_depth"].eq("METHOD_READ").sum() == 1
    assert result.matrix["title"].str.contains("Soil", case=False).sum() == 0
    assert set(result.exclusions["reason"]) >= {"DOMAIN_EXCLUSION", "DUPLICATE_TITLE"}
    assert result.matrix["primary_url"].str.startswith("https://").all()


def test_deep_read_core_is_large_unique_primary_and_mechanism_complete() -> None:
    core = pd.DataFrame.from_records(build_deep_read_records())

    assert len(core) >= 24
    assert core["screening_depth"].eq("DEEP_READ").all()
    assert core["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).is_unique
    assert core["primary_source_verified"].eq(True).all()  # noqa: E712
    assert core["primary_url"].str.startswith("https://").all()
    assert core["sections_checked"].str.contains("METHOD").all()
    assert core["sections_checked"].str.contains("EXPERIMENT").all()
    assert core["sections_checked"].str.contains("LIMITATION").all()
    assert {
        "D1_CONDITIONAL_VALUE",
        "D2_DYNAMIC_REPLAY",
        "D3_WEAK_DEFECT_GUARD",
        "D4_GRADIENT_PILOT",
        "D5_REALIZED_EXPOSURE",
        "D6_DIVERSITY_COVERAGE",
        "D7_OPERATIONAL_TAIL",
    }.issubset(set(core["direction_id"]))
    titles = "\n".join(core["title"])
    assert "Bayesian Approach To Analysing Training Data Attribution" in titles
    assert "Distributional Training Data Attribution" in titles
    assert "Replay Scheduling" in titles


def test_screening_normalizes_linebreaks_and_rejects_surveys_and_non_primary() -> None:
    candidates = pd.DataFrame(
        [
            {
                "openalex_id": "W1",
                "title": "Important Dataset\\n Pruning",
                "authors": "A. Author",
                "year": 2024,
                "venue": "Primary Venue",
                "primary_url": "https://arxiv.org/abs/2401.00001",
                "doi": "",
                "abstract": "We propose a dataset pruning method and evaluate it on two datasets.",
                "matched_categories": "data_subset",
                "keyword_score": 10,
                "cited_by_count": 20,
            },
            {
                "openalex_id": "W2",
                "title": "Experience Replay from a Non-primary Index",
                "authors": "B. Author",
                "year": 2023,
                "venue": "Index",
                "primary_url": "https://dblp.uni-trier.de/rec/conf/test/one",
                "doi": "",
                "abstract": "We propose a replay method and report experiments.",
                "matched_categories": "replay",
                "keyword_score": 9,
                "cited_by_count": 10,
            },
            {
                "openalex_id": "W3",
                "title": "A Survey on Dataset Pruning",
                "authors": "C. Author",
                "year": 2024,
                "venue": "Primary Venue",
                "primary_url": "https://doi.org/10.1000/survey",
                "doi": "https://doi.org/10.1000/survey",
                "abstract": "We survey data pruning methods.",
                "matched_categories": "data_subset",
                "keyword_score": 8,
                "cited_by_count": 5,
            },
            {
                "openalex_id": "W4",
                "title": "Online Dataset Pruning with Gradient Correction",
                "authors": "D. Author",
                "year": 2025,
                "venue": "Primary Venue",
                "primary_url": "https://arxiv.org/abs/2501.00001",
                "doi": "",
                "abstract": "We propose an online dataset pruning method and evaluate it on three benchmarks.",
                "matched_categories": "optimization_stability",
                "keyword_score": 4,
                "cited_by_count": 1,
            },
        ]
    )
    core = [_row("deep_core", "Important Dataset Pruning", "DEEP_READ", "D6_DIVERSITY_COVERAGE")]

    result = build_literature_matrix_from_candidates(
        candidates,
        target_count=2,
        method_target=1,
        core_records=core,
    )

    assert set(result.matrix["title"]) == {
        "Important Dataset Pruning",
        "Online Dataset Pruning with Gradient Correction",
    }
    selected = result.matrix[result.matrix["screening_depth"].eq("METHOD_READ")]
    assert selected["title"].tolist() == [
        "Online Dataset Pruning with Gradient Correction"
    ]
    assert selected["direction_id"].tolist() == ["D6_DIVERSITY_COVERAGE"]
    assert set(result.exclusions["reason"]) >= {
        "NON_PRIMARY_SOURCE",
        "REVIEW_OR_SURVEY",
        "SUPERSEDED_BY_DEEP_READ",
    }
