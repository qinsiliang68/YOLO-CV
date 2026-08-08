from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_literature_pipeline import (
    LiteraturePipelineError,
    publish_campaign_literature,
)


def _candidate(candidate_id: str, title: str, score: int) -> dict[str, object]:
    return {
        "category_query": "data_subset",
        "query": "dataset pruning training dynamics",
        "openalex_id": candidate_id,
        "title": title,
        "authors": "A. Author",
        "year": 2024,
        "venue": "Primary Venue",
        "primary_url": f"https://arxiv.org/abs/2401.{score:05d}",
        "doi": "",
        "abstract": "We propose a dataset pruning method and evaluate it on benchmark datasets.",
        "matched_categories": "data_subset",
        "matched_queries": "dataset pruning training dynamics",
        "keyword_score": score,
        "cited_by_count": score,
        "type": "article",
    }


def _deep_record() -> dict[str, object]:
    return {
        "evidence_id": "DEEP_ONE",
        "title": "A Deep Training Dynamics Study",
        "authors": "D. Author",
        "year": 2023,
        "venue": "Primary Proceedings",
        "primary_url": "https://proceedings.mlr.press/v1/deep.html",
        "doi": "",
        "topic": "TRAINING_DYNAMICS",
        "direction_id": "D1_CONDITIONAL_VALUE",
        "screening_depth": "DEEP_READ",
        "reading_basis": "PRIMARY_PAPER_METHOD_EXPERIMENTS_AND_SCOPE",
        "sections_checked": "ABSTRACT;METHOD;EXPERIMENTS;LIMITATIONS_OR_SCOPE_BOUNDARY",
        "method_family": "training dynamics",
        "measured_quantity": "sample trajectory",
        "selection_unit": "sample",
        "training_stage_dependency": "YES",
        "distinguishes_beneficial_harmful": "PARTIAL",
        "required_fields": "per_sample_trajectory",
        "stage1_testability": "DIRECT",
        "stage1_implication": "Measure process trajectories over seeds.",
        "method_summary": "Compares sample trajectories across training.",
        "claim_boundary": "Does not establish a fixed scalar value.",
        "evidence_relation": "SUPPORTS",
        "primary_source_verified": True,
        "verified_at": "2026-08-07",
        "abstract": "A primary study of training trajectories.",
    }


def test_campaign_literature_pipeline_publishes_traceable_atomic_snapshot(
    tmp_path: Path,
) -> None:
    candidates = pd.DataFrame(
        [
            _candidate("W1", "Dataset Pruning One", 9),
            _candidate("W2", "Dataset Pruning Two", 8),
        ]
    )
    source = tmp_path / "candidates.csv"
    candidates.to_csv(source, index=False)
    output = tmp_path / "02_literature"

    receipt = publish_campaign_literature(
        source,
        output,
        campaign_id="campaign_test",
        core_records=[_deep_record()],
        target_count=3,
        method_target=1,
        min_screened=3,
        min_method=1,
        min_deep=1,
        discovery_counts={"raw_results": 10, "deduplicated": 7, "shortlisted": 2},
    )

    expected = {
        "discovery/OPENALEX_CANDIDATES.csv",
        "discovery/DISCOVERY_QUERY_LOG.csv",
        "discovery/DISCOVERY_PROVENANCE.json",
        "SCREENING_EXCLUSIONS.csv",
        "LITERATURE_EVIDENCE_MATRIX.csv",
        "READING_LOG.csv",
        "RESEARCH_SYNTHESIS.md",
        "LITERATURE_VALIDATION.json",
        "LITERATURE_REVIEW_METHOD.md",
    }
    assert expected.issubset(
        {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    )
    assert receipt["status"] == "complete"
    assert receipt["counts"]["screened"] == 3
    provenance = json.loads(
        (output / "discovery/DISCOVERY_PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert provenance["discovery_counts"]["shortlisted"] == 2
    assert provenance["source_sha256"]
    assert not list(output.rglob("*.tmp"))

    with pytest.raises(LiteraturePipelineError, match="refusing to overwrite"):
        publish_campaign_literature(
            source,
            output,
            campaign_id="campaign_test",
            core_records=[_deep_record()],
            target_count=3,
            method_target=1,
            min_screened=3,
            min_method=1,
            min_deep=1,
        )
