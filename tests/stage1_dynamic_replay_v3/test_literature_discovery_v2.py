import csv
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_discovery_v2 import (
    build_manual_screen_queue,
    DiscoveryError,
    build_candidate_inventory,
    load_query_plan,
    reconstruct_openalex_abstract,
    triage_candidate_inventory,
)


def _work(
    work_id: str,
    title: str,
    *,
    doi: str | None,
    query_score: float,
) -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{work_id}",
        "doi": doi,
        "title": title,
        "display_name": title,
        "publication_year": 2024,
        "type": "article",
        "relevance_score": query_score,
        "cited_by_count": 12,
        "authorships": [
            {"author": {"display_name": "Alice Example"}},
            {"author": {"display_name": "Bob Example"}},
        ],
        "primary_location": {
            "landing_page_url": "https://publisher.example/paper",
            "source": {"display_name": "Fixture Proceedings"},
        },
        "open_access": {"oa_url": "https://publisher.example/paper.pdf"},
        "abstract_inverted_index": {
            "Samples": [0],
            "can": [1],
            "be": [2],
            "learnable": [3],
            "or": [4],
            "harmful.": [5],
        },
    }


def test_query_plan_requires_unique_ids_and_research_question_mapping(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "queries": [
                    {
                        "query_id": "Q001",
                        "database": "OPENALEX",
                        "query": "sample learnability training dynamics",
                        "rq_ids": ["RQ1", "RQ2"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    specs = load_query_plan(path)
    assert specs[0].query_id == "Q001"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["queries"].append(dict(payload["queries"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DiscoveryError, match="duplicate"):
        load_query_plan(path)


def test_openalex_abstract_is_reconstructed_by_token_position() -> None:
    assert reconstruct_openalex_abstract({"second": [1], "first": [0], "last": [2]}) == (
        "first second last"
    )
    assert reconstruct_openalex_abstract({"first": [0], "third": [5]}) == "first third"
    assert reconstruct_openalex_abstract(None) == "NOT_REPORTED_BY_SOURCE"


def test_candidate_inventory_merges_query_hits_and_versions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    first = {
        "meta": {"count": 2},
        "results": [
            _work(
                "W1",
                "Learning Which Samples Are Worth Learning",
                doi="https://doi.org/10.1000/example",
                query_score=91.0,
            ),
            _work(
                "W2",
                "Coverage Matters More Than Difficulty",
                doi=None,
                query_score=70.0,
            ),
        ],
    }
    second = {
        "meta": {"count": 1},
        "results": [
            _work(
                "W9",
                "Learning Which Samples Are Worth Learning",
                doi="https://doi.org/10.1000/example-preprint",
                query_score=88.0,
            )
        ],
    }
    (raw / "Q001.json").write_text(json.dumps(first), encoding="utf-8")
    (raw / "Q002.json").write_text(json.dumps(second), encoding="utf-8")
    snapshots = [
        {"query_id": "Q001", "snapshot_path": str(raw / "Q001.json")},
        {"query_id": "Q002", "snapshot_path": str(raw / "Q002.json")},
    ]

    rows = build_candidate_inventory(snapshots)

    assert len(rows) == 2
    merged = next(row for row in rows if row["title"] == "Learning Which Samples Are Worth Learning")
    assert merged["doi"] == "10.1000/example;10.1000/example-preprint"
    assert merged["query_ids"] == "Q001;Q002"
    assert merged["openalex_ids"] == "W1;W9"
    assert merged["max_relevance_score"] == 91.0
    assert "learnable or harmful" in merged["abstract"]


def test_candidate_inventory_rejects_corrupt_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "bad.json"
    snapshot.write_text('{"results": "not-a-list"}', encoding="utf-8")
    with pytest.raises(DiscoveryError, match="results"):
        build_candidate_inventory([{"query_id": "Q001", "snapshot_path": str(snapshot)}])


def test_triage_uses_explicit_mechanism_gates_without_weighted_score() -> None:
    rows = [
        {
            "candidate_key": "C1",
            "title": "Prioritized Training on Points that are Learnable and Worth Learning",
            "abstract": "We select training examples using reducible loss and compare model learning.",
        },
        {
            "candidate_key": "C2",
            "title": "A Survey of Data Selection",
            "abstract": "This survey reviews machine learning training data selection.",
        },
        {
            "candidate_key": "C3",
            "title": "Entrepreneurial Learning Dynamics in Knowledge-Intensive Firms",
            "abstract": "We interview managers about organizational learning.",
        },
        {
            "candidate_key": "C4",
            "title": "Training Data Selection for Industrial Fault Classification",
            "abstract": "A classifier is trained after selecting a finite subset of labeled samples.",
        },
        {
            "candidate_key": "C5",
            "title": "Sample Selection Bias in Econometrics",
            "abstract": "A statistical correction for survey population bias.",
        },
    ]

    triaged = {row["candidate_key"]: row for row in triage_candidate_inventory(rows)}

    assert triaged["C1"]["prefilter_decision"] == "MANUAL_SCREEN_REQUIRED"
    assert "RQ2" in triaged["C1"]["rq_ids"]
    assert triaged["C2"]["prefilter_decision"] == "EXCLUDED_SECONDARY_REVIEW"
    assert triaged["C3"]["prefilter_decision"] == "EXCLUDED_NON_ML_DYNAMICS"
    assert triaged["C4"]["prefilter_decision"] == "MANUAL_SCREEN_REQUIRED"
    assert triaged["C4"]["priority_band"] == "APPLICATION_TRANSFER"
    assert triaged["C5"]["prefilter_decision"] == "EXCLUDED_NO_TRAINING_SELECTION"
    assert "weighted_score" not in triaged["C1"]


def test_manual_queue_merges_legacy_and_new_candidates_without_promoting_old_depth() -> None:
    openalex = [
        {
            "candidate_key": "OA-1",
            "title": "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics",
            "authors": "A; B",
            "year": 2020,
            "venue": "Conference",
            "primary_url": "https://primary.example/cartography",
            "doi": "10.1/cartography",
            "abstract": "Training dynamics characterize difficult examples.",
            "rq_ids": "RQ1",
            "prefilter_decision": "MANUAL_SCREEN_REQUIRED",
            "priority_band": "CORE_MECHANISM",
            "prefilter_reason": "Direct mechanism.",
        },
        {
            "candidate_key": "OA-2",
            "title": "Unrelated Excluded Work",
            "authors": "C",
            "year": 2021,
            "venue": "Journal",
            "primary_url": "https://primary.example/excluded",
            "doi": "10.1/excluded",
            "abstract": "No mechanism.",
            "rq_ids": "NOT_APPLICABLE_WITH_REASON:no mapping",
            "prefilter_decision": "EXCLUDED_NO_DIRECT_MECHANISM",
            "priority_band": "NOT_APPLICABLE",
            "prefilter_reason": "No mechanism.",
        },
    ]
    legacy = [
        {
            "evidence_id": "LEGACY-1",
            "title": "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics",
            "authors": "A; B",
            "year": 2020,
            "venue": "Conference",
            "primary_url": "https://primary.example/cartography",
            "doi": "10.1/cartography",
            "abstract": "Legacy abstract.",
            "topic": "TRAINING_DYNAMICS",
            "screening_depth": "DEEP_READ",
        },
        {
            "evidence_id": "LEGACY-2",
            "title": "A Neyman-Pearson Approach to Statistical Learning",
            "authors": "D; E",
            "year": 2005,
            "venue": "IEEE",
            "primary_url": "https://doi.org/10.1/np",
            "doi": "10.1/np",
            "abstract": "Constrained classification evidence.",
            "topic": "OPERATIONAL_TAIL",
            "screening_depth": "METHOD_READ",
        },
    ]

    rows = build_manual_screen_queue(
        openalex,
        legacy,
        legacy_note_ids={"LEGACY-1"},
        legacy_source_ids=set(),
    )

    assert len(rows) == 2
    merged = next(row for row in rows if row["legacy_evidence_ids"] == "LEGACY-1")
    assert merged["source_origins"] == "LEGACY_155;OPENALEX"
    assert merged["legacy_depth"] == "DEEP_READ"
    assert merged["legacy_note_present"] == "true"
    assert merged["legacy_source_bytes_present"] == "false"
    assert merged["queue_status"] == "PENDING_PRIMARY_SOURCE_REVIEW"
    assert "weighted_score" not in merged
    np_row = next(row for row in rows if row["legacy_evidence_ids"] == "LEGACY-2")
    assert np_row["rq_ids"] == "RQ8"
