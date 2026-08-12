import csv
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_candidate_universe_v2 import (
    CandidateUniverseError,
    build_candidate_universe,
    build_manual_review_groups,
    publish_candidate_universe,
)


def _openalex(record_id: str, title: str, *, decision: str) -> dict[str, str]:
    return {
        "candidate_key": record_id,
        "title": title,
        "authors": "Alex Author; Bailey Author",
        "year": "2024",
        "venue": "Primary Conference",
        "doi": "10.1000/shared" if "Shared" in title else "NOT_REPORTED_BY_SOURCE",
        "openalex_ids": "W100",
        "primary_url": "https://primary.example/openalex",
        "full_text_url": "https://primary.example/openalex.pdf",
        "abstract": "The study selects training samples under a fixed budget.",
        "query_ids": "OA001;OT001",
        "rq_ids": "RQ2;RQ5",
        "prefilter_decision": decision,
        "priority_band": "CORE_MECHANISM",
        "prefilter_reason": "Explicit preregistered mechanism.",
    }


def _legacy(record_id: str, title: str) -> dict[str, str]:
    return {
        "evidence_id": record_id,
        "title": title,
        "authors": "Alex Author; Bailey Author",
        "year": "2023",
        "venue": "Legacy Conference",
        "primary_url": "https://primary.example/legacy",
        "doi": "10.1000/shared" if "Shared" in title else "NOT_REPORTED_BY_SOURCE",
        "abstract": "Legacy evidence about training-example utility.",
        "topic": "DATA_SUBSET_SELECTION",
        "screening_depth": "DEEP_READ",
    }


def _targeted(record_id: str, title: str) -> dict[str, str]:
    return {
        "target_id": record_id,
        "title": title,
        "authors": "Alex Author; Bailey Author",
        "year": "2026",
        "venue": "Official Primary Venue",
        "primary_url": "https://primary.example/targeted",
        "full_text_url": "https://primary.example/targeted.pdf",
        "doi_or_repository_id": "10.1000/shared" if "Shared" in title else "OPENREVIEW:abc",
        "proposed_rq_ids": "RQ2;RQ7",
        "directness_hint": "D1_DIRECT_UTILITY",
        "discovery_query": "exact targeted title query",
        "discovered_at": "2026-08-09T18:00:00+08:00",
        "duplicate_check_status": "NO_EXACT_MATCH_IN_OPENALEX_V1",
        "screening_status": "PENDING_PRIMARY_SOURCE_REVIEW",
        "version_group_note": "Candidate only; identity requires manual review.",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_universe_preserves_every_version_and_is_invariant_to_input_order() -> None:
    openalex = [
        _openalex("OA-1", "Shared Study", decision="MANUAL_SCREEN_REQUIRED"),
        _openalex("OA-2", "Excluded Study", decision="EXCLUDED_NO_DIRECT_MECHANISM"),
    ]
    legacy = [_legacy("LEGACY-1", "Shared Study")]
    targeted = [_targeted("TARGET-1", "Shared Study")]

    first = build_candidate_universe(openalex, legacy, targeted)
    second = build_candidate_universe(list(reversed(openalex)), legacy, targeted)

    assert len(first) == 4
    assert first == second
    assert {row["source_origin"] for row in first} == {
        "OPENALEX_V1",
        "LEGACY_155",
        "TARGETED_PRIMARY_V1",
    }
    assert len({row["candidate_version_id"] for row in first}) == 4
    assert all("canonical_work_id" not in row for row in first)
    shared = [row for row in first if row["normalized_title"] == "shared study"]
    assert len(shared) == 3
    assert len({row["exact_title_group_hint"] for row in shared}) == 1
    assert all(row["identity_status"] == "UNRESOLVED_MANUAL" for row in shared)


def test_openalex_query_rqs_are_retained_as_hints_not_reading_evidence() -> None:
    row = _openalex("OA-1", "Shared Study", decision="MANUAL_SCREEN_REQUIRED")
    row["rq_ids"] = "RQ5"
    row["query_ids"] = "OA001;OA002"

    universe = build_candidate_universe(
        [row],
        [],
        [],
        openalex_query_rq_map={"OA001": {"RQ2", "RQ7"}, "OA002": {"RQ1"}},
    )

    assert universe[0]["title_rq_ids"] == "RQ5"
    assert universe[0]["query_rq_ids"] == "RQ1;RQ2;RQ7"
    assert universe[0]["proposed_rq_ids"] == "RQ1;RQ2;RQ5;RQ7"
    assert universe[0]["v2_reading_credit"] == "NONE_DISCOVERY_ONLY"


def test_universe_rejects_duplicate_source_identity() -> None:
    duplicate = _openalex("OA-1", "First", decision="MANUAL_SCREEN_REQUIRED")
    with pytest.raises(CandidateUniverseError, match="duplicate source record"):
        build_candidate_universe([duplicate, dict(duplicate)], [], [])


def test_manual_review_groups_keep_exclusions_out_without_losing_universe_rows() -> None:
    universe = build_candidate_universe(
        [
            _openalex("OA-1", "Shared Study", decision="MANUAL_SCREEN_REQUIRED"),
            _openalex("OA-2", "Excluded Study", decision="EXCLUDED_NO_DIRECT_MECHANISM"),
        ],
        [_legacy("LEGACY-1", "Shared Study")],
        [_targeted("TARGET-1", "Shared Study")],
    )

    groups = build_manual_review_groups(universe)

    assert len(groups) == 1
    group = groups[0]
    assert group["candidate_version_count"] == "3"
    assert len(group["candidate_version_ids"].split(";")) == 3
    assert group["identity_status"] == "UNRESOLVED_MANUAL"
    assert group["queue_status"] == "PENDING_PRIMARY_SOURCE_REVIEW"
    assert "EXCLUDED_NO_DIRECT_MECHANISM" not in group["discovery_decisions"]


def test_publish_manifest_binds_all_inputs_and_outputs(tmp_path: Path) -> None:
    openalex_path = tmp_path / "openalex.csv"
    legacy_path = tmp_path / "legacy.csv"
    targeted_path = tmp_path / "targeted.csv"
    query_plan_path = tmp_path / "query_plan.json"
    universe_path = tmp_path / "universe.csv"
    queue_path = tmp_path / "queue.csv"
    manifest_path = tmp_path / "manifest.json"
    _write_csv(
        openalex_path,
        [_openalex("OA-1", "Shared Study", decision="MANUAL_SCREEN_REQUIRED")],
    )
    _write_csv(legacy_path, [_legacy("LEGACY-1", "Shared Study")])
    _write_csv(targeted_path, [_targeted("TARGET-1", "Shared Study")])
    query_plan_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "queries": [
                    {
                        "query_id": "OA001",
                        "database": "OPENALEX",
                        "query": "sample learnability training dynamics",
                        "rq_ids": ["RQ2", "RQ7"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = publish_candidate_universe(
        openalex_path=openalex_path,
        legacy_path=legacy_path,
        targeted_path=targeted_path,
        query_plan_path=query_plan_path,
        universe_path=universe_path,
        manual_queue_path=queue_path,
        manifest_path=manifest_path,
    )

    assert manifest["status"] == "PASS"
    assert manifest["candidate_version_count"] == 3
    assert manifest["manual_review_group_count"] == 1
    assert {entry["role"] for entry in manifest["inputs"]} == {
        "OPENALEX_V1",
        "OPENALEX_QUERY_PLAN",
        "LEGACY_155",
        "TARGETED_PRIMARY_V1",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["inputs"])
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert universe_path.exists()
    assert queue_path.exists()
