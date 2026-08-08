from __future__ import annotations

import pandas as pd
import pytest

from stage1_gapvalue240.literature_evidence import (
    DEFAULT_COLLECTED_CAPABILITIES,
    LiteratureEvidenceError,
    assert_literature_matrix_integrity,
    build_literature_evidence_matrix,
    classify_testability,
)


def test_literature_matrix_covers_every_preregistered_mechanism_family() -> None:
    matrix = build_literature_evidence_matrix()

    required_topics = {
        "memorization",
        "forgetting",
        "dataset_cartography",
        "aum",
        "rho_loss",
        "grand_el2n",
        "influence",
        "tracin",
        "gradient_matching",
        "replay_overfit",
        "learning_rate_stability",
        "edge_of_stability",
        "parameter_drift",
        "swa_mode_connectivity",
        "early_stopping",
        "neyman_pearson",
        "partial_auc",
    }
    assert required_topics.issubset(set(matrix["topic"]))
    assert len(matrix) >= 20
    assert matrix["evidence_id"].is_unique
    assert matrix["primary_url"].str.startswith("https://").all()
    assert matrix["citation"].str.len().gt(20).all()
    assert matrix["stage1_hypothesis"].str.len().gt(20).all()
    assert matrix["claim_boundary"].str.len().gt(20).all()
    assert_literature_matrix_integrity(matrix)


def test_testability_is_computed_from_required_capabilities_not_topic_names() -> None:
    assert classify_testability(("a", "b"), {"a", "b"}) == "DIRECTLY_TESTABLE"
    assert classify_testability(("a", "b"), {"a"}) == "PARTIALLY_TESTABLE"
    assert classify_testability(("a", "b"), {"c"}) == "NOT_TESTABLE"
    assert classify_testability((), {"a"}) == "CONTEXT_ONLY"


def test_stage1_matrix_marks_gradient_and_hessian_claims_as_not_testable() -> None:
    matrix = build_literature_evidence_matrix(DEFAULT_COLLECTED_CAPABILITIES)
    by_id = matrix.set_index("evidence_id")

    for evidence_id in (
        "GRAND_MAGNITUDE",
        "INFLUENCE_ALIGNMENT",
        "TRACIN_ALIGNMENT",
        "GRAD_MATCH",
        "RHO_LOSS",
    ):
        assert by_id.loc[evidence_id, "testability_status"] == "NOT_TESTABLE"
        assert by_id.loc[evidence_id, "missing_required_capabilities"]

    assert (
        by_id.loc["FORGETTING_EVENTS", "testability_status"]
        == "DIRECTLY_TESTABLE"
    )
    assert (
        by_id.loc["OPERATIONAL_NEYMAN_PEARSON", "testability_status"]
        == "DIRECTLY_TESTABLE"
    )
    assert (
        by_id.loc["EDGE_OF_STABILITY", "testability_status"]
        == "PARTIALLY_TESTABLE"
    )


def test_integrity_gate_rejects_duplicate_or_non_primary_records() -> None:
    matrix = build_literature_evidence_matrix()
    duplicate = pd.concat([matrix, matrix.iloc[[0]]], ignore_index=True)
    with pytest.raises(LiteratureEvidenceError, match="duplicate evidence_id"):
        assert_literature_matrix_integrity(duplicate)

    broken = matrix.copy()
    broken.loc[0, "primary_url"] = "https://www.google.com/search?q=paper"
    with pytest.raises(LiteratureEvidenceError, match="primary source URL"):
        assert_literature_matrix_integrity(broken)
