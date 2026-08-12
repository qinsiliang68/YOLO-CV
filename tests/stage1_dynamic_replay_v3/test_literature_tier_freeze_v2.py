from __future__ import annotations

from dataclasses import replace
import random

import pytest

from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import (
    BroadCandidate,
    ExplicitMerge,
    TierSelectionError,
    TierSelectionPolicy,
    canonicalize_candidates,
    select_broad_candidates,
)


def _candidate(
    index: int,
    *rqs: str,
    relevance_class: str = "DIRECT_INTERVENTION",
    title: str | None = None,
    authors: str | None = None,
    year: int = 2024,
    doi: str = "",
    effect_relation: str = "SUPPORTED",
) -> BroadCandidate:
    return BroadCandidate(
        queue_id=f"RG{index:04d}",
        canonical_work_id=f"CW{index:04d}",
        title=title or f"Distinct training-sample study {index}",
        authors=authors or f"Author {index}",
        year=year,
        direct_rqs=tuple(rqs),
        relevance_class=relevance_class,
        citation_count=index * 100,
        doi=doi,
        effect_relation=effect_relation,
    )


def test_canonicalization_requires_explicit_evidence_for_duplicate_versions() -> None:
    first = _candidate(
        1,
        "RQ2",
        "RQ5",
        title="Extracting In-domain Training Corpora for Neural Machine Translation Using Data Selection Methods",
        authors="Catarina Cruz Silva; Chao-Hong Liu; Alberto Poncelas; Andy Way",
        year=2018,
    )
    second = replace(
        first,
        queue_id="RG9999",
        canonical_work_id="CW9999",
        title="Extracting in-domain training corpora for neural machine translationusing data selection methods",
    )

    with pytest.raises(TierSelectionError, match="explicit merge"):
        canonicalize_candidates([first, second], merges=[])

    result = canonicalize_candidates(
        [first, second],
        merges=[
            ExplicitMerge(
                alias_queue_id="RG9999",
                canonical_queue_id="RG0001",
                evidence="same authors, year, method, experiments, and official DOI version",
            )
        ],
    )

    assert len(result) == 1
    assert result[0].queue_id == "RG0001"
    assert result[0].merged_queue_ids == ("RG9999",)
    assert set(result[0].direct_rqs) == {"RQ2", "RQ5"}


def test_canonicalization_rejects_unmerged_duplicate_doi() -> None:
    first = _candidate(1, "RQ1", doi="10.1234/same-work")
    second = _candidate(2, "RQ2", doi="https://doi.org/10.1234/SAME-WORK")

    with pytest.raises(TierSelectionError, match="DOI"):
        canonicalize_candidates([first, second], merges=[])


def test_canonicalization_allows_explicit_version_identity_with_shared_key() -> None:
    preprint = _candidate(
        1,
        "RQ2",
        title="Estimating Training Data Influence by Tracking Gradient Descent",
        authors="Garima Pruthi; Frederick Liu; Mukund Sundararajan; Satyen Kale",
        year=2020,
    )
    proceedings = _candidate(
        2,
        "RQ3",
        title="Estimating Training Data Influence by Tracing Gradient Descent",
        authors="Garima Pruthi; Frederick Liu; Satyen Kale; Mukund Sundararajan",
        year=2021,
    )

    result = canonicalize_candidates(
        [preprint, proceedings],
        merges=[
            ExplicitMerge(
                alias_queue_id="RG0001",
                canonical_queue_id="RG0002",
                evidence="Same authors, method, experiments, arXiv preprint, and proceedings work.",
                merge_basis="VERSION_IDENTITY",
                shared_identity="ARXIV:2002.08484",
            )
        ],
    )

    assert len(result) == 1
    assert result[0].queue_id == "RG0002"
    assert result[0].merged_queue_ids == ("RG0001",)
    assert set(result[0].direct_rqs) == {"RQ2", "RQ3"}


def test_version_identity_rejects_different_author_sets() -> None:
    first = _candidate(1, "RQ1", title="Preprint title", authors="A One; B Two")
    second = _candidate(2, "RQ2", title="Journal title", authors="A One; C Three")

    with pytest.raises(TierSelectionError, match="author mismatch"):
        canonicalize_candidates(
            [first, second],
            merges=[
                ExplicitMerge(
                    alias_queue_id="RG0001",
                    canonical_queue_id="RG0002",
                    evidence="Claimed version relationship with an explicit identity key.",
                    merge_basis="VERSION_IDENTITY",
                    shared_identity="DOI:10.1234/example",
                )
            ],
        )


def test_broad_selection_is_exact_balanced_and_order_invariant() -> None:
    candidates = [
        _candidate(index, f"RQ{index}") for index in range(1, 9)
    ] + [
        _candidate(9, "RQ1", "RQ2"),
        _candidate(10, "RQ3", "RQ4"),
        _candidate(11, "RQ5", relevance_class="TRANSFER_COMPONENT"),
        _candidate(12, "RQ6", relevance_class="TRANSFER_COMPONENT"),
    ]
    policy = TierSelectionPolicy(
        total=10,
        minimum_per_rq=1,
        maximum_per_rq=3,
        maximum_transfer=2,
        frozen_seed="tier-freeze-test",
    )

    first = select_broad_candidates(candidates, policy)
    shuffled = list(candidates)
    random.Random(20260810).shuffle(shuffled)
    second = select_broad_candidates(shuffled, policy)

    assert len(first.selected) == 10
    assert sum(first.quota_counts.values()) == 10
    assert all(count >= 1 for count in first.quota_counts.values())
    assert all(count <= 3 for count in first.quota_counts.values())
    assert sum(item.directness == "D4_TRANSFER_ANALOG" for item in first.selected) <= 2
    assert [(item.candidate.queue_id, item.quota_rq) for item in first.selected] == [
        (item.candidate.queue_id, item.quota_rq) for item in second.selected
    ]


def test_forbidden_bibliometrics_do_not_change_selection() -> None:
    candidates = [
        _candidate(index, f"RQ{((index - 1) % 8) + 1}")
        for index in range(1, 17)
    ]
    policy = TierSelectionPolicy(
        total=12,
        minimum_per_rq=1,
        maximum_per_rq=3,
        maximum_transfer=4,
        frozen_seed="forbidden-field-invariance",
    )
    baseline = select_broad_candidates(candidates, policy)
    mutated = [
        replace(item, year=1900 + index, citation_count=10_000_000 - index)
        for index, item in enumerate(reversed(candidates), start=1)
    ]
    changed = select_broad_candidates(mutated, policy)

    assert [item.candidate.queue_id for item in baseline.selected] == [
        item.candidate.queue_id for item in changed.selected
    ]


def test_selection_fails_closed_when_an_rq_minimum_is_impossible() -> None:
    candidates = [_candidate(index, "RQ1") for index in range(1, 9)]
    policy = TierSelectionPolicy(
        total=8,
        minimum_per_rq=1,
        maximum_per_rq=4,
        maximum_transfer=8,
        frozen_seed="infeasible",
    )

    with pytest.raises(TierSelectionError, match="RQ2"):
        select_broad_candidates(candidates, policy)


def test_screened_queue_preserves_counterevidence_per_rq() -> None:
    candidates: list[BroadCandidate] = []
    index = 1
    for rq_index in range(1, 9):
        rq = f"RQ{rq_index}"
        candidates.append(_candidate(index, rq, effect_relation="NULL_NEGATIVE"))
        index += 1
        candidates.append(_candidate(index, rq, effect_relation="MIXED"))
        index += 1
        candidates.extend(_candidate(index + offset, rq) for offset in range(3))
        index += 3
    policy = TierSelectionPolicy(
        total=32,
        minimum_per_rq=3,
        maximum_per_rq=5,
        maximum_transfer=8,
        minimum_counterevidence_per_rq=2,
        tier_label="SCREENED",
        frozen_seed="screened-counterevidence",
    )

    result = select_broad_candidates(candidates, policy)

    for rq in (f"RQ{index}" for index in range(1, 9)):
        counter_count = sum(
            item.quota_rq == rq
            and item.candidate.effect_relation in {"NULL_NEGATIVE", "MIXED"}
            for item in result.selected
        )
        assert counter_count >= 2


def test_mandatory_anchor_is_selected_before_lexicographic_fill() -> None:
    candidates = [
        _candidate(index, f"RQ{((index - 1) % 8) + 1}")
        for index in range(1, 25)
    ]
    baseline_policy = TierSelectionPolicy(
        total=8,
        minimum_per_rq=1,
        maximum_per_rq=2,
        maximum_transfer=2,
        frozen_seed="mandatory-anchor-baseline",
    )
    baseline = select_broad_candidates(candidates, baseline_policy)
    baseline_ids = {item.candidate.canonical_work_id for item in baseline.selected}
    anchor_id = next(
        candidate.canonical_work_id
        for candidate in reversed(candidates)
        if candidate.canonical_work_id not in baseline_ids
    )

    result = select_broad_candidates(
        candidates,
        replace(
            baseline_policy,
            mandatory_canonical_work_ids=(anchor_id,),
        ),
    )

    selected = {item.candidate.canonical_work_id: item for item in result.selected}
    assert anchor_id in selected
    assert selected[anchor_id].selection_phase == "MANDATORY_ANCHOR"
    assert len(result.selected) == 8


def test_mandatory_anchor_missing_from_eligible_pool_fails_closed() -> None:
    candidates = [
        _candidate(index, f"RQ{((index - 1) % 8) + 1}")
        for index in range(1, 17)
    ]
    policy = TierSelectionPolicy(
        total=8,
        minimum_per_rq=1,
        maximum_per_rq=2,
        maximum_transfer=2,
        mandatory_canonical_work_ids=("CW_NOT_ELIGIBLE",),
        frozen_seed="missing-mandatory-anchor",
    )

    with pytest.raises(TierSelectionError, match="mandatory anchor.*CW_NOT_ELIGIBLE"):
        select_broad_candidates(candidates, policy)


def test_mandatory_anchor_cannot_override_transfer_cap() -> None:
    candidates = [
        _candidate(index, f"RQ{((index - 1) % 8) + 1}")
        for index in range(1, 17)
    ]
    candidates.extend(
        [
            _candidate(101, "RQ1", relevance_class="TRANSFER_COMPONENT"),
            _candidate(102, "RQ2", relevance_class="TRANSFER_COMPONENT"),
        ]
    )
    policy = TierSelectionPolicy(
        total=8,
        minimum_per_rq=1,
        maximum_per_rq=2,
        maximum_transfer=1,
        mandatory_canonical_work_ids=("CW0101", "CW0102"),
        frozen_seed="mandatory-transfer-cap",
    )

    with pytest.raises(TierSelectionError, match="mandatory anchor.*transfer cap"):
        select_broad_candidates(candidates, policy)
