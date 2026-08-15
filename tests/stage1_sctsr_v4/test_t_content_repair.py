from __future__ import annotations

from stage1_sctsr_v4.t_content_repair import repair_t_content_duplicates


def _t_row(rank: int, sample_id: str) -> dict[str, object]:
    return {
        "run_slot": "RUN_010",
        "triad_id": "TRIAD_004",
        "condition_id": "A02_GapCritical-Strict_B3000",
        "arm": "T",
        "training_seed": 1,
        "selection_seed": 2,
        "rank": rank,
        "sample_id": sample_id,
        "y_true": 0,
        "oof_fold": 8,
        "dynamic_bucket": "learnable_hard",
        "mean_p_defect": 0.5,
        "correct_rate": 0.6,
        "std_p_defect": 0.2,
        "replay_role": "normal_replay",
        "source_method": "GapCritical-Strict",
    }


def test_t_content_duplicate_repair_preserves_exact_quota_and_uses_best_nonterminal_candidate() -> None:
    t_rows = [_t_row(1, "base/a.png"), _t_row(2, "base/b.png")]
    candidates = [
        {
            "sample_id": "base/c.png",
            "y_true": 0,
            "oof_fold": 8,
            "dynamic_bucket": "learnable_hard",
            "oof_group_id": "bucket:1",
            "gap_critical_score": 0.7,
            "mean_p_defect": 0.4,
            "correct_rate": 0.8,
            "std_p_defect": 0.1,
        },
        {
            "sample_id": "base/d.png",
            "y_true": 0,
            "oof_fold": 8,
            "dynamic_bucket": "learnable_hard",
            "oof_group_id": "bucket:1",
            "gap_critical_score": 0.2,
            "mean_p_defect": 0.3,
            "correct_rate": 0.9,
            "std_p_defect": 0.1,
        },
    ]
    oof_groups = {sample_id: "bucket:1" for sample_id in ("base/a.png", "base/b.png")}
    content = {
        "base/a.png": {"image_sha256": "A" * 64},
        "base/b.png": {"image_sha256": "A" * 64},
        "base/c.png": {"image_sha256": "C" * 64},
        "base/d.png": {"image_sha256": "D" * 64},
    }

    repaired, audit = repair_t_content_duplicates(
        t_rows,
        candidate_rows=candidates,
        oof_groups=oof_groups,
        content_by_id=content,
    )

    assert [row["sample_id"] for row in repaired] == ["base/a.png", "base/c.png"]
    assert [row["rank"] for row in repaired] == [1, 2]
    assert audit["replacement_count"] == 1
    assert audit["replacements"][0]["removed_sample_id"] == "base/b.png"
    assert audit["replacements"][0]["replacement_sample_id"] == "base/c.png"
    assert audit["exact_four_field_quota_preserved"] is True
    assert audit["content_unique_after"] == 2
