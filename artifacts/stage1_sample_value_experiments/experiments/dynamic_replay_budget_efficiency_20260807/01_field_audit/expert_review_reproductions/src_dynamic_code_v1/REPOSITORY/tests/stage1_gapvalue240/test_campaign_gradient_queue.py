from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_gradient_queue import (
    GradientCandidateManifestError,
    build_gradient_candidate_manifest,
)


LOCK_SHA = "A" * 64


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    treatment = tmp_path / "TREATMENT.csv"
    pd.DataFrame(
        [
            {
                "selection_id": "TREATMENT_GAPCRITICAL_NESTED",
                "selection_rank": index,
                "role_rank": index,
                "sample_id": f"normal/n{index}.png",
                "y_true": 0,
                "replay_role": "normal_replay",
                "oof_fold": index % 2,
                "dynamic_bucket": "learnable_hard",
                "mean_p_defect": 0.9 - index / 10,
                "correct_rate": 0.25,
                "std_p_defect": 0.3,
            }
            for index in range(1, 4)
        ]
    ).to_csv(treatment, index=False)
    monitor = tmp_path / "MONITOR.csv"
    pd.DataFrame(
        [
            {
                "monitor_rank": 1,
                "monitor_group_rank": 1,
                "sample_id": "normal/n1.png",
                "y_true": 0,
                "monitor_group": "NORMAL_REPLAY_CORE_0P5_CLASS_PERCENT",
                "oof_fold": 1,
                "dynamic_bucket": "learnable_hard",
                "oof_mean_p_defect": 0.8,
                "oof_correct_rate": 0.25,
                "oof_std_p_defect": 0.3,
            },
            {
                "monitor_rank": 2,
                "monitor_group_rank": 1,
                "sample_id": "normal/n3.png",
                "y_true": 0,
                "monitor_group": "NORMAL_REPLAY_REMAINDER_STRATIFIED",
                "oof_fold": 1,
                "dynamic_bucket": "learnable_hard",
                "oof_mean_p_defect": 0.6,
                "oof_correct_rate": 0.3,
                "oof_std_p_defect": 0.2,
            },
            {
                "monitor_rank": 3,
                "monitor_group_rank": 1,
                "sample_id": "defect/d1.png",
                "y_true": 1,
                "monitor_group": "WEAK_DEFECT_CORE_0P5_CLASS_PERCENT",
                "oof_fold": 0,
                "dynamic_bucket": "learnable_hard",
                "oof_mean_p_defect": 0.1,
                "oof_correct_rate": 0.2,
                "oof_std_p_defect": 0.3,
            },
            {
                "monitor_rank": 4,
                "monitor_group_rank": 1,
                "sample_id": "defect/d2.png",
                "y_true": 1,
                "monitor_group": "WEAK_DEFECT_BUFFER_0P5_TO_5_CLASS_PERCENT_STRATIFIED",
                "oof_fold": 1,
                "dynamic_bucket": "easy_clean",
                "oof_mean_p_defect": 0.2,
                "oof_correct_rate": 0.4,
                "oof_std_p_defect": 0.2,
            },
        ]
    ).to_csv(monitor, index=False)
    return treatment, monitor


def test_gradient_candidate_manifest_uses_treatment_plus_fixed_tail_targets_without_score(
    tmp_path: Path,
) -> None:
    treatment, monitor = _sources(tmp_path)
    result = build_gradient_candidate_manifest(
        treatment,
        monitor,
        tmp_path / "gradient",
        canonical_lock_file_sha256=LOCK_SHA,
        expected_target_counts={"normal": 1, "defect": 1},
    )

    frame = pd.read_csv(result.candidate_manifest, keep_default_na=False)
    assert frame.sample_id.tolist() == [
        "normal/n1.png",
        "normal/n2.png",
        "normal/n3.png",
        "defect/d1.png",
        "defect/d2.png",
    ]
    assert frame.normal_target_member.tolist() == [True, False, False, False, False]
    assert frame.defect_target_member.tolist() == [False, False, False, True, False]
    assert frame.loc[frame.sample_id.eq("normal/n1.png"), "candidate_groups"].item() == (
        "TREATMENT_GAPCRITICAL_NESTED;NORMAL_REPLAY_CORE_0P5_CLASS_PERCENT"
    )
    assert frame.treatment_member.tolist() == [True, True, True, False, False]
    assert "gradient_value_score" not in frame.columns
    assert "composite_score" not in frame.columns

    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["union_sample_count"] == 5
    assert validation["normal_target_count"] == 1
    assert validation["defect_target_count"] == 1
    assert validation["canonical_lock_file_sha256"] == LOCK_SHA

    rerun = build_gradient_candidate_manifest(
        treatment,
        monitor,
        tmp_path / "gradient",
        canonical_lock_file_sha256=LOCK_SHA,
        expected_target_counts={"normal": 1, "defect": 1},
    )
    assert rerun.candidate_manifest == result.candidate_manifest


def test_gradient_candidate_manifest_rejects_cross_source_label_conflicts(tmp_path: Path) -> None:
    treatment, monitor = _sources(tmp_path)
    conflicting = pd.read_csv(monitor)
    conflicting.loc[conflicting.sample_id.eq("normal/n1.png"), "y_true"] = 1
    conflicting.to_csv(monitor, index=False)

    with pytest.raises(GradientCandidateManifestError, match="label conflict"):
        build_gradient_candidate_manifest(
            treatment,
            monitor,
            tmp_path / "gradient",
            canonical_lock_file_sha256=LOCK_SHA,
        )


def test_gradient_candidate_manifest_rejects_half_published_output(tmp_path: Path) -> None:
    treatment, monitor = _sources(tmp_path)
    output = tmp_path / "gradient"
    output.mkdir()
    (output / "GRADIENT_CANDIDATE_SAMPLES.csv").write_text("partial", encoding="utf-8")

    with pytest.raises(GradientCandidateManifestError, match="half-published"):
        build_gradient_candidate_manifest(
            treatment,
            monitor,
            output,
            canonical_lock_file_sha256=LOCK_SHA,
        )
