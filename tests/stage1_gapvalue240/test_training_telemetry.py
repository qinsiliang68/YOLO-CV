from __future__ import annotations

from pathlib import Path

import pandas as pd

from stage1_gapvalue240.training_telemetry import (
    build_execution_exposure_audit,
    build_paired_telemetry_deltas,
    build_run_telemetry_features,
    build_triad_outcomes,
    parse_effective_optimizers,
)


def test_triad_outcomes_keep_both_controls_and_four_goal_cohorts() -> None:
    rows = []
    specs = {
        "TRIAD_001": {"T": (110, 5), "R1": (100, 5), "R2": (105, 6)},
        "TRIAD_002": {"T": (1400, 12), "R1": (1000, 10), "R2": (1000, 10)},
        "TRIAD_003": {"T": (900, 11), "R1": (1000, 10), "R2": (1100, 9)},
        "TRIAD_004": {"T": (1100, 11), "R1": (1000, 10), "R2": (1200, 12)},
    }
    for triad, arms in specs.items():
        for arm, (tn, fn) in arms.items():
            rows.append(
                {
                    "run_slot": f"{triad}_{arm}",
                    "triad_id": triad,
                    "arm": arm,
                    "TN_at_FN95": tn,
                    "FN_at_TN68253": fn,
                    "training_seed": 11,
                    "condition_id": triad,
                }
            )

    outcomes = build_triad_outcomes(pd.DataFrame(rows)).set_index("triad_id")

    assert outcomes.loc["TRIAD_001", "dual_improvement"]
    assert outcomes.loc["TRIAD_001", "exclusive_cohort"] == "DUAL_IMPROVEMENT"
    assert outcomes.loc["TRIAD_002", "high_value"]
    assert not outcomes.loc["TRIAD_002", "dual_improvement"]
    assert outcomes.loc["TRIAD_002", "exclusive_cohort"] == "HIGH_VALUE"
    assert outcomes.loc["TRIAD_003", "dual_harm"]
    assert outcomes.loc["TRIAD_003", "exclusive_cohort"] == "DUAL_HARM"
    assert outcomes.loc["TRIAD_004", "exclusive_cohort"] == "MIXED_OR_REVERSAL"
    assert outcomes.loc["TRIAD_001", "G_TN"] == 5
    assert outcomes.loc["TRIAD_001", "G_FN"] == 0


def test_run_telemetry_uses_all_metrics_and_cutoff_local_windows() -> None:
    rows = []
    for arm, offset in (("T", 0.0), ("R1", 0.1), ("R2", -0.1)):
        for epoch in range(1, 7):
            rows.append(
                {
                    "run_slot": arm,
                    "triad_id": "TRIAD_001",
                    "arm": arm,
                    "epoch": epoch,
                    "time": float(epoch),
                    "train/loss": 1.0 - 0.1 * epoch + offset,
                    "metrics/accuracy_top1": 0.5 + 0.05 * epoch - offset,
                    "metrics/accuracy_top5": 1.0,
                    "val/loss": 0.8 - 0.05 * epoch + offset,
                    **{
                        f"lr/pg{i}": 0.01 - 0.001 * epoch
                        for i in range(8)
                    },
                }
            )
    curves = pd.DataFrame(rows)

    features = build_run_telemetry_features(
        curves,
        cutoffs=(4, 6),
        slope_window=4,
    ).set_index("run_slot")

    assert "train_loss__at_4" in features
    assert "accuracy_top5__at_6" in features
    assert "lr_pg7__slope_3_to_6" in features
    assert "val_loss__curvature_3_to_6" in features
    assert features.loc["T", "train_loss__at_4"] == 0.6
    assert abs(features.loc["T", "train_loss__slope_3_to_6"] + 0.1) < 1e-12
    assert features.loc["T", "accuracy_top5__unique_to_6"] == 1
    assert features.loc["T", "time_reset_count"] == 0


def test_effective_optimizer_parser_prefers_real_log_over_auto_config(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    log = attempt / "02_logs/train_20260712T115558_c05653f5.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and "
        "'momentum=0.937'\n"
        "optimizer: MuSGD(lr=0.01, momentum=0.9) with parameter groups "
        "82 weight(decay=0.0), 83 weight(decay=0.001), "
        "83 bias(decay=0.0)\n",
        encoding="utf-8",
    )
    files = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "attempt_dir": str(attempt),
                "relative_path": "02_logs/train_20260712T115558_c05653f5.log",
                "normalized_path": "02_logs/train_{TIMESTAMP}_{TOKEN}.log",
            }
        ]
    )

    parsed = parse_effective_optimizers(files).iloc[0]

    assert parsed["effective_optimizer"] == "MuSGD"
    assert parsed["effective_lr"] == 0.01
    assert parsed["effective_momentum"] == 0.9
    assert parsed["configured_lr_ignored"]
    assert parsed["parameter_group_count"] == 3
    assert parsed["parameter_group_weight_decay"] == "0,0.001,0"


def test_execution_exposure_audit_integrates_steps_lr_and_replay(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    manifest = attempt / "01_manifests/manifest_summary.json"
    audit = attempt / "02_logs/training_execution_audit.json"
    manifest.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    manifest.write_text(
        """{
          "base_normal_rows": 8,
          "base_train_rows": 10,
          "epoch_samples": 20,
          "replay_defect_rows": 0,
          "replay_normal_rows": 2
        }""",
        encoding="utf-8",
    )
    audit.write_text(
        """{
          "completed_epochs": 3,
          "effective_batch_size": 4,
          "expected_steps_per_epoch": 5,
          "optimizer_steps_total": 15,
          "resume_count": 0,
          "resume_mode": "native_approximate",
          "resume_segments": [{"start_epoch": 1, "end_epoch": 3}]
        }""",
        encoding="utf-8",
    )
    files = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "attempt_dir": str(attempt),
                "relative_path": "01_manifests/manifest_summary.json",
                "normalized_path": "01_manifests/manifest_summary.json",
            },
            {
                "run_slot": "RUN_001",
                "attempt_dir": str(attempt),
                "relative_path": "02_logs/training_execution_audit.json",
                "normalized_path": "02_logs/training_execution_audit.json",
            },
        ]
    )
    curves = pd.DataFrame(
        {
            "run_slot": ["RUN_001"] * 3,
            "epoch": [1, 2, 3],
            **{f"lr/pg{i}": [0.1, 0.05, 0.01] for i in range(8)},
        }
    )

    result = build_execution_exposure_audit(files, curves, cutoffs=(2, 3)).iloc[0]

    assert result["base_total_rows"] == 18
    assert result["replay_total_rows"] == 2
    assert result["total_replay_exposures"] == 6
    assert result["optimizer_steps_to_2"] == 10
    assert result["replay_exposures_to_2"] == 4
    assert abs(result["lr_step_integral_pg0_to_2"] - 0.75) < 1e-12
    assert abs(result["lr_replay_integral_pg0_to_2"] - 0.3) < 1e-12


def test_paired_telemetry_deltas_preserve_r1_and_r2() -> None:
    features = pd.DataFrame(
        [
            {
                "run_slot": "T",
                "triad_id": "TRIAD_001",
                "arm": "T",
                "training_seed": 11,
                "train_loss__at_150": 0.2,
                "val_loss__slope_131_to_150": 0.01,
            },
            {
                "run_slot": "R1",
                "triad_id": "TRIAD_001",
                "arm": "R1",
                "training_seed": 11,
                "train_loss__at_150": 0.3,
                "val_loss__slope_131_to_150": -0.01,
            },
            {
                "run_slot": "R2",
                "triad_id": "TRIAD_001",
                "arm": "R2",
                "training_seed": 11,
                "train_loss__at_150": 0.4,
                "val_loss__slope_131_to_150": 0.0,
            },
        ]
    )

    paired = build_paired_telemetry_deltas(features).set_index("control")

    assert set(paired.index) == {"R1", "R2"}
    assert abs(paired.loc["R1", "delta__train_loss__at_150"] + 0.1) < 1e-12
    assert abs(paired.loc["R2", "delta__train_loss__at_150"] + 0.2) < 1e-12
    assert abs(
        paired.loc["R1", "delta__val_loss__slope_131_to_150"] - 0.02
    ) < 1e-12
