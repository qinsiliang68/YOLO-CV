from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1_sample_value_experiment_layout_20260708 import (  # noqa: E402
    BUDGETS,
    METHOD_IDS,
    STAGE_DIRS,
    default_experiment_rows,
    experiment_root,
    initialize_experiment_family_layout,
    method_budget_dir,
)
from scripts.export_stage1_oof_checkpoint_dynamics_20260708 import (  # noqa: E402
    add_epoch_rank_columns,
    append_resource_log,
    checkpoint_id,
    completed_epoch_is_valid,
    job_dynamics_root,
    parse_epoch_spec,
    read_checkpoint_map,
    read_fold_run_map,
    resolve_checkpoint_path,
    sample_id_for_row,
    validate_export_request,
    write_epoch_predictions_atomic,
)
from scripts.summarize_stage1_oof_checkpoint_dynamics_20260708 import (  # noqa: E402
    bucket_for_summary,
    collect_prediction_files,
    validate_summary_inputs,
    summarize_trajectory,
)
from scripts.build_stage1_sample_value_replay_manifests_20260708 import (  # noqa: E402
    build_replay_manifest_for_run,
    method_score_column,
    select_value_rows,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_experiment_family_layout_registers_legacy_runs_as_parallel_experiments(tmp_path: Path) -> None:
    family_root = tmp_path / "stage1_sample_value_experiments"

    initialize_experiment_family_layout(family_root)

    registry_rows = read_csv(family_root / "00_registry" / "experiments.csv")
    experiment_ids = [row["experiment_id"] for row in registry_rows]
    assert experiment_ids == [row["experiment_id"] for row in default_experiment_rows()]
    assert "hn_rn_40runs_20260630" in experiment_ids
    assert "hn_band_120runs_20260707" in experiment_ids
    assert "oof_dynamics_gap_value_20260708" in experiment_ids

    new_root = experiment_root(family_root, "oof_dynamics_gap_value_20260708")
    for stage_dir in STAGE_DIRS.values():
        assert (new_root / stage_dir).is_dir()

    for method_id in METHOD_IDS:
        for budget in BUDGETS:
            assert method_budget_dir(new_root, "03_replay_manifests", method_id, budget).is_dir()
            assert method_budget_dir(new_root, "04_training_runs", method_id, budget).is_dir()
            assert method_budget_dir(new_root, "05_eval", method_id, budget).is_dir()

    methods = read_csv(new_root / "00_registry" / "methods.csv")
    by_method = {row["method_id"]: row for row in methods}
    assert by_method["gap_critical"]["manifest_build_status"] == "default_ready"
    assert by_method["grad_align_guard"]["manifest_build_status"] == "requires_defect_guard_budget"

    legacy_root = experiment_root(family_root, "hn_band_120runs_20260707")
    pointer = legacy_root / "README.md"
    assert pointer.exists()
    assert "legacy_artifact_root" in pointer.read_text(encoding="utf-8")


def test_job_dynamics_root_keeps_parallel_jobs_from_overwriting(tmp_path: Path) -> None:
    dynamics_root = tmp_path / "01_oof_dynamics"

    assert job_dynamics_root(dynamics_root, "node01_folds_00_03") == dynamics_root / "jobs" / "node01_folds_00_03"
    assert job_dynamics_root(dynamics_root, "node02_folds_04_06") == dynamics_root / "jobs" / "node02_folds_04_06"


def test_validate_export_request_requires_job_id_and_bound_run_for_formal_export(tmp_path: Path) -> None:
    try:
        validate_export_request(dry_run=False, job_id="", folds=[0], fold_run_map={}, allow_auto_discover=False)
    except ValueError as exc:
        assert "--job-id" in str(exc)
    else:
        raise AssertionError("formal export without --job-id should fail")

    try:
        validate_export_request(dry_run=False, job_id="node01", folds=[0], fold_run_map={}, allow_auto_discover=False)
    except ValueError as exc:
        assert "--fold-run-map" in str(exc)
    else:
        raise AssertionError("formal export without fold-run-map should fail")

    validate_export_request(dry_run=True, job_id="", folds=[0], fold_run_map={}, allow_auto_discover=False)
    validate_export_request(dry_run=False, job_id="node01", folds=[0], fold_run_map={}, allow_auto_discover=True)


def test_parse_epoch_spec_supports_ranges_and_keeps_order() -> None:
    assert parse_epoch_spec("1-3,7,3,10") == [1, 2, 3, 7, 10]


def test_parse_epoch_spec_rejects_invalid_values() -> None:
    for raw in ["", "0", "3-1", "a"]:
        try:
            parse_epoch_spec(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid epoch spec should fail: {raw}")


def test_resolve_checkpoint_path_prefers_zero_based_epoch_names_when_available(tmp_path: Path) -> None:
    fold_root = tmp_path / "fold_00" / "train_run" / "weights"
    fold_root.mkdir(parents=True)
    one_based = fold_root / "epoch1.pt"
    zero_based = fold_root / "epoch0.pt"
    one_based.write_text("one", encoding="utf-8")
    zero_based.write_text("zero", encoding="utf-8")

    resolved = resolve_checkpoint_path(tmp_path, fold=0, epoch=1)

    assert resolved == zero_based


def test_resolve_checkpoint_path_uses_one_based_epoch_names_without_epoch0(tmp_path: Path) -> None:
    fold_root = tmp_path / "fold_00" / "train_run" / "weights"
    fold_root.mkdir(parents=True)
    one_based = fold_root / "epoch1.pt"
    one_based.write_text("one", encoding="utf-8")

    resolved = resolve_checkpoint_path(tmp_path, fold=0, epoch=1)

    assert resolved == one_based


def test_resolve_checkpoint_path_finds_nested_extracted_archive_root(tmp_path: Path) -> None:
    weights_root = tmp_path / "ssh" / "AI" / "runs" / "YOLOv11" / "stage1_oof_10fold" / "fold_09" / "run" / "weights"
    weights_root.mkdir(parents=True)
    expected = weights_root / "epoch0.pt"
    expected.write_text("zero", encoding="utf-8")

    resolved = resolve_checkpoint_path(tmp_path, fold=9, epoch=1)

    assert resolved == expected


def test_resolve_checkpoint_path_uses_explicit_map(tmp_path: Path) -> None:
    mapped = tmp_path / "custom" / "fold0_epoch1.pt"
    mapped.parent.mkdir(parents=True)
    mapped.write_text("mapped", encoding="utf-8")
    checkpoint_map = {(0, 1): mapped}

    assert resolve_checkpoint_path(tmp_path / "runs", fold=0, epoch=1, checkpoint_map=checkpoint_map) == mapped


def test_checkpoint_and_fold_run_maps_reject_duplicate_keys(tmp_path: Path) -> None:
    checkpoint_map_path = tmp_path / "checkpoint_map.csv"
    write_csv(
        checkpoint_map_path,
        [
            {"oof_fold": "0", "epoch": "1", "checkpoint_path": "a.pt"},
            {"oof_fold": "0", "epoch": "1", "checkpoint_path": "b.pt"},
        ],
        ["oof_fold", "epoch", "checkpoint_path"],
    )
    try:
        read_checkpoint_map(checkpoint_map_path)
    except ValueError as exc:
        assert "Duplicate checkpoint map key" in str(exc)
    else:
        raise AssertionError("duplicate checkpoint map keys should fail")

    fold_run_map_path = tmp_path / "fold_run_map.csv"
    write_csv(
        fold_run_map_path,
        [
            {"oof_fold": "0", "run_dir": "run_a"},
            {"oof_fold": "0", "run_dir": "run_b"},
        ],
        ["oof_fold", "run_dir"],
    )
    try:
        read_fold_run_map(fold_run_map_path)
    except ValueError as exc:
        assert "Duplicate fold-run map key" in str(exc)
    else:
        raise AssertionError("duplicate fold-run map keys should fail")


def test_read_fold_run_map_binds_fold_to_exact_run_dir(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    (run_a / "weights").mkdir(parents=True)
    (run_b / "weights").mkdir(parents=True)
    (run_a / "weights" / "epoch0.pt").write_text("a", encoding="utf-8")
    expected = run_b / "weights" / "epoch0.pt"
    expected.write_text("b", encoding="utf-8")
    map_path = tmp_path / "fold_run_map.csv"
    write_csv(map_path, [{"oof_fold": "0", "run_dir": str(run_b)}], ["oof_fold", "run_dir"])

    fold_run_map = read_fold_run_map(map_path)

    assert fold_run_map[0] == run_b
    assert resolve_checkpoint_path(tmp_path, fold=0, epoch=1, fold_run_map=fold_run_map, allow_auto_discover=False) == expected


def test_checkpoint_id_and_sample_id_are_stable() -> None:
    assert checkpoint_id(3, 17) == "fold_03_epoch_017"
    row = {"canonical_image_relpath": "normal_train\\0001.png", "Filename": "ignored.png", "y_true": "0"}
    assert sample_id_for_row(row) == "normal_train/0001.png"


def test_add_epoch_rank_columns_ranks_high_defect_scores_within_each_label() -> None:
    rows = [
        {"y_true": "0", "p_defect_raw": "0.90", "Filename": "n1.png"},
        {"y_true": "0", "p_defect_raw": "0.20", "Filename": "n2.png"},
        {"y_true": "0", "p_defect_raw": "0.90", "Filename": "n3.png"},
        {"y_true": "1", "p_defect_raw": "0.10", "Filename": "d1.png"},
        {"y_true": "1", "p_defect_raw": "0.80", "Filename": "d2.png"},
    ]

    ranked = add_epoch_rank_columns(rows)

    by_name = {row["Filename"]: row for row in ranked}
    assert by_name["n1.png"]["risk_rank_desc_within_label"] == "1"
    assert by_name["n3.png"]["risk_rank_desc_within_label"] == "2"
    assert by_name["n2.png"]["risk_rank_desc_within_label"] == "3"
    assert by_name["n1.png"]["risk_rank_pct_desc_within_label"] == "0.3333333333"
    assert by_name["d2.png"]["risk_rank_desc_within_label"] == "1"
    assert by_name["d1.png"]["risk_rank_desc_within_label"] == "2"


def test_atomic_epoch_write_requires_unique_samples_and_sidecar(tmp_path: Path) -> None:
    output_path = tmp_path / "fold_00" / "epoch_001_predictions.csv"
    rows = [
        {"sample_id": "a.png", "y_true": "0", "p_defect_raw": "0.1"},
        {"sample_id": "b.png", "y_true": "1", "p_defect_raw": "0.9"},
    ]

    sidecar = write_epoch_predictions_atomic(
        output_path,
        rows,
        ("sample_id", "y_true", "p_defect_raw"),
        expected_row_count=2,
        manifest_payload={"checkpoint_id": "fold_00_epoch_001"},
    )

    assert output_path.exists()
    assert sidecar.exists()
    assert completed_epoch_is_valid(output_path, expected_row_count=2)


def test_atomic_epoch_write_removes_tmp_and_rejects_duplicate_samples(tmp_path: Path) -> None:
    output_path = tmp_path / "fold_00" / "epoch_001_predictions.csv"
    rows = [
        {"sample_id": "dup.png", "y_true": "0", "p_defect_raw": "0.1"},
        {"sample_id": "dup.png", "y_true": "1", "p_defect_raw": "0.9"},
    ]

    try:
        write_epoch_predictions_atomic(
            output_path,
            rows,
            ("sample_id", "y_true", "p_defect_raw"),
            expected_row_count=2,
            manifest_payload={"checkpoint_id": "fold_00_epoch_001"},
        )
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("duplicate sample ids should fail")

    assert not output_path.exists()
    assert not (tmp_path / "fold_00" / "epoch_001_predictions.csv.tmp").exists()


def test_completed_epoch_is_invalid_without_sidecar_or_wrong_row_count(tmp_path: Path) -> None:
    output_path = tmp_path / "fold_00" / "epoch_001_predictions.csv"
    write_csv(output_path, [{"sample_id": "a.png"}], ["sample_id"])

    assert not completed_epoch_is_valid(output_path, expected_row_count=1)

    write_epoch_predictions_atomic(
        output_path,
        [{"sample_id": "a.png"}],
        ("sample_id",),
        expected_row_count=1,
        manifest_payload={"checkpoint_id": "fold_00_epoch_001"},
    )

    assert not completed_epoch_is_valid(output_path, expected_row_count=2)


def test_completed_epoch_requires_current_checkpoint_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "fold_00" / "epoch_001_predictions.csv"
    old_checkpoint = tmp_path / "old" / "epoch0.pt"
    new_checkpoint = tmp_path / "new" / "epoch0.pt"
    old_checkpoint.parent.mkdir(parents=True)
    new_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_text("old", encoding="utf-8")
    new_checkpoint.write_text("new", encoding="utf-8")
    write_epoch_predictions_atomic(
        output_path,
        [
            {
                "sample_id": "a.png",
                "checkpoint_id": "fold_00_epoch_001",
                "oof_fold": "00",
                "epoch": "1",
                "checkpoint_path": str(old_checkpoint),
            }
        ],
        ("sample_id", "checkpoint_id", "oof_fold", "epoch", "checkpoint_path"),
        expected_row_count=1,
        manifest_payload={
            "checkpoint_id": "fold_00_epoch_001",
            "oof_fold": "00",
            "epoch": 1,
            "checkpoint_path": str(old_checkpoint),
        },
    )

    assert completed_epoch_is_valid(
        output_path,
        expected_row_count=1,
        expected_checkpoint_id="fold_00_epoch_001",
        expected_oof_fold="00",
        expected_epoch=1,
        expected_checkpoint_path=old_checkpoint,
    )
    assert not completed_epoch_is_valid(
        output_path,
        expected_row_count=1,
        expected_checkpoint_id="fold_00_epoch_001",
        expected_oof_fold="00",
        expected_epoch=1,
        expected_checkpoint_path=new_checkpoint,
    )


def test_resource_log_keeps_before_and_after_cleanup_columns(tmp_path: Path) -> None:
    log_path = tmp_path / "resource_log.csv"

    append_resource_log(
        log_path,
        {
            "created_at": "now",
            "checkpoint_id": "fold_00_epoch_001",
            "oof_fold": "00",
            "epoch": "1",
            "status": "complete",
            "duration_sec": "1.0",
            "rss_mb_before_cleanup": "100.0",
            "rss_mb_after_cleanup": "50.0",
            "cuda_allocated_mb_before_cleanup": "200.0",
            "cuda_allocated_mb_after_cleanup": "0.0",
        },
    )

    header = log_path.read_text(encoding="utf-8").splitlines()[0]
    assert "rss_mb_before_cleanup" in header
    assert "rss_mb_after_cleanup" in header
    assert "cuda_allocated_mb_before_cleanup" in header
    assert "cuda_allocated_mb_after_cleanup" in header


def test_summarize_trajectory_tracks_forgetting_and_auc() -> None:
    rows = [
        {"epoch": "1", "y_true": "1", "p_defect_raw": "0.20", "raw_correct": "0", "raw_cross_entropy": "1.6094"},
        {"epoch": "2", "y_true": "1", "p_defect_raw": "0.70", "raw_correct": "1", "raw_cross_entropy": "0.3567"},
        {"epoch": "3", "y_true": "1", "p_defect_raw": "0.40", "raw_correct": "0", "raw_cross_entropy": "0.9163"},
        {"epoch": "4", "y_true": "1", "p_defect_raw": "0.90", "raw_correct": "1", "raw_cross_entropy": "0.1054"},
    ]

    summary = summarize_trajectory("defect/0001.png", rows)

    assert summary["sample_id"] == "defect/0001.png"
    assert summary["epoch_count"] == "4"
    assert summary["first_learned_epoch"] == "2"
    assert summary["last_wrong_epoch"] == "3"
    assert summary["forgetting_count"] == "1"
    assert summary["final_correct"] == "1"
    assert summary["mean_p_defect"] == "0.5500000000"
    assert summary["p_defect_trend"] == "0.7000000000"
    assert float(summary["loss_auc_mean"]) > 0.0


def test_summarize_trajectory_rejects_duplicate_sample_epoch() -> None:
    rows = [
        {"epoch": "1", "y_true": "0", "p_defect_raw": "0.10", "raw_correct": "1"},
        {"epoch": "1", "y_true": "0", "p_defect_raw": "0.20", "raw_correct": "1"},
    ]

    try:
        summarize_trajectory("normal_train/a.png", rows)
    except ValueError as exc:
        assert "duplicate epoch" in str(exc)
    else:
        raise AssertionError("duplicate epochs for one sample should fail")


def test_bucket_for_summary_separates_boundary_persistent_and_learnable_cases() -> None:
    assert bucket_for_summary({"correct_rate": "1.0", "mean_loss": "0.05", "forgetting_count": "0"}) == "easy_clean"
    assert (
        bucket_for_summary(
            {
                "correct_rate": "0.75",
                "first_learned_epoch": "2",
                "last_wrong_epoch": "3",
                "final_correct": "1",
                "forgetting_count": "0",
                "mean_abs_margin": "0.4",
            }
        )
        == "learnable_hard"
    )
    assert (
        bucket_for_summary(
            {
                "correct_rate": "0.55",
                "first_learned_epoch": "1",
                "last_wrong_epoch": "4",
                "final_correct": "1",
                "forgetting_count": "2",
                "mean_abs_margin": "0.05",
            }
        )
        == "unstable_boundary"
    )
    assert (
        bucket_for_summary(
            {
                "correct_rate": "0.05",
                "final_correct": "0",
                "forgetting_count": "0",
                "mean_loss": "2.5",
            }
        )
        == "possible_noise_or_label_issue"
    )


def test_summary_collects_job_outputs_and_rejects_duplicate_fold_epoch(tmp_path: Path) -> None:
    dynamics_root = tmp_path / "01_oof_dynamics"
    for job_id in ("node01", "node02"):
        output_path = dynamics_root / "jobs" / job_id / "fold_00" / "epoch_001_predictions.csv"
        write_epoch_predictions_atomic(
            output_path,
            [{"sample_id": f"{job_id}.png", "oof_fold": "00", "epoch": "1", "y_true": "0", "p_defect_raw": "0.1"}],
            ("sample_id", "oof_fold", "epoch", "y_true", "p_defect_raw"),
            expected_row_count=1,
            manifest_payload={"checkpoint_id": "fold_00_epoch_001"},
        )

    files = collect_prediction_files(dynamics_root)
    try:
        validate_summary_inputs(files, expected_folds=[0], expected_epochs=[1], output_root=dynamics_root / "summary")
    except ValueError as exc:
        assert "Duplicate fold/epoch" in str(exc)
    else:
        raise AssertionError("duplicate fold/epoch across jobs should fail")


def test_summary_validation_rejects_missing_expected_epoch(tmp_path: Path) -> None:
    dynamics_root = tmp_path / "01_oof_dynamics"
    output_path = dynamics_root / "jobs" / "node01" / "fold_00" / "epoch_001_predictions.csv"
    write_epoch_predictions_atomic(
        output_path,
        [{"sample_id": "a.png", "oof_fold": "00", "epoch": "1", "y_true": "0", "p_defect_raw": "0.1"}],
        ("sample_id", "oof_fold", "epoch", "y_true", "p_defect_raw"),
        expected_row_count=1,
        manifest_payload={"checkpoint_id": "fold_00_epoch_001"},
    )

    try:
        validate_summary_inputs(
            collect_prediction_files(dynamics_root),
            expected_folds=[0],
            expected_epochs=[1, 2],
            output_root=dynamics_root / "summary",
        )
    except ValueError as exc:
        assert "Missing expected fold/epoch" in str(exc)
    else:
        raise AssertionError("missing expected epoch should fail")


def test_summary_validation_rejects_prediction_csv_without_complete_sidecar(tmp_path: Path) -> None:
    dynamics_root = tmp_path / "01_oof_dynamics"
    output_path = dynamics_root / "jobs" / "node01" / "fold_00" / "epoch_001_predictions.csv"
    write_csv(
        output_path,
        [{"sample_id": "a.png", "oof_fold": "00", "epoch": "1", "y_true": "0", "p_defect_raw": "0.1"}],
        ["sample_id", "oof_fold", "epoch", "y_true", "p_defect_raw"],
    )

    try:
        validate_summary_inputs(
            collect_prediction_files(dynamics_root),
            expected_folds=[0],
            expected_epochs=[1],
            output_root=dynamics_root / "summary",
        )
    except ValueError as exc:
        assert "sidecar" in str(exc)
    else:
        raise AssertionError("prediction CSV without sidecar should fail")


def test_summary_validation_rejects_csv_epoch_that_disagrees_with_path(tmp_path: Path) -> None:
    dynamics_root = tmp_path / "01_oof_dynamics"
    good_path = dynamics_root / "jobs" / "node01" / "fold_00" / "epoch_001_predictions.csv"
    bad_path = dynamics_root / "jobs" / "node01" / "fold_00" / "epoch_002_predictions.csv"
    write_epoch_predictions_atomic(
        good_path,
        [{"sample_id": "a.png", "oof_fold": "00", "epoch": "1", "y_true": "0", "p_defect_raw": "0.1"}],
        ("sample_id", "oof_fold", "epoch", "y_true", "p_defect_raw"),
        expected_row_count=1,
        manifest_payload={"checkpoint_id": "fold_00_epoch_001", "oof_fold": "00", "epoch": 1},
    )
    write_epoch_predictions_atomic(
        bad_path,
        [{"sample_id": "a.png", "oof_fold": "00", "epoch": "1", "y_true": "0", "p_defect_raw": "0.2"}],
        ("sample_id", "oof_fold", "epoch", "y_true", "p_defect_raw"),
        expected_row_count=1,
        manifest_payload={"checkpoint_id": "fold_00_epoch_002", "oof_fold": "00", "epoch": 2},
    )

    try:
        validate_summary_inputs(
            collect_prediction_files(dynamics_root),
            expected_folds=[0],
            expected_epochs=[1, 2],
            output_root=dynamics_root / "summary",
        )
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("CSV row epoch should match the file path epoch")


def test_summary_validation_rejects_sample_with_incomplete_epoch_count(tmp_path: Path) -> None:
    rows = [
        {"epoch": "1", "y_true": "0", "p_defect_raw": "0.1", "raw_correct": "1"},
    ]
    summary = summarize_trajectory("normal_train/a.png", rows)
    try:
        validate_summary_inputs([], expected_folds=[], expected_epochs=[1, 2], output_root=tmp_path, summaries=[summary])
    except ValueError as exc:
        assert "epoch_count" in str(exc)
    else:
        raise AssertionError("incomplete sample epoch_count should fail")


def test_summary_validation_rejects_sample_with_wrong_epoch_set(tmp_path: Path) -> None:
    summary = {"sample_id": "normal_train/a.png", "epoch_count": "2"}
    try:
        validate_summary_inputs(
            [],
            expected_folds=[],
            expected_epochs=[1, 2],
            output_root=tmp_path,
            summaries=[summary],
            sample_epoch_sets={"normal_train/a.png": {1}},
        )
    except ValueError as exc:
        assert "epoch set" in str(exc)
    else:
        raise AssertionError("sample epoch set should match the expected epochs")


def test_method_score_column_is_stable() -> None:
    assert method_score_column("gap_critical") == "gap_critical_score"
    assert method_score_column("grad_align_guard") == "grad_align_guard_score"


def test_fp_only_selection_never_returns_defect_rows() -> None:
    selected = select_value_rows(
        [
            {"sample_id": "train/d0.png", "y_true": "1", "gap_critical_score": "999"},
            {"sample_id": "normal_train/n0.png", "y_true": "0", "gap_critical_score": "1"},
        ],
        "gap_critical",
        1,
    )

    assert [row["sample_id"] for row in selected] == ["normal_train/n0.png"]


def test_guard_selection_requires_explicit_defect_guard_budget() -> None:
    try:
        select_value_rows(
            [{"sample_id": "normal_train/n0.png", "y_true": "0", "grad_align_guard_score": "1"}],
            "grad_align_guard",
            1,
        )
    except ValueError as exc:
        assert "defect guard budget" in str(exc)
    else:
        raise AssertionError("guard methods should fail without explicit defect guard budget")


def test_replay_manifest_builder_writes_method_budget_run_tree(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_root = dataset_root / "manifests"
    train_rows = [
        {"Filename": "d0.png", "canonical_image_relpath": "train/d0.png"},
        {"Filename": "d1.png", "canonical_image_relpath": "train/d1.png"},
    ]
    normal_rows = [
        {"Filename": "n0.png", "canonical_image_relpath": "normal_train/n0.png"},
        {"Filename": "n1.png", "canonical_image_relpath": "normal_train/n1.png"},
    ]
    val_rows = [{"Filename": "vd.png", "canonical_image_relpath": "val/vd.png"}]
    normal_val_rows = [{"Filename": "vn.png", "canonical_image_relpath": "normal_val/vn.png"}]
    write_csv(manifest_root / "train_manifest.csv", train_rows, ["Filename", "canonical_image_relpath"])
    write_csv(manifest_root / "normal_train_manifest.csv", normal_rows, ["Filename", "canonical_image_relpath"])
    write_csv(manifest_root / "val_model_manifest.csv", val_rows, ["Filename", "canonical_image_relpath"])
    write_csv(manifest_root / "normal_val_model_manifest.csv", normal_val_rows, ["Filename", "canonical_image_relpath"])

    value_rows = [
        {"sample_id": "normal_train/n0.png", "y_true": "0", "gap_critical_score": "0.9"},
        {"sample_id": "normal_train/n1.png", "y_true": "0", "gap_critical_score": "0.1"},
        {"sample_id": "train/d0.png", "y_true": "1", "gap_critical_score": "2.0"},
    ]
    run_root = tmp_path / "family" / "experiments" / "oof" / "03_replay_manifests" / "gap_critical" / "budget_00001"

    summary = build_replay_manifest_for_run(
        dataset_root=dataset_root,
        value_rows=value_rows,
        method_id="gap_critical",
        budget=1,
        run_id="gap_critical_b00001_r001",
        run_root=run_root,
        force=False,
    )

    assert summary["selected_total"] == 1
    assert summary["selected_normal"] == 1
    assert summary["selected_defect"] == 0
    out_normals = read_csv(run_root / "gap_critical_b00001_r001" / "normal_train_manifest.csv")
    out_defects = read_csv(run_root / "gap_critical_b00001_r001" / "train_manifest.csv")
    selection = read_csv(run_root / "gap_critical_b00001_r001" / "selection_manifest.csv")
    assert len(out_normals) == 3
    assert len(out_defects) == 2
    assert out_normals[-1]["Filename"].startswith("replay__gap_critical_b00001_r001__00001__")
    assert out_normals[-1]["canonical_image_relpath"] == "normal_train/n0.png"
    assert selection[0]["sample_value_score"] == "0.9000000000"
