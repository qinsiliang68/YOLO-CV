from __future__ import annotations

import torch

from stage1_gapvalue240.checkpoint_drift import (
    best_epoch_candidates,
    checkpoint_model_state,
    checkpoint_optimizer_metadata,
    compare_state_dicts,
    build_paired_checkpoint_deltas,
    publish_checkpoint_pair_analysis,
    summarize_checkpoint_cohorts,
    scan_checkpoint_triplet,
    scan_inventory_checkpoints,
)


def test_state_drift_records_shape_changes_and_layerwise_metrics() -> None:
    left = {
        "model.0.conv.weight": torch.tensor([1.0, 2.0]),
        "model.0.bn.running_mean": torch.tensor([0.0, 1.0]),
        "model.10.linear.weight": torch.tensor([1.0, 2.0, 3.0]),
        "model.10.linear.bias": torch.tensor([0.0, 0.0, 0.0]),
    }
    right = {
        "model.0.conv.weight": torch.tensor([2.0, 2.0]),
        "model.0.bn.running_mean": torch.tensor([1.0, 1.0]),
        "model.10.linear.weight": torch.tensor([1.0, 2.0]),
        "model.10.linear.bias": torch.tensor([0.0, 0.0]),
    }

    result = compare_state_dicts(left, right, transition="initial_to_best")
    tensors = result.tensor_drift.set_index("tensor_name")

    assert tensors.loc["model.0.conv.weight", "comparison_status"] == "MATCHED"
    assert tensors.loc["model.10.linear.weight", "comparison_status"] == "SHAPE_MISMATCH"
    assert tensors.loc["model.10.linear.bias", "comparison_status"] == "SHAPE_MISMATCH"
    assert abs(tensors.loc["model.0.conv.weight", "delta_l2"] - 1.0) < 1e-12
    block0 = result.group_drift.set_index("group_name").loc["block_0"]
    assert block0["matched_float_tensors"] == 2
    assert block0["matched_elements"] == 4
    assert block0["relative_l2"] > 0
    overall = result.group_drift.set_index("group_name").loc["ALL"]
    assert overall["matched_float_tensors"] == 2
    assert overall["matched_elements"] == 4
    assert result.summary["shape_mismatch_tensors"] == 2


def test_optimizer_metadata_distinguishes_active_and_empty_lr_groups() -> None:
    optimizer = {
        "state": {10: {"momentum_buffer": torch.tensor([1.0])}},
        "param_groups": [
            {
                "params": [],
                "lr": 0.03,
                "initial_lr": 0.03,
                "momentum": 0.937,
                "weight_decay": 0.001,
                "param_group": None,
                "use_muon": False,
            },
            {
                "params": [10],
                "lr": 0.01,
                "initial_lr": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.001,
                "param_group": "muon",
                "use_muon": True,
            },
        ],
    }

    groups, summary = checkpoint_optimizer_metadata(optimizer)

    assert len(groups) == 2
    assert summary["optimizer_group_count"] == 2
    assert summary["active_optimizer_group_count"] == 1
    assert summary["empty_optimizer_group_count"] == 1
    assert summary["optimizer_state_entries"] == 1
    assert groups.loc[groups["group_index"] == 1, "use_muon"].iloc[0]


def test_checkpoint_model_state_prefers_last_ema_and_rejects_missing_model() -> None:
    model = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    checkpoint = {"model": model, "ema": ema}

    assert torch.equal(
        checkpoint_model_state(checkpoint, anchor="initial")["weight"],
        model.state_dict()["weight"],
    )
    assert torch.equal(
        checkpoint_model_state(checkpoint, anchor="best")["weight"],
        model.state_dict()["weight"],
    )
    assert torch.equal(
        checkpoint_model_state(checkpoint, anchor="last")["weight"],
        ema.state_dict()["weight"],
    )

    try:
        checkpoint_model_state({}, anchor="last")
    except ValueError as exc:
        assert "model state" in str(exc).lower()
    else:
        raise AssertionError("missing model state must fail")


def test_best_epoch_candidates_reports_all_metric_ties() -> None:
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "train_metrics": {
            "metrics/accuracy_top1": 0.945,
            "metrics/accuracy_top5": 1.0,
            "val/loss": 0.248,
            "fitness": 0.9725,
        },
    }
    epochs = {
        "epoch": [193, 194, 195, 196],
        "metrics/accuracy_top1": [0.944, 0.945, 0.9449, 0.945],
        "metrics/accuracy_top5": [1.0, 1.0, 1.0, 1.0],
        "val/loss": [0.20, 0.248, 0.21, 0.248],
    }

    result = best_epoch_candidates(checkpoint, __import__("pandas").DataFrame(epochs))

    assert result["checkpoint_epoch"] == -1
    assert result["candidate_count"] == 2
    assert result["candidate_epochs"] == "194;196"
    assert result["candidate_epoch_earliest"] == 194
    assert result["candidate_epoch_latest"] == 196
    assert result["candidate_basis"] == "checkpoint_train_metrics_top1_top5"


def test_scan_checkpoint_triplet_streams_all_anchors(tmp_path) -> None:
    initial_model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    best_model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    last_ema = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.no_grad():
        best_model[0].weight.add_(0.5)
        last_ema[0].weight.copy_(best_model[0].weight + 0.1)
    initial_path = tmp_path / "initial.pt"
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    torch.save({"model": initial_model, "version": "test"}, initial_path)
    torch.save(
        {
            "model": best_model,
            "epoch": -1,
            "train_metrics": {
                "metrics/accuracy_top1": 0.8,
                "metrics/accuracy_top5": 1.0,
                "fitness": 0.9,
            },
            "version": "test",
        },
        best_path,
    )
    torch.save(
        {
            "model": best_model,
            "ema": last_ema,
            "epoch": 1,
            "optimizer": {
                "state": {1: {}},
                "param_groups": [
                    {
                        "params": [1],
                        "param_group": "muon",
                        "lr": 0.01,
                        "initial_lr": 0.01,
                        "momentum": 0.937,
                        "weight_decay": 0.001,
                        "nesterov": True,
                        "use_muon": True,
                    }
                ],
            },
            "version": "test",
        },
        last_path,
    )
    epochs = __import__("pandas").DataFrame(
        {
            "epoch": [1, 2],
            "metrics/accuracy_top1": [0.7, 0.8],
            "metrics/accuracy_top5": [1.0, 1.0],
            "val/loss": [0.3, 0.2],
        }
    )

    result = scan_checkpoint_triplet(
        run_slot="RUN_TEST",
        initial_checkpoint=initial_path,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        epoch_rows=epochs,
    )

    assert set(result.tensor_drift["transition"]) == {
        "initial_to_best",
        "best_to_last_ema",
    }
    assert set(result.group_drift["group_name"]) >= {"ALL", "0"}
    assert len(result.optimizer_groups) == 1
    assert result.optimizer_groups.loc[0, "active"]
    assert result.run_summary["run_slot"] == "RUN_TEST"
    assert result.run_summary["last_checkpoint_epoch"] == 1
    assert result.best_epoch_candidates["candidate_epochs"] == "2"


def test_scan_inventory_checkpoints_publishes_atomic_complete_tables(tmp_path) -> None:
    initial_model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    best_model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    last_ema = torch.nn.Sequential(torch.nn.Linear(2, 2))
    initial_path = tmp_path / "initial.pt"
    torch.save({"model": initial_model, "version": "test"}, initial_path)
    attempt = (
        tmp_path
        / "extracted"
        / "stage1_gapvalue240_P1_upload"
        / "runs"
        / "RUN_001"
        / "attempt_x"
    )
    checkpoint_dir = attempt / "03_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "model": best_model,
            "epoch": -1,
            "train_metrics": {
                "metrics/accuracy_top1": 0.8,
                "metrics/accuracy_top5": 1.0,
            },
        },
        checkpoint_dir / "best.pt",
    )
    torch.save(
        {
            "ema": last_ema,
            "epoch": 1,
            "optimizer": {"state": {}, "param_groups": []},
        },
        checkpoint_dir / "last.pt",
    )
    pd = __import__("pandas")
    inventory = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "triad_id": "TRIAD_001",
                "arm": "T",
                "package": "P1",
                "attempt_id": "attempt_x",
            }
        ]
    )
    inventory_path = tmp_path / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    epochs = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_001"],
            "epoch": [1, 2],
            "metrics/accuracy_top1": [0.7, 0.8],
            "metrics/accuracy_top5": [1.0, 1.0],
            "val/loss": [0.3, 0.2],
        }
    )
    epoch_path = tmp_path / "epochs.csv"
    epochs.to_csv(epoch_path, index=False)
    output = tmp_path / "analysis.inprogress"

    report = scan_inventory_checkpoints(
        extracted_root=tmp_path / "extracted",
        inventory_path=inventory_path,
        initial_checkpoint=initial_path,
        epoch_curves_path=epoch_path,
        output_dir=output,
        expected_runs=1,
        expected_epochs=2,
    )

    assert report["canonical_runs"] == 1
    assert report["tensor_drift_rows"] == 4
    assert pd.read_csv(output / "tables/checkpoint_run_summary.csv")["run_slot"].tolist() == ["RUN_001"]
    assert not list(output.rglob("*.tmp"))


def test_build_paired_checkpoint_deltas_preserves_each_control() -> None:
    pd = __import__("pandas")
    metadata = pd.DataFrame(
        {
            "run_slot": ["T", "R1", "R2"],
            "triad_id": ["Q", "Q", "Q"],
            "arm": ["T", "R1", "R2"],
        }
    )
    drift = pd.DataFrame(
        {
            "run_slot": ["T", "R1", "R2"],
            "transition": ["best_to_last_ema"] * 3,
            "group_name": ["ALL"] * 3,
            "matched_float_tensors": [10] * 3,
            "matched_elements": [100] * 3,
            "left_l2": [10.0] * 3,
            "right_l2": [10.0] * 3,
            "delta_l2": [2.0, 1.0, 4.0],
            "relative_l2": [0.2, 0.1, 0.4],
            "cosine_similarity": [0.9, 0.95, 0.8],
        }
    )

    paired = build_paired_checkpoint_deltas(drift, metadata)

    assert len(paired) == 2
    r1 = paired.set_index("control_arm").loc["R1"]
    r2 = paired.set_index("control_arm").loc["R2"]
    assert r1["delta_relative_l2"] == 0.1
    assert r2["delta_relative_l2"] == -0.2
    assert abs(r1["delta_cosine_similarity"] + 0.05) < 1e-12

    outcomes = pd.DataFrame(
        {"triad_id": ["Q"], "exclusive_cohort": ["dual_improvement"]}
    )
    summary = summarize_checkpoint_cohorts(paired, outcomes)
    relative = summary.loc[
        (summary["control_arm"] == "R1")
        & (summary["metric"] == "delta_relative_l2")
    ].iloc[0]
    assert relative["triad_count"] == 1
    assert relative["mean"] == 0.1


def test_publish_checkpoint_pair_analysis_validates_and_writes(tmp_path) -> None:
    pd = __import__("pandas")
    output = tmp_path / "report.inprogress"
    (output / "tables").mkdir(parents=True)
    metadata = pd.DataFrame(
        {
            "run_slot": ["T", "R1", "R2"],
            "triad_id": ["Q", "Q", "Q"],
            "arm": ["T", "R1", "R2"],
        }
    )
    inventory_path = tmp_path / "inventory.csv"
    metadata.to_csv(inventory_path, index=False)
    drift = pd.DataFrame(
        {
            "run_slot": ["T", "R1", "R2"],
            "transition": ["best_to_last_ema"] * 3,
            "group_name": ["ALL"] * 3,
            "matched_float_tensors": [10] * 3,
            "matched_elements": [100] * 3,
            "left_l2": [10.0] * 3,
            "right_l2": [10.0] * 3,
            "delta_l2": [2.0, 1.0, 4.0],
            "relative_l2": [0.2, 0.1, 0.4],
            "cosine_similarity": [0.9, 0.95, 0.8],
        }
    )
    drift.to_csv(output / "tables/checkpoint_block_drift.csv", index=False)
    outcomes = pd.DataFrame(
        {"triad_id": ["Q"], "exclusive_cohort": ["dual_improvement"]}
    )
    outcomes_path = tmp_path / "outcomes.csv"
    outcomes.to_csv(outcomes_path, index=False)

    report = publish_checkpoint_pair_analysis(
        output_dir=output,
        inventory_path=inventory_path,
        triad_outcomes_path=outcomes_path,
        expected_triads=1,
    )

    assert report["paired_rows"] == 2
    assert report["cohort_summary_rows"] == 10
    assert (output / "tables/checkpoint_paired_deltas.csv").is_file()
