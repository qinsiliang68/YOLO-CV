from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from stage1_gapvalue240.campaign_benchmark import (
    compare_training_numerical_parity,
    summarize_crossed_telemetry_benchmarks,
)
from scripts.stage1_gapvalue240.local_dynamic_telemetry_benchmark import parse_args


def test_local_benchmark_accepts_multi_epoch_smoke() -> None:
    args = parse_args(["--epochs", "3", "--skip-baseline"])

    assert args.epochs == 3
    assert args.skip_baseline is True


def _write_checkpoint(path: Path, *, weight_delta: float = 0.0) -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0 + weight_delta, 2.0]]))
        model.bias.copy_(torch.tensor([0.5]))
    torch.save(
        {
            "epoch": 0,
            "updates": 3,
            "model": None,
            "ema": model,
            "optimizer": {
                "state": {0: {"momentum_buffer": torch.tensor([1.0, 2.0])}},
                "param_groups": [{"lr": 0.01}],
            },
            "train_args": {"project": str(path.parent)},
        },
        path,
    )


def _write_results(path: Path, *, loss_delta: float = 0.0, elapsed: float = 1.0) -> None:
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "time": elapsed,
                "train/loss": 0.25 + loss_delta,
                "metrics/accuracy_top1": 0.75,
            }
        ]
    ).to_csv(path, index=False)


def test_numerical_parity_ignores_only_runtime_metadata(tmp_path: Path) -> None:
    baseline_checkpoint = tmp_path / "baseline.pt"
    telemetry_checkpoint = tmp_path / "telemetry.pt"
    baseline_results = tmp_path / "baseline.csv"
    telemetry_results = tmp_path / "telemetry.csv"
    _write_checkpoint(baseline_checkpoint)
    _write_checkpoint(telemetry_checkpoint)
    _write_results(baseline_results, elapsed=1.0)
    _write_results(telemetry_results, elapsed=2.0)

    report = compare_training_numerical_parity(
        baseline_checkpoint=baseline_checkpoint,
        telemetry_checkpoint=telemetry_checkpoint,
        baseline_results=baseline_results,
        telemetry_results=telemetry_results,
    )

    assert report["numerical_parity_passed"] is True
    assert report["ema"]["different_values"] == 0
    assert report["optimizer"]["different_values"] == 0
    assert report["results"]["excluded_columns"] == ["time"]


def test_numerical_parity_detects_weight_and_metric_drift(tmp_path: Path) -> None:
    baseline_checkpoint = tmp_path / "baseline.pt"
    telemetry_checkpoint = tmp_path / "telemetry.pt"
    baseline_results = tmp_path / "baseline.csv"
    telemetry_results = tmp_path / "telemetry.csv"
    _write_checkpoint(baseline_checkpoint)
    _write_checkpoint(telemetry_checkpoint, weight_delta=0.1)
    _write_results(baseline_results)
    _write_results(telemetry_results, loss_delta=0.01)

    report = compare_training_numerical_parity(
        baseline_checkpoint=baseline_checkpoint,
        telemetry_checkpoint=telemetry_checkpoint,
        baseline_results=baseline_results,
        telemetry_results=telemetry_results,
    )

    assert report["numerical_parity_passed"] is False
    assert report["ema"]["different_values"] == 1
    assert report["results"]["different_values"] == 1


def test_crossed_benchmark_summary_removes_execution_order_warmup_bias() -> None:
    baseline_first = {
        "status": "PASS",
        "epochs": 2,
        "batch": 8,
        "execution_order": ["LOCAL_BASELINE", "LOCAL_TELEMETRY"],
        "numerical_parity": {"numerical_parity_passed": True},
        "runs": [
            {"run_id": "LOCAL_BASELINE", "duration_seconds": 10.0, "telemetry_bytes": 0},
            {"run_id": "LOCAL_TELEMETRY", "duration_seconds": 5.1, "telemetry_bytes": 32000},
        ],
    }
    telemetry_first = {
        "status": "PASS",
        "epochs": 2,
        "batch": 8,
        "execution_order": ["LOCAL_TELEMETRY", "LOCAL_BASELINE"],
        "numerical_parity": {"numerical_parity_passed": True},
        "runs": [
            {"run_id": "LOCAL_TELEMETRY", "duration_seconds": 10.2, "telemetry_bytes": 32000},
            {"run_id": "LOCAL_BASELINE", "duration_seconds": 5.0, "telemetry_bytes": 0},
        ],
    }

    report = summarize_crossed_telemetry_benchmarks(baseline_first, telemetry_first)

    assert report["status"] == "PASS"
    assert report["numerical_parity_both_orders"] is True
    assert report["position_matched_runtime_ratios"] == {"first": 1.02, "second": 1.02}
    assert report["order_balanced_runtime_ratio"] == 1.02
    assert report["telemetry_bytes_per_epoch"] == 16000
