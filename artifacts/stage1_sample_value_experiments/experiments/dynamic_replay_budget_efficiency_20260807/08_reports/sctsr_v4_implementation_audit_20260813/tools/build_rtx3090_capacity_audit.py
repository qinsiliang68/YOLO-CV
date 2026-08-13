from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORICAL_ROOT = Path("artifacts/stage1_phase1_hn_rn_40runs_20260630_redownload")
TASKBOOK = Path(
    "artifacts/stage1_sample_value_experiments/experiments/"
    "dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/"
    "SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md"
)
EXPECTED_GPU = "NVIDIA GeForce RTX 3090"
BASE_DENOMINATOR = 120_000
BASE_STEPS_PER_EPOCH = 938
PARENT_EPOCHS = 120
CHILD_EPOCHS = 80
CHILDREN_PER_SEED = 8
FORMAL_EPOCHS_PER_HISTORICAL_RUN = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _nearest_rank(values: list[float], proportion: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(proportion * len(ordered)) - 1))
    return ordered[index]


def load_historical_runs(repository_root: str | Path) -> list[dict[str, Any]]:
    repository = Path(repository_root).resolve()
    source_root = repository / HISTORICAL_ROOT
    paths = sorted(source_root.rglob("run_meta.json"), key=lambda item: item.as_posix())
    if len(paths) != 40:
        raise ValueError(f"expected exactly 40 historical run_meta.json files, observed {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        gpu_names = data.get("environment", {}).get("torch", {}).get("gpu_names", [])
        config = data.get("config", {})
        counts = data.get("dataset_counts", {})
        match = re.search(r"(?:^|/)node(\d+)(?:/|$)", path.relative_to(repository).as_posix())
        row = {
            "source_path": path.relative_to(repository).as_posix(),
            "source_bytes": path.stat().st_size,
            "source_sha256": _sha256(path),
            "node_id": int(match.group(1)) if match else -1,
            "experiment_id": path.parents[2].name,
            "run_name": str(data.get("run_name", "")),
            "status": str(data.get("status", "")),
            "duration_sec": float(data.get("duration_sec", -1)),
            "duration_hours": float(data.get("duration_sec", -1)) / 3600.0,
            "model_key": str(data.get("model_key", "")),
            "epochs": int(config.get("epochs", -1)),
            "batch": int(config.get("batch", -1)),
            "imgsz": int(config.get("imgsz", -1)),
            "workers": int(config.get("workers", -1)),
            "training_seed": int(config.get("seed", -1)),
            "train_image_count": int(counts.get("train_no_target", 0)) + int(counts.get("train_target_defect", 0)),
            "gpu_name": str(gpu_names[0]) if len(gpu_names) == 1 else "INVALID_GPU_INVENTORY",
            "python_version": str(data.get("environment", {}).get("python_version", "")),
            "torch_version": str(data.get("environment", {}).get("torch", {}).get("torch_version", "")),
        }
        expected = {
            "status": "ok",
            "model_key": "l",
            "epochs": 200,
            "batch": 128,
            "imgsz": 224,
            "gpu_name": EXPECTED_GPU,
        }
        mismatches = {key: {"observed": row[key], "expected": value} for key, value in expected.items() if row[key] != value}
        if mismatches or row["duration_sec"] <= 0:
            raise ValueError(f"historical timing identity mismatch for {row['source_path']}: {mismatches}")
        rows.append(row)
    return rows


def simulate_phase(*, seed_count: int, machine_count: int) -> dict[str, Any]:
    if seed_count <= 0 or machine_count <= 0:
        raise ValueError("seed_count and machine_count must be positive")
    now = 0
    free_machines = list(range(machine_count))
    parent_queue = list(range(seed_count))
    child_queue: list[int] = []
    running: list[tuple[int, int, str, int]] = []
    timeline: list[dict[str, Any]] = []
    while parent_queue or child_queue or running:
        while free_machines and (parent_queue or child_queue):
            machine = free_machines.pop(0)
            if parent_queue:
                seed_index = parent_queue.pop(0)
                role = "COMMON_PARENT_E1_E120"
                duration = PARENT_EPOCHS
            else:
                seed_index = child_queue.pop(0)
                role = "CHILD_E121_E200"
                duration = CHILD_EPOCHS
            end = now + duration
            heapq.heappush(running, (end, machine, role, seed_index))
            timeline.append(
                {
                    "machine_index": machine,
                    "seed_index": seed_index,
                    "role": role,
                    "start_epoch_units": now,
                    "end_epoch_units": end,
                }
            )
        now = running[0][0]
        completed: list[tuple[int, int, str, int]] = []
        while running and running[0][0] == now:
            completed.append(heapq.heappop(running))
        for _, machine, role, seed_index in sorted(completed, key=lambda item: item[1]):
            free_machines.append(machine)
            if role == "COMMON_PARENT_E1_E120":
                child_queue.extend([seed_index] * CHILDREN_PER_SEED)
        free_machines.sort()
    total_work = seed_count * (PARENT_EPOCHS + CHILDREN_PER_SEED * CHILD_EPOCHS)
    granularity = math.ceil(total_work / (machine_count * 40)) * 40
    machine_loads = {
        str(machine): sum(row["end_epoch_units"] - row["start_epoch_units"] for row in timeline if row["machine_index"] == machine)
        for machine in range(machine_count)
    }
    return {
        "seed_count": seed_count,
        "machine_count": machine_count,
        "job_count": seed_count * (1 + CHILDREN_PER_SEED),
        "parent_job_count": seed_count,
        "child_job_count": seed_count * CHILDREN_PER_SEED,
        "total_work_epoch_units": total_work,
        "capacity_lower_bound_epoch_units": math.ceil(total_work / machine_count),
        "granularity_lower_bound_epoch_units": granularity,
        "makespan_epoch_units": now,
        "machine_load_epoch_units": machine_loads,
        "timeline": timeline,
    }


def sctsr_workload(*, seed_count: int) -> dict[str, Any]:
    parent_base_epochs = PARENT_EPOCHS
    child_base_epochs = CHILDREN_PER_SEED * CHILD_EPOCHS
    total_base_epochs = parent_base_epochs + child_base_epochs
    u_full_calls = 80 * 600
    u_stop_calls = 40 * 600
    f_calls = 40 * BASE_STEPS_PER_EPOCH
    replay_calls = 4 * u_full_calls + u_stop_calls + 2 * f_calls
    replay_occurrences = 4 * (80 * 600) + (40 * 600) + 2 * (40 * 1_200)
    base_steps = total_base_epochs * BASE_STEPS_PER_EPOCH
    base_rows = total_base_epochs * BASE_DENOMINATOR
    return {
        "seed_count": seed_count,
        "arm_count": CHILDREN_PER_SEED,
        "physical_jobs_per_seed": 1 + CHILDREN_PER_SEED,
        "parent_base_epochs_per_seed": parent_base_epochs,
        "child_base_epochs_per_seed": child_base_epochs,
        "total_base_epochs_per_seed": total_base_epochs,
        "base_steps_per_seed": base_steps,
        "base_occurrence_rows_per_seed": base_rows,
        "replay_microbatch_calls_per_seed": replay_calls,
        "replay_occurrences_per_seed": replay_occurrences,
        "replay_call_derivation": {
            "four_full_U_arms": 4 * u_full_calls,
            "T_TO_NR_first_40_epochs": u_stop_calls,
            "two_F_arms": 2 * f_calls,
            "F_call_rule": "40 epochs * 938 occupied base steps; 262 slots have two occurrences and 676 have one",
        },
        "aggregate_physical_jobs": seed_count * (1 + CHILDREN_PER_SEED),
        "aggregate_base_epochs": seed_count * total_base_epochs,
        "aggregate_base_steps": seed_count * base_steps,
        "aggregate_base_occurrence_rows": seed_count * base_rows,
        "aggregate_replay_microbatch_calls": seed_count * replay_calls,
        "aggregate_replay_occurrences": seed_count * replay_occurrences,
    }


def _reference_estimate(reference_hours: float, discovery_units: int, confirmation_units: int) -> dict[str, float]:
    discovery = discovery_units / FORMAL_EPOCHS_PER_HISTORICAL_RUN * reference_hours
    confirmation = confirmation_units / FORMAL_EPOCHS_PER_HISTORICAL_RUN * reference_hours
    total = discovery + confirmation
    return {
        "reference_200_epoch_run_hours": round(reference_hours, 6),
        "discovery_hours": round(discovery, 3),
        "confirmation_hours": round(confirmation, 3),
        "hours": round(total, 3),
        "calendar_days": round(total / 24.0, 3),
    }


def build_report(repository_root: str | Path) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    taskbook = repository / TASKBOOK
    if not taskbook.is_file():
        raise ValueError(f"taskbook missing: {taskbook}")
    rows = load_historical_runs(repository)
    durations = [row["duration_hours"] for row in rows]
    closest = [row["duration_hours"] for row in rows if row["train_image_count"] == 120_600]
    if len(closest) != 2:
        raise ValueError(f"expected two closest 120600-image calibration runs, observed {len(closest)}")
    discovery = simulate_phase(seed_count=8, machine_count=10)
    confirmation = simulate_phase(seed_count=14, machine_count=10)
    references = {
        "closest_120600_mean": statistics.mean(closest),
        "all_40_median": statistics.median(durations),
        "all_40_nearest_rank_p90": _nearest_rank(durations, 0.90),
        "all_40_maximum": max(durations),
    }
    base_estimates = {
        name: _reference_estimate(value, discovery["makespan_epoch_units"], confirmation["makespan_epoch_units"])
        for name, value in references.items()
    }
    reserve_percent = 20
    p90_base = base_estimates["all_40_nearest_rank_p90"]["hours"]
    reserved_hours = p90_base * (1 + reserve_percent / 100)
    historical_counts = Counter(row["train_image_count"] for row in rows)
    return {
        "schema_version": "stage1.sctsr.rtx3090_capacity_audit.v1",
        "status": "PASS_WITH_EXPLICIT_RUNTIME_UNCERTAINTY",
        "semantic_status": "ENGINEERING_CAPACITY_ESTIMATE_NOT_SCIENTIFIC_RESULT_NOT_TRAINING_AUTHORIZATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": repository.as_posix(),
        "taskbook": {
            "path": TASKBOOK.as_posix(),
            "bytes": taskbook.stat().st_size,
            "sha256": _sha256(taskbook),
        },
        "hardware_contract": {
            "machine_count": 10,
            "gpu_per_machine": 1,
            "gpu_name": EXPECTED_GPU,
            "total_gpu_count": 10,
            "source": "USER_CONFIRMED_2026_08_14",
        },
        "historical_evidence": {
            "root": HISTORICAL_ROOT.as_posix(),
            "run_count": len(rows),
            "all_status_ok": all(row["status"] == "ok" for row in rows),
            "all_same_gpu_model_hyperparameters": True,
            "duration_hours": {
                "minimum": round(min(durations), 6),
                "median": round(statistics.median(durations), 6),
                "mean": round(statistics.mean(durations), 6),
                "nearest_rank_p90": round(_nearest_rank(durations, 0.90), 6),
                "maximum": round(max(durations), 6),
            },
            "train_image_count_distribution": {str(key): historical_counts[key] for key in sorted(historical_counts)},
            "closest_120600_run_count": len(closest),
            "closest_120600_mean_hours": round(statistics.mean(closest), 6),
            "limitations": [
                "Historical jobs append replay images to the dataset; SCTSR v4 injects separate replay microbatches into fixed base steps.",
                "Historical jobs do not measure SCTSR v4 per-occurrence Parquet logging, atomic epoch publication, or resume validation overhead.",
                "The 40-run distribution contains node and dataset-size heterogeneity; the maximum includes a visibly slower node30 pair.",
            ],
            "rows": rows,
        },
        "workload": {
            "discovery": sctsr_workload(seed_count=8),
            "confirmation": sctsr_workload(seed_count=14),
            "all_if_discovery_passes": sctsr_workload(seed_count=22),
        },
        "dependency_schedule": {
            "phase_order": ["DISCOVERY_8_UNSEEN_SEEDS", "FREEZE_AND_DECISION", "CONFIRMATION_14_NEW_UNSEEN_SEEDS"],
            "human_decision_gap_included": False,
            "discovery": discovery,
            "confirmation": confirmation,
            "combined_makespan_epoch_units_excluding_decision_gap": discovery["makespan_epoch_units"] + confirmation["makespan_epoch_units"],
        },
        "estimate": {
            "base_only_dependency_aware": base_estimates,
            "unmeasured_overhead": {
                "status": "NOT_DIRECTLY_MEASURED_ON_RTX3090",
                "components": [
                    "6,402,880 separate replay microbatch forward/backward calls across 22 seeds",
                    "2,006,400,000 base occurrence rows plus 6,864,000 replay occurrence rows",
                    "Zstd Parquet compression, one-second telemetry, checkpoint hashing, atomic publish, validation, and closeout",
                    "machine synchronization, retries, and discovery-to-confirmation review latency",
                ],
                "included_in_recommended_reserve_percent": reserve_percent,
                "interpretation": "capacity buffer, not a measured speed factor and not a confidence interval",
            },
            "recommended_capacity_reservation": {
                "basis": "all_40_nearest_rank_p90 base-only dependency schedule plus 20 percent unmeasured-overhead reserve",
                "hours": round(reserved_hours, 3),
                "calendar_days_unrounded": round(reserved_hours / 24.0, 3),
                "calendar_days": math.ceil(reserved_hours / 24.0),
                "operational_recommendation": "reserve 8 continuous calendar days after release, excluding human review pauses",
            },
            "stop_after_discovery_if_gate_fails": {
                "p90_base_hours": round(discovery["makespan_epoch_units"] / 200 * references["all_40_nearest_rank_p90"], 3),
                "p90_plus_20_percent_hours": round(discovery["makespan_epoch_units"] / 200 * references["all_40_nearest_rank_p90"] * 1.2, 3),
            },
        },
        "required_pre_release_calibration": {
            "status": "REQUIRED_FOR_TIGHTER_ESTIMATE",
            "instruction": "On one target RTX3090, run registered engineering-only NR and replay one-epoch canaries with full 120000-row ledgers; bind wall time, GPU identity, artifact bytes/SHA, and forbid formal seed/test access.",
        },
        "formal_training_started": False,
        "assignments_generated": False,
        "engineering_gate_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    base = report["estimate"]["base_only_dependency_aware"]
    reserve = report["estimate"]["recommended_capacity_reservation"]
    discovery = report["estimate"]["stop_after_discovery_if_gate_fails"]
    hist = report["historical_evidence"]
    workload = report["workload"]["all_if_discovery_passes"]
    return "\n".join(
        [
            "# SCTSR v4：10×RTX 3090 训练容量审计",
            "",
            f"状态：`{report['status']}`。这是工程排期证据，不是训练授权或方法有效性证据。",
            "",
            "## 可直接采用的排期结论",
            "",
            f"- 只计算历史同卡基础训练速度时：约 {base['closest_120600_mean']['hours']:.1f}–{base['all_40_nearest_rank_p90']['hours']:.1f} 小时，即 {base['closest_120600_mean']['calendar_days']:.2f}–{base['all_40_nearest_rank_p90']['calendar_days']:.2f} 天。",
            f"- 建议实际预留：`{reserve['calendar_days']} 个连续自然日`（{reserve['hours']:.1f} 小时容量），不含 discovery 决策停顿。",
            f"- 若 discovery 门失败并按合同停止：p90 加 20% 容量约 {discovery['p90_plus_20_percent_hours']:.1f} 小时。",
            "",
            "## 同型号历史证据",
            "",
            f"- 40/40 个 run 为 RTX 3090、yolo11l、200 epochs、batch 128、imgsz 224 且 status=ok。",
            f"- 历史全体中位数 {hist['duration_hours']['median']:.3f} 小时；最近似 120,600-image 两次均值 {hist['closest_120600_mean_hours']:.3f} 小时；nearest-rank p90 {hist['duration_hours']['nearest_rank_p90']:.3f} 小时。",
            "",
            "## 依赖与工作量",
            "",
            "- discovery：8 个 parent + 64 个 child，在 10 卡上的依赖 makespan 为 680 epoch-time units。",
            "- confirmation：14 个 parent + 112 个 child，最优容量粒度下为 1080 epoch-time units。",
            f"- 全部通过时：{workload['aggregate_physical_jobs']} 个物理训练 job、{workload['aggregate_base_epochs']:,} 个 base epochs、{workload['aggregate_base_steps']:,} 个 base optimizer steps。",
            f"- 另有 {workload['aggregate_replay_microbatch_calls']:,} 次独立 replay microbatch 调用、{workload['aggregate_replay_occurrences']:,} 次 replay occurrence。",
            f"- occurrence 证据规模为 {workload['aggregate_base_occurrence_rows']:,} 条 base rows，必须单独计入 CPU、磁盘和 Parquet 开销。",
            "",
            "## 为什么不是一个精确小时数",
            "",
            "历史 3090 run 将附加样本并入 Dataset，而 v4 在固定 base step 内执行独立 replay forward/backward，并记录全量 occurrence/telemetry/事务证据；历史记录没有直接测量这些 v4 开销。因此 20% 是容量缓冲，不是测得的加速/减速系数，也不是置信区间。正式排队前仍需在一台目标 3090 上跑 NR 与 replay 的一 epoch 工程校准。",
            "",
            f"Taskbook SHA-256：`{report['taskbook']['sha256']}`。",
            "",
        ]
    )


def write_report(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    report = build_report(repository_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "RTX3090_CAPACITY_AUDIT.json"
    csv_path = destination / "RTX3090_HISTORICAL_RUN_LEDGER.csv"
    markdown_path = destination / "RTX3090_CAPACITY_REPORT.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(report["historical_evidence"]["rows"], csv_path)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    receipt = {
        "status": report["status"],
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (json_path, csv_path, markdown_path)
        },
        "formal_training_started": False,
    }
    receipt_path = destination / "RTX3090_CAPACITY_OUTPUT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_report(args.repository_root, args.output_root)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
