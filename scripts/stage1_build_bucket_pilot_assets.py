from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline_common import REPO_ROOT, YOLOV11_ROOT
from stage1_formal_capacity_suite import print_step, resolve_path, resolve_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary assets for Layer-1 gate bucket pilot.")
    parser.add_argument("--config", required=True, help="Bucket pilot runtime config.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def expected_experiments(cfg: dict[str, Any], score_output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_count = int(cfg.get("bucket_count", 5) or 5)
    fixed_budget_count = int(cfg.get("fixed_budget_count", 151) or 151)
    candidate_top_k = int(cfg.get("candidate_top_k", 250) or 250)
    for signal_name in ("R", "C", "D"):
        for bucket_index in range(bucket_count):
            bucket_name = f"Q{bucket_index + 1}"
            experiment_id = f"{signal_name}-{bucket_name}"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "signal": signal_name,
                    "bucket": bucket_name,
                    "setting_name": f"{signal_name.lower()}_{bucket_name.lower()}_bucket_uniform151",
                    "candidate_scores_csv": str(score_output_dir / "experiments" / f"{experiment_id}_candidate_scores.csv"),
                    "metadata_json": str(score_output_dir / "metadata" / f"{experiment_id}_bucket_metadata.json"),
                    "n_unique": candidate_top_k // bucket_count,
                    "n_replay": fixed_budget_count,
                }
            )
    rows.append(
        {
            "experiment_id": "G0",
            "signal": "baseline",
            "bucket": "G0",
            "setting_name": "g0_uniform_random_pool250",
            "candidate_scores_csv": str(score_output_dir / "experiments" / "G0_candidate_scores.csv"),
            "metadata_json": str(score_output_dir / "metadata" / "G0_bucket_metadata.json"),
            "n_unique": fixed_budget_count,
            "n_replay": fixed_budget_count,
        }
    )
    return rows


def maybe_plot_signal(results_dir: Path, signal_name: str, rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    bucket_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ordered_rows = sorted(rows, key=lambda row: bucket_order.index(str(row["bucket"])))
    xs = [str(row["bucket"]) for row in ordered_rows]
    ys = [float(row["spec_r995"]) for row in ordered_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(xs, ys, color="#355C7D")
    ax.set_title(f"{signal_name} bucket ranking on Spec@R99.5")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Spec@R99.5")
    ax.set_ylim(0.0, max(1.0, max(ys) + 0.05))
    for bar, value in zip(bars, ys, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output_path = results_dir / f"fig_{signal_name}_bucket_ranking.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def verdict_for_signal(signal_name: str, rows: list[dict[str, Any]], g0_spec: float | None) -> dict[str, Any]:
    bucket_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ordered_rows = sorted(rows, key=lambda row: bucket_order.index(str(row["bucket"])))
    values = [float(row["spec_r995"]) for row in ordered_rows]
    q1 = values[0]
    q5 = values[-1]
    q1_gt_q5 = q1 - q5
    q1_gt_g0 = None if g0_spec is None else q1 - float(g0_spec)
    monotonic = values[0] >= values[1] >= values[2]
    reversed_monotonic = values[0] <= values[1] <= values[2]
    if q1_gt_q5 > 0.03 and (q1_gt_g0 is None or q1_gt_g0 > 0.0) and monotonic:
        verdict = "retain"
        rationale = "Q1 明显优于 Q5，且高分端整体不倒挂。"
    elif q1_gt_q5 < -0.03 and reversed_monotonic:
        verdict = "flip_direction"
        rationale = "低分桶优于高分桶，信号方向可能反了。"
    elif max(values) - min(values) < 0.02:
        verdict = "drop"
        rationale = "各桶基本持平，排序力不足。"
    else:
        verdict = "mixed"
        rationale = "存在非单调或局部增益，需结合样本分析进一步判断。"
    return {
        "signal": signal_name,
        "verdict": verdict,
        "q1_spec_r995": round(q1, 6),
        "q5_spec_r995": round(q5, 6),
        "q1_minus_q5": round(q1_gt_q5, 6),
        "q1_minus_g0": None if q1_gt_g0 is None else round(q1_gt_g0, 6),
        "monotonic_q1_q3": bool(monotonic),
        "rationale": rationale,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    cfg = load_json(config_path)

    materials_root = resolve_path(cfg.get("materials_root"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot")
    score_output_dir = resolve_path(cfg.get("score_output_dir"), base=materials_root / "score_inputs")
    results_dir = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_bucket_pilot")
    results_dir.mkdir(parents=True, exist_ok=True)

    registry_path = score_output_dir / "bucket_experiment_registry.csv"
    registry_rows = load_csv_rows(registry_path) if registry_path.exists() else expected_experiments(cfg, score_output_dir)

    summary_rows: list[dict[str, Any]] = []
    for registry_row in registry_rows:
        experiment_id = str(registry_row["experiment_id"])
        summary_dir = materials_root / experiment_id
        best_manifest = load_json_if_exists(summary_dir / "best_epoch_manifest.json")
        metadata = load_json_if_exists(Path(str(registry_row["metadata_json"])))
        if not best_manifest:
            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "signal": registry_row["signal"],
                    "bucket": registry_row["bucket"],
                    "n_unique": registry_row["n_unique"],
                    "n_replay": registry_row["n_replay"],
                    "best_epoch": "",
                    "spec_r995": "",
                    "spec_r990": "",
                    "prec_r990": "",
                    "ptr_r990": "",
                    "tau_r995": "",
                    "tau_r990": "",
                    "temperature_T": "",
                    "status": "missing",
                }
            )
            continue
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "signal": registry_row["signal"],
                "bucket": registry_row["bucket"],
                "n_unique": metadata.get("n_unique", registry_row["n_unique"]),
                "n_replay": metadata.get("n_replay", registry_row["n_replay"]),
                "best_epoch": int(best_manifest.get("epoch", -1)),
                "spec_r995": round(float(best_manifest.get("spec_at_r995", 0.0)), 6),
                "spec_r990": round(float(best_manifest.get("spec_at_r990", 0.0)), 6),
                "prec_r990": round(float(best_manifest.get("prec_at_r990", 0.0)), 6),
                "ptr_r990": round(float(best_manifest.get("ptr_at_r990", 0.0)), 6),
                "tau_r995": round(float(best_manifest.get("tau_r995", 0.0)), 6),
                "tau_r990": round(float(best_manifest.get("tau_r990", 0.0)), 6),
                "temperature_T": round(float(best_manifest.get("temperature_T", 0.0)), 6),
                "status": "completed",
            }
        )

    fieldnames = [
        "experiment_id",
        "signal",
        "bucket",
        "n_unique",
        "n_replay",
        "best_epoch",
        "spec_r995",
        "spec_r990",
        "prec_r990",
        "ptr_r990",
        "tau_r995",
        "tau_r990",
        "temperature_T",
        "status",
    ]
    write_csv(results_dir / "bucket_pilot_summary.csv", fieldnames, summary_rows)
    (results_dir / "bucket_pilot_summary.md").write_text(
        "\n".join(
            [
                "# Gate Bucket Pilot Summary",
                "",
                "| Experiment | Signal | Bucket | Unique | Replay | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Status |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                *[
                    "| {experiment_id} | {signal} | {bucket} | {n_unique} | {n_replay} | {best_epoch} | {spec_r995} | {spec_r990} | {prec_r990} | {ptr_r990} | {status} |".format(
                        **row
                    )
                    for row in summary_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    g0_row = next((row for row in summary_rows if row["experiment_id"] == "G0" and row["status"] == "completed"), None)
    g0_spec = None if g0_row is None else float(g0_row["spec_r995"])
    verdict_rows: list[dict[str, Any]] = []
    generated_plots: dict[str, str] = {}
    for signal_name in ("R", "C", "D"):
        signal_rows = [row for row in summary_rows if row["signal"] == signal_name and row["status"] == "completed"]
        if len(signal_rows) != 5:
            continue
        verdict_rows.append(verdict_for_signal(signal_name, signal_rows, g0_spec))
        plot_path = maybe_plot_signal(results_dir, signal_name, signal_rows)
        if plot_path:
            generated_plots[signal_name] = plot_path

    write_json(
        results_dir / "bucket_pilot_verdict.json",
        {
            "g0_spec_r995": g0_spec,
            "rows": verdict_rows,
            "generated_plots": generated_plots,
        },
    )
    (results_dir / "bucket_pilot_verdict.md").write_text(
        "\n".join(
            [
                "# Bucket Pilot Verdict",
                "",
                f"- G0 Spec@R99.5: `{'' if g0_spec is None else f'{g0_spec:.6f}'}`",
                "",
                "| Signal | Verdict | Q1 Spec@R99.5 | Q5 Spec@R99.5 | Q1-Q5 | Q1-G0 | Q1-Q3 Monotonic | Rationale |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
                *[
                    "| {signal} | {verdict} | {q1_spec_r995:.6f} | {q5_spec_r995:.6f} | {q1_minus_q5:.6f} | {q1_minus_g0_text} | {monotonic_q1_q3} | {rationale} |".format(
                        signal=row["signal"],
                        verdict=row["verdict"],
                        q1_spec_r995=float(row["q1_spec_r995"]),
                        q5_spec_r995=float(row["q5_spec_r995"]),
                        q1_minus_q5=float(row["q1_minus_q5"]),
                        q1_minus_g0_text="" if row["q1_minus_g0"] is None else f"{float(row['q1_minus_g0']):.6f}",
                        monotonic_q1_q3=str(bool(row["monotonic_q1_q3"])),
                        rationale=row["rationale"],
                    )
                    for row in verdict_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print_step("done", f"wrote bucket pilot assets to {results_dir}")


if __name__ == "__main__":
    main()
