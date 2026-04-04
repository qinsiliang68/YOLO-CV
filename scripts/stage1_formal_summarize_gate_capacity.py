from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SUMMARY_FIELDS = [
    "model",
    "run_name",
    "best_epoch",
    "best_checkpoint_path",
    "spec_at_r995",
    "spec_at_r990",
    "prec_at_r990",
    "ptr_at_r990",
    "tau_r995",
    "tau_r990",
    "temperature_T",
    "tn_at_r995",
    "fn_at_r995",
    "tn_at_r990",
    "fn_at_r990",
    "run_dir",
    "summary_dir",
    "commit_hash",
    "machine_name",
    "dataset_manifest_path",
]

REGISTRY_FIELDS = [
    "task",
    "model",
    "run_dir",
    "summary_dir",
    "best_epoch",
    "best_checkpoint_path",
    "commit_hash",
    "machine_name",
    "dataset_manifest_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the formal binary gate five-model capacity summary.")
    parser.add_argument("--materials-root", required=True, help="Per-model materials root.")
    parser.add_argument("--results-dir", required=True, help="Global formal results directory for binary gate.")
    parser.add_argument("--registry-csv", required=True, help="Formal capacity registry CSV.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            as_float(row["spec_at_r995"]),
            as_float(row["spec_at_r990"]),
            as_float(row["prec_at_r990"]),
            -as_float(row["ptr_at_r990"]),
        )

    return max(rows, key=key_fn)


def infer_model_name(run_name: str) -> str:
    token = run_name.split("_")[0]
    return f"{token}-cls" if token.startswith("yolo11") else run_name


def collect_rows(materials_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        best_path = summary_dir / "best_epoch_manifest.json"
        run_manifest_path = summary_dir / "run_manifest.json"
        dataset_manifest_path = summary_dir / "dataset_manifest.json"
        if not best_path.exists() or not run_manifest_path.exists():
            continue
        best = load_json(best_path)
        run_manifest = load_json(run_manifest_path)
        run_name = Path(run_manifest.get("run_dir", summary_dir.name)).name
        rows.append(
            {
                "model": infer_model_name(run_name),
                "run_name": run_name,
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_path": best["checkpoint_path"],
                "spec_at_r995": best["spec_at_r995"],
                "spec_at_r990": best["spec_at_r990"],
                "prec_at_r990": best["prec_at_r990"],
                "ptr_at_r990": best["ptr_at_r990"],
                "tau_r995": best["tau_r995"],
                "tau_r990": best["tau_r990"],
                "temperature_T": best["temperature_T"],
                "tn_at_r995": best["tn_at_r995"],
                "fn_at_r995": best["fn_at_r995"],
                "tn_at_r990": best["tn_at_r990"],
                "fn_at_r990": best["fn_at_r990"],
                "run_dir": run_manifest.get("run_dir", ""),
                "summary_dir": str(summary_dir.resolve()),
                "commit_hash": run_manifest.get("commit_hash", ""),
                "machine_name": run_manifest.get("machine_name", ""),
                "dataset_manifest_path": str(dataset_manifest_path.resolve()) if dataset_manifest_path.exists() else "",
            }
        )
    return rows


def update_registry(registry_path: Path, rows: list[dict[str, Any]]) -> None:
    registry_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                registry_by_key[(str(row.get("task", "")), str(row.get("model", "")))] = row
    for row in rows:
        registry_by_key[("gate", str(row["model"]))] = {
            "task": "gate",
            "model": row["model"],
            "run_dir": row["run_dir"],
            "summary_dir": row["summary_dir"],
            "best_epoch": row["best_epoch"],
            "best_checkpoint_path": row["best_checkpoint_path"],
            "commit_hash": row["commit_hash"],
            "machine_name": row["machine_name"],
            "dataset_manifest_path": row["dataset_manifest_path"],
        }
    merged_rows = sorted(registry_by_key.values(), key=lambda item: (item["task"], item["model"]))
    write_csv(registry_path, REGISTRY_FIELDS, merged_rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_md(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
    lines = [
        "# Formal Binary Gate Capacity Summary",
        "",
        "| Model | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {best_epoch} | {spec_at_r995} | {spec_at_r990} | {prec_at_r990} | {ptr_at_r990} | {tau_r995} | {tau_r990} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Best formal gate model: `{best_row['model']}`",
            f"- Best epoch: `{best_row['best_epoch']}`",
            "- Ranking rule: `Spec@R99.5 -> Spec@R99.0 -> Prec@R99.0 -> PTR@R99.0 ascending`",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    models = [str(row["model"]) for row in rows]
    x = range(len(models))
    metrics = [
        ("spec_at_r995", "Spec@R99.5", "#355C7D"),
        ("spec_at_r990", "Spec@R99.0", "#6C5B7B"),
        ("prec_at_r990", "Prec@R99.0", "#F67280"),
        ("ptr_at_r990", "PTR@R99.0", "#C06C84"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for ax, (field, title, color) in zip(axes.flatten(), metrics, strict=True):
        values = [as_float(row[field]) for row in rows]
        ax.bar(list(x), values, color=color, alpha=0.9)
        ax.set_xticks(list(x), models, rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Formal binary gate capacity scan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    materials_root = Path(args.materials_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    registry_csv = Path(args.registry_csv).resolve()
    rows = collect_rows(materials_root)
    if not rows:
        return
    rows = sorted(rows, key=lambda item: item["model"])
    best_row = choose_best(rows)
    update_registry(registry_csv, rows)
    write_csv(results_dir / "binary_gate_capacity_summary.csv", SUMMARY_FIELDS, rows)
    write_json(
        results_dir / "binary_gate_capacity_summary.json",
        {
            "ranking_rule": [
                "Spec@R99.5 descending",
                "Spec@R99.0 descending",
                "Prec@R99.0 descending",
                "PTR@R99.0 ascending",
            ],
            "best_row": best_row,
            "rows": rows,
        },
    )
    (results_dir / "binary_gate_capacity_summary.md").write_text(build_md(rows, best_row), encoding="utf-8")
    plot_comparison(rows, results_dir / "binary_gate_capacity_comparison.png")


if __name__ == "__main__":
    main()
