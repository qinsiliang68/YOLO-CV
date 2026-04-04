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
    "accuracy",
    "auroc",
    "auprc",
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
    parser = argparse.ArgumentParser(description="Build the formal cls6 five-model capacity summary.")
    parser.add_argument("--materials-root", required=True, help="Per-model materials root.")
    parser.add_argument("--results-dir", required=True, help="Global formal results directory for cls6.")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float]:
        return (as_float(row["accuracy"]), as_float(row["auroc"]), as_float(row["auprc"]))

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
                "accuracy": best["accuracy"],
                "auroc": best["auroc"],
                "auprc": best["auprc"],
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
        registry_by_key[("cls6", str(row["model"]))] = {
            "task": "cls6",
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


def build_md(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
    lines = [
        "# Formal CLS6 Capacity Summary",
        "",
        "| Model | Best Epoch | Accuracy | AUROC | AUPRC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {model} | {best_epoch} | {accuracy} | {auroc} | {auprc} |".format(**row))
    lines.extend(
        [
            "",
            f"- Best formal cls6 model: `{best_row['model']}`",
            f"- Best epoch: `{best_row['best_epoch']}`",
            "- Ranking rule: `Accuracy -> AUROC -> AUPRC`",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    models = [str(row["model"]) for row in rows]
    x = range(len(models))
    metrics = [("accuracy", "Accuracy", "#355C7D"), ("auroc", "AUROC", "#6C5B7B"), ("auprc", "AUPRC", "#F67280")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, (field, title, color) in zip(axes, metrics, strict=True):
        values = [as_float(row[field]) for row in rows]
        ax.bar(list(x), values, color=color, alpha=0.9)
        ax.set_xticks(list(x), models, rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Formal cls6 capacity scan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    write_csv(results_dir / "cls6_capacity_summary.csv", SUMMARY_FIELDS, rows)
    write_json(
        results_dir / "cls6_capacity_summary.json",
        {
            "ranking_rule": ["Accuracy descending", "AUROC descending", "AUPRC descending"],
            "best_row": best_row,
            "rows": rows,
        },
    )
    (results_dir / "cls6_capacity_summary.md").write_text(build_md(rows, best_row), encoding="utf-8")
    plot_comparison(rows, results_dir / "cls6_capacity_comparison.png")


if __name__ == "__main__":
    main()
