from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from scripts.stage1_sample_value_experiment_layout_20260708 import (
    ACTIVE_EXPERIMENT_ID,
    BUDGETS,
    METHOD_IDS,
    experiment_root,
    family_root,
    initialize_experiment_family_layout,
    method_budget_dir,
    run_dir,
    stage_root,
)


DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"

TRAIN_MANIFEST = "train_manifest.csv"
NORMAL_TRAIN_MANIFEST = "normal_train_manifest.csv"
VAL_MODEL_MANIFEST = "val_model_manifest.csv"
NORMAL_VAL_MODEL_MANIFEST = "normal_val_model_manifest.csv"

REPLAY_COLUMNS = (
    "replay_run_id",
    "replay_method_id",
    "replay_budget",
    "replay_slot_index",
    "replay_source_filename",
    "replay_source_canonical_image_relpath",
    "replay_selected",
    "sample_value_score",
)

SELECTION_COLUMNS = (
    "run_id",
    "method_id",
    "budget",
    "selection_rank",
    "sample_id",
    "y_true",
    "sample_value_score",
    "score_column",
    "source_filename",
    "source_canonical_image_relpath",
    "replay_filename",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: Path | None, default: Path) -> Path:
    chosen = path if path is not None else default
    return chosen if chosen.is_absolute() else root / chosen


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def canonical_key(row: dict[str, str]) -> str:
    for key in ("canonical_image_relpath", "sample_id", "image_path", "Filename"):
        value = row.get(key, "")
        if value:
            return value.replace("\\", "/")
    raise ValueError(f"Unable to build canonical key from row: {row}")


def method_score_column(method_id: str) -> str:
    if method_id not in METHOD_IDS:
        raise ValueError(f"Unknown method_id: {method_id}")
    return f"{method_id}_score"


def method_target_label(method_id: str) -> str | None:
    if method_id.endswith("_guard"):
        return None
    return "0"


def score_value(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"Missing score column {column} for sample {canonical_key(row)}")
    return float(value)


def select_value_rows(
    value_rows: list[dict[str, str]],
    method_id: str,
    budget: int,
    *,
    defect_guard_budget: int | None = None,
) -> list[dict[str, str]]:
    score_column = method_score_column(method_id)
    if method_id.endswith("_guard"):
        if defect_guard_budget is None:
            raise ValueError(
                f"{method_id} requires an explicit defect guard budget before guard replay manifests can be built."
            )
        normal_candidates = [row for row in value_rows if row.get("y_true") == "0" and row.get(score_column, "") != ""]
        defect_candidates = [row for row in value_rows if row.get("y_true") == "1" and row.get(score_column, "") != ""]
        if budget > len(normal_candidates):
            raise ValueError(f"{method_id} normal budget={budget} exceeds candidate pool size={len(normal_candidates)}")
        if defect_guard_budget > len(defect_candidates):
            raise ValueError(
                f"{method_id} defect guard budget={defect_guard_budget} exceeds candidate pool size={len(defect_candidates)}"
            )
        normal_candidates.sort(key=lambda row: (-score_value(row, score_column), canonical_key(row)))
        defect_candidates.sort(key=lambda row: (-score_value(row, score_column), canonical_key(row)))
        selected = [*normal_candidates[:budget], *defect_candidates[:defect_guard_budget]]
        keys = [canonical_key(row) for row in selected]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{method_id} selected duplicate sample keys")
        return selected

    target_label = method_target_label(method_id)
    candidates = [
        row
        for row in value_rows
        if (target_label is None or row.get("y_true") == target_label) and row.get(score_column, "") != ""
    ]
    if budget > len(candidates):
        raise ValueError(f"{method_id} budget={budget} exceeds candidate pool size={len(candidates)}")
    candidates.sort(key=lambda row: (-score_value(row, score_column), canonical_key(row)))
    selected = candidates[:budget]
    keys = [canonical_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{method_id} selected duplicate sample keys")
    return selected


def index_rows(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        key = canonical_key(row)
        if key in output:
            raise ValueError(f"Duplicate {label} manifest key: {key}")
        output[key] = row
    return output


def replay_filename(run_id: str, slot_index: int, source_filename: str) -> str:
    return f"replay__{run_id}__{slot_index:05d}__{Path(source_filename).name}"


def make_replay_row(
    source_row: dict[str, str],
    value_row: dict[str, str],
    *,
    method_id: str,
    budget: int,
    run_id: str,
    slot_index: int,
) -> dict[str, str]:
    score = score_value(value_row, method_score_column(method_id))
    output = dict(source_row)
    output.update(
        {
            "Filename": replay_filename(run_id, slot_index, source_row.get("Filename", f"sample_{slot_index}.png")),
            "replay_run_id": run_id,
            "replay_method_id": method_id,
            "replay_budget": str(budget),
            "replay_slot_index": str(slot_index),
            "replay_source_filename": source_row.get("Filename", ""),
            "replay_source_canonical_image_relpath": canonical_key(source_row),
            "replay_selected": "1",
            "sample_value_score": f"{score:.10f}",
        }
    )
    return output


def assert_output_safe(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    existing_files = [item for item in path.rglob("*") if item.is_file()]
    if existing_files and not force:
        raise FileExistsError(f"Output run directory already has files: {path}. Use --force to overwrite manifests.")


def build_replay_manifest_for_run(
    *,
    dataset_root: Path,
    value_rows: list[dict[str, str]],
    method_id: str,
    budget: int,
    run_id: str,
    run_root: Path,
    force: bool,
    defect_guard_budget: int | None = None,
) -> dict[str, object]:
    manifest_root = dataset_root / "manifests"
    train_rows = read_csv(manifest_root / TRAIN_MANIFEST)
    normal_rows = read_csv(manifest_root / NORMAL_TRAIN_MANIFEST)
    val_rows = read_csv(manifest_root / VAL_MODEL_MANIFEST)
    normal_val_rows = read_csv(manifest_root / NORMAL_VAL_MODEL_MANIFEST)
    train_by_key = index_rows(train_rows, "defect train")
    normal_by_key = index_rows(normal_rows, "normal train")

    selected_rows = select_value_rows(value_rows, method_id, budget, defect_guard_budget=defect_guard_budget)
    selected_normal: list[tuple[dict[str, str], dict[str, str]]] = []
    selected_defect: list[tuple[dict[str, str], dict[str, str]]] = []
    for row in selected_rows:
        key = canonical_key(row)
        y_true = row.get("y_true", "")
        if y_true == "0":
            if key not in normal_by_key:
                raise KeyError(f"Selected normal sample not found in normal_train_manifest.csv: {key}")
            selected_normal.append((normal_by_key[key], row))
        elif y_true == "1":
            if key not in train_by_key:
                raise KeyError(f"Selected defect sample not found in train_manifest.csv: {key}")
            selected_defect.append((train_by_key[key], row))
        else:
            raise ValueError(f"Selected row has invalid y_true={y_true!r}: {key}")

    output_run_dir = run_root / run_id
    assert_output_safe(output_run_dir, force=force)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    normal_replays = [
        make_replay_row(source, value, method_id=method_id, budget=budget, run_id=run_id, slot_index=index)
        for index, (source, value) in enumerate(selected_normal, start=1)
    ]
    defect_replays = [
        make_replay_row(
            source,
            value,
            method_id=method_id,
            budget=budget,
            run_id=run_id,
            slot_index=len(normal_replays) + index,
        )
        for index, (source, value) in enumerate(selected_defect, start=1)
    ]

    write_csv(output_run_dir / NORMAL_TRAIN_MANIFEST, [*normal_rows, *normal_replays], [*normal_rows[0].keys(), *REPLAY_COLUMNS])
    write_csv(output_run_dir / TRAIN_MANIFEST, [*train_rows, *defect_replays], [*train_rows[0].keys(), *REPLAY_COLUMNS])
    write_csv(output_run_dir / VAL_MODEL_MANIFEST, val_rows, val_rows[0].keys())
    write_csv(output_run_dir / NORMAL_VAL_MODEL_MANIFEST, normal_val_rows, normal_val_rows[0].keys())

    selection_manifest: list[dict[str, str]] = []
    score_column = method_score_column(method_id)
    replay_rows_by_key = {canonical_key(row): row for row in [*normal_replays, *defect_replays]}
    for rank, row in enumerate(selected_rows, start=1):
        key = canonical_key(row)
        replay_row = replay_rows_by_key[key]
        selection_manifest.append(
            {
                "run_id": run_id,
                "method_id": method_id,
                "budget": str(budget),
                "selection_rank": str(rank),
                "sample_id": key,
                "y_true": row["y_true"],
                "sample_value_score": f"{score_value(row, score_column):.10f}",
                "score_column": score_column,
                "source_filename": replay_row.get("replay_source_filename", ""),
                "source_canonical_image_relpath": replay_row.get("replay_source_canonical_image_relpath", ""),
                "replay_filename": replay_row.get("Filename", ""),
            }
        )
    write_csv(output_run_dir / "selection_manifest.csv", selection_manifest, SELECTION_COLUMNS)

    summary: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": ACTIVE_EXPERIMENT_ID,
        "method_id": method_id,
        "budget": budget,
        "run_id": run_id,
        "run_manifest_dir": str(output_run_dir),
        "score_column": score_column,
        "selected_total": len(selected_rows),
        "selected_normal": len(selected_normal),
        "selected_defect": len(selected_defect),
        "defect_guard_budget": defect_guard_budget if defect_guard_budget is not None else "",
        "base_normal_rows": len(normal_rows),
        "base_defect_rows": len(train_rows),
        "final_normal_rows": len(normal_rows) + len(normal_replays),
        "final_defect_rows": len(train_rows) + len(defect_replays),
    }
    write_json(output_run_dir / "selection_summary.json", summary)
    return summary


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_budget_list(value: str) -> list[int]:
    budgets = [int(item) for item in parse_csv_list(value)]
    for budget in budgets:
        if budget <= 0:
            raise ValueError(f"Budget must be positive: {budget}")
    return budgets


def build_all(args: argparse.Namespace) -> Path:
    root = repo_root()
    fam_root = family_root(repo=root, override=args.family_root)
    initialize_experiment_family_layout(fam_root)
    active_root = experiment_root(fam_root, ACTIVE_EXPERIMENT_ID)
    dataset_root = resolve_path(root, args.dataset_root, DEFAULT_DATASET_ROOT)
    value_table = resolve_path(root, args.sample_value_table, stage_root(active_root, "02_sample_value_tables") / "sample_value_table.csv")
    value_rows = read_csv(value_table)
    default_methods = tuple(method_id for method_id in METHOD_IDS if not method_id.endswith("_guard"))
    methods = tuple(parse_csv_list(args.methods)) if args.methods else default_methods
    budgets = tuple(parse_budget_list(args.budgets)) if args.budgets else BUDGETS

    run_summaries: list[dict[str, object]] = []
    for method_id in methods:
        if method_id not in METHOD_IDS:
            raise ValueError(f"Unknown method_id: {method_id}")
        for budget in budgets:
            run_id = f"{method_id}_b{budget:05d}_r001"
            root_for_budget = method_budget_dir(active_root, "03_replay_manifests", method_id, budget)
            summary = build_replay_manifest_for_run(
                dataset_root=dataset_root,
                value_rows=value_rows,
                method_id=method_id,
                budget=budget,
                run_id=run_id,
                run_root=root_for_budget,
                force=args.force,
                defect_guard_budget=args.defect_guard_budget,
            )
            run_summaries.append(summary)

    write_csv(
        stage_root(active_root, "03_replay_manifests") / "build_run_matrix.csv",
        ({key: str(value) for key, value in summary.items()} for summary in run_summaries),
        (
            "experiment_id",
            "method_id",
            "budget",
            "run_id",
            "run_manifest_dir",
            "score_column",
            "selected_total",
            "selected_normal",
            "selected_defect",
            "defect_guard_budget",
            "base_normal_rows",
            "base_defect_rows",
            "final_normal_rows",
            "final_defect_rows",
        ),
    )
    write_json(
        stage_root(active_root, "03_replay_manifests") / "build_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": ACTIVE_EXPERIMENT_ID,
            "sample_value_table": str(value_table),
            "method_count": len(methods),
            "budget_count": len(budgets),
            "run_count": len(run_summaries),
        },
    )
    return stage_root(active_root, "03_replay_manifests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped replay manifests from a Stage-1 sample-value table.")
    parser.add_argument("--family-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--sample-value-table", type=Path, default=None)
    parser.add_argument("--methods", default="", help="Comma-separated method ids. Defaults to all methods.")
    parser.add_argument("--budgets", default="", help="Comma-separated budgets. Defaults to 600,3000,6000.")
    parser.add_argument("--defect-guard-budget", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        out = build_all(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote_replay_manifests={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
