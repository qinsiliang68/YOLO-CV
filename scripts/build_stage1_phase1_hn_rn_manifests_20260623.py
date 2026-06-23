# -*- coding: utf-8 -*-
"""Build phase-1 hard-normal and random-normal replay manifests.

The output is a set of manifest directories that can be passed directly to
scripts/train_stage1_cls_sweep.py with --manifest-dir.

Replay policies:
- append (default): keep the base normal slots and append one duplicate replay
  slot for each selected HN/RN sample. With 100 base normal slots, HN-01 has
  101 normal rows and HN-20 has 120 normal rows.
- fixed: keep the final normal slot count fixed by displacing the same number
  of unselected normal samples as replay duplicates.
- Filenames for replay slots are made unique, while canonical_image_relpath
  still points to the original image.

This script writes only CSV/JSON metadata. It never copies image files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


SEED = 20260606
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OOF_PREDICTIONS = (
    Path("artifacts")
    / "stage1_oof_predictions_calop_20260621"
    / "merged_10fold_20260622"
    / "oof_predictions_merged.csv"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "stage1_phase1_hn_rn_20260623"

TRAIN_MANIFEST = "train_manifest.csv"
NORMAL_TRAIN_MANIFEST = "normal_train_manifest.csv"
VAL_MODEL_MANIFEST = "val_model_manifest.csv"
NORMAL_VAL_MODEL_MANIFEST = "normal_val_model_manifest.csv"

REPLAY_COLUMNS = (
    "replay_run_id",
    "replay_mode",
    "replay_group",
    "replay_q_percent",
    "replay_slot_type",
    "replay_slot_index",
    "replay_source_filename",
    "replay_source_canonical_image_relpath",
    "replay_selected",
    "replay_displaced",
    "oof_human_fold",
    "oof_fold",
    "oof_p_defect_operational",
    "oof_operational_threshold",
    "oof_y_pred_operational",
    "oof_operational_correct",
)

SELECTION_COLUMNS = (
    "run_id",
    "replay_mode",
    "group",
    "q_percent",
    "role",
    "slot_count",
    "source_filename",
    "source_canonical_image_relpath",
    "human_fold",
    "oof_fold",
    "p_defect_operational",
    "operational_threshold",
    "y_pred_operational",
    "operational_correct",
)

RUN_MATRIX_COLUMNS = (
    "run_id",
    "replay_mode",
    "group",
    "q_percent",
    "manifest_dir",
    "normal_slots",
    "defect_slots",
    "selected_unique",
    "replay_duplicate_slots",
    "displaced_unique",
    "kept_unselected",
    "final_normal_rows",
    "final_defect_rows",
    "selected_actual_oof_fp",
    "selection_min_p_defect",
    "selection_max_p_defect",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_key(row: dict[str, str]) -> str:
    key = row.get("canonical_image_relpath", "")
    if not key:
        raise ValueError(f"Missing canonical_image_relpath in row: {row}")
    return key.replace("\\", "/")


def allocate_counts(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError(f"total must be non-negative: {total}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Cannot allocate over empty weights")
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    out = {key: int(value) for key, value in raw.items()}
    missing = total - sum(out.values())
    order = sorted(raw, key=lambda key: (raw[key] - out[key], weights[key], key), reverse=True)
    for key in order[:missing]:
        out[key] += 1
    return out


def sample_rows(rows: list[dict[str, str]], count: int, seed_text: str) -> list[dict[str, str]]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from pool of {len(rows)}")
    rng = random.Random(seed_text)
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    return [rows[index] for index in indexes[:count]]


def group_oof_normals(oof_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in oof_rows:
        if row.get("source_split") != "normal_train":
            continue
        if row.get("y_true") != "0":
            continue
        key = canonical_key(row)
        if key in out:
            raise ValueError(f"Duplicate OOF normal key: {key}")
        out[key] = row
    return out


def rows_by_fold(rows: list[dict[str, str]], oof_by_key: dict[str, dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing = []
    for row in rows:
        key = canonical_key(row)
        oof = oof_by_key.get(key)
        if oof is None:
            missing.append(key)
            continue
        grouped[oof["human_fold"]].append(row)
    if missing:
        raise ValueError(f"Missing OOF rows for {len(missing)} normal_train rows; first={missing[:3]}")
    return dict(grouped)


def sort_hn(rows: list[dict[str, str]], oof_by_key: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            float(oof_by_key[canonical_key(row)]["p_defect_operational"]),
            canonical_key(row),
        ),
        reverse=True,
    )


def select_rows_global(
    normal_rows: list[dict[str, str]],
    oof_by_key: dict[str, dict[str, str]],
    count: int,
    group: str,
    seed: int,
    run_id: str,
) -> list[dict[str, str]]:
    if count == 0:
        return []
    if group == "HN":
        return sort_hn(normal_rows, oof_by_key)[:count]
    if group == "RN":
        return sample_rows(normal_rows, count, f"{seed}:{run_id}:selected:global")
    raise ValueError(f"Unexpected group for selection: {group}")


def fold_counts_for_rows(rows: list[dict[str, str]], oof_by_key: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[oof_by_key[canonical_key(row)]["human_fold"]] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def replay_filename(run_id: str, slot_index: int, source_filename: str) -> str:
    path = Path(source_filename)
    stem = path.stem[:80]
    suffix = path.suffix or ".png"
    return f"replay_{run_id}_{slot_index:06d}__{stem}{suffix}"


def enrich_row(
    row: dict[str, str],
    oof: dict[str, str] | None,
    run_id: str,
    group: str,
    replay_mode: str,
    q_percent: int,
    slot_type: str,
    slot_index: int,
    selected: bool,
    displaced: bool,
    filename_override: str | None = None,
) -> dict[str, str]:
    out = dict(row)
    source_filename = row.get("Filename", "")
    if filename_override is not None:
        out["Filename"] = filename_override
    out.update(
        {
            "replay_run_id": run_id,
            "replay_mode": replay_mode,
            "replay_group": group,
            "replay_q_percent": str(q_percent),
            "replay_slot_type": slot_type,
            "replay_slot_index": str(slot_index),
            "replay_source_filename": source_filename,
            "replay_source_canonical_image_relpath": canonical_key(row),
            "replay_selected": "1" if selected else "0",
            "replay_displaced": "1" if displaced else "0",
            "oof_human_fold": oof.get("human_fold", "") if oof else "",
            "oof_fold": oof.get("oof_fold", "") if oof else "",
            "oof_p_defect_operational": oof.get("p_defect_operational", "") if oof else "",
            "oof_operational_threshold": oof.get("operational_threshold", "") if oof else "",
            "oof_y_pred_operational": oof.get("y_pred_operational", "") if oof else "",
            "oof_operational_correct": oof.get("operational_correct", "") if oof else "",
        }
    )
    return out


def build_selection_manifest_rows(
    run_id: str,
    group: str,
    replay_mode: str,
    q_percent: int,
    role: str,
    rows: list[dict[str, str]],
    oof_by_key: dict[str, dict[str, str]],
    slot_count: int = 1,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        key = canonical_key(row)
        oof = oof_by_key.get(key, {})
        out.append(
            {
                "run_id": run_id,
                "replay_mode": replay_mode,
                "group": group,
                "q_percent": str(q_percent),
                "role": role,
                "slot_count": str(slot_count),
                "source_filename": row.get("Filename", ""),
                "source_canonical_image_relpath": key,
                "human_fold": oof.get("human_fold", ""),
                "oof_fold": oof.get("oof_fold", ""),
                "p_defect_operational": oof.get("p_defect_operational", ""),
                "operational_threshold": oof.get("operational_threshold", ""),
                "y_pred_operational": oof.get("y_pred_operational", ""),
                "operational_correct": oof.get("operational_correct", ""),
            }
        )
    return out


def build_normal_manifest_for_run(
    run_id: str,
    group: str,
    q_percent: int,
    normal_rows: list[dict[str, str]],
    oof_by_key: dict[str, dict[str, str]],
    normal_slots: int,
    seed: int,
    replay_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    if group == "BL":
        selected_total = 0
    else:
        selected_total = normal_slots * q_percent // 100
        if normal_slots * q_percent % 100:
            raise ValueError(f"normal_slots={normal_slots} does not divide q={q_percent}% exactly")
    if selected_total > normal_slots:
        raise ValueError(f"{run_id} selected_total={selected_total} exceeds normal_slots={normal_slots}")
    if replay_mode == "fixed" and 2 * selected_total > normal_slots:
        raise ValueError(f"{run_id} fixed replay needs 2*selected_total <= normal_slots")

    selected_rows_all = select_rows_global(normal_rows, oof_by_key, selected_total, group, seed, run_id)
    selected_keys = {canonical_key(row) for row in selected_rows_all}
    if len(selected_keys) != len(selected_rows_all):
        raise ValueError(f"{run_id} selected duplicate canonical keys")

    unselected_rows = [row for row in normal_rows if canonical_key(row) not in selected_keys]
    if replay_mode == "fixed":
        displaced_count = selected_total
        kept_count = normal_slots - 2 * selected_total
    elif replay_mode == "append":
        displaced_count = 0
        kept_count = normal_slots - selected_total
    else:
        raise ValueError(f"Unknown replay_mode={replay_mode}")

    displaced_rows_all = sample_rows(unselected_rows, displaced_count, f"{seed}:{run_id}:displaced:global")
    displaced_keys = {canonical_key(row) for row in displaced_rows_all}
    kept_pool = [row for row in unselected_rows if canonical_key(row) not in displaced_keys]
    kept_rows_all = sample_rows(kept_pool, kept_count, f"{seed}:{run_id}:kept:global")

    final_rows: list[dict[str, str]] = []
    slot_index = 1

    for row in selected_rows_all:
        oof = oof_by_key[canonical_key(row)]
        final_rows.append(
            enrich_row(row, oof, run_id, group, replay_mode, q_percent, "base_selected", slot_index, True, False)
        )
        slot_index += 1
    for row in selected_rows_all:
        oof = oof_by_key[canonical_key(row)]
        final_rows.append(
            enrich_row(
                row,
                oof,
                run_id,
                group,
                replay_mode,
                q_percent,
                "replay_duplicate",
                slot_index,
                True,
                False,
                filename_override=replay_filename(run_id, slot_index, row.get("Filename", "")),
            )
        )
        slot_index += 1
    for row in kept_rows_all:
        oof = oof_by_key[canonical_key(row)]
        final_rows.append(
            enrich_row(row, oof, run_id, group, replay_mode, q_percent, "base_unselected", slot_index, False, False)
        )
        slot_index += 1

    filenames = [row["Filename"] for row in final_rows]
    if len(filenames) != len(set(filenames)):
        duplicates = [name for name in filenames if filenames.count(name) > 1][:5]
        raise ValueError(f"Duplicate output Filename values in {run_id}: {duplicates}")
    expected_final_rows = normal_slots if replay_mode == "fixed" else normal_slots + len(selected_rows_all)
    if len(final_rows) != expected_final_rows:
        raise ValueError(f"{run_id} produced {len(final_rows)} normal rows, expected {expected_final_rows}")

    selection_rows = []
    selection_rows += build_selection_manifest_rows(
        run_id, group, replay_mode, q_percent, "selected", selected_rows_all, oof_by_key, 2
    )
    selection_rows += build_selection_manifest_rows(
        run_id, group, replay_mode, q_percent, "displaced", displaced_rows_all, oof_by_key, 0
    )
    selection_rows += build_selection_manifest_rows(
        run_id, group, replay_mode, q_percent, "kept_unselected", kept_rows_all, oof_by_key, 1
    )

    selected_scores = [float(oof_by_key[canonical_key(row)]["p_defect_operational"]) for row in selected_rows_all]
    summary = {
        "run_id": run_id,
        "replay_mode": replay_mode,
        "group": group,
        "q_percent": q_percent,
        "normal_slots": normal_slots,
        "selected_unique": len(selected_rows_all),
        "replay_duplicate_slots": len(selected_rows_all),
        "displaced_unique": len(displaced_rows_all),
        "kept_unselected": len(kept_rows_all),
        "final_normal_rows": len(final_rows),
        "selected_actual_oof_fp": sum(
            1 for row in selected_rows_all if oof_by_key[canonical_key(row)].get("y_pred_operational") == "1"
        ),
        "selection_min_p_defect": min(selected_scores) if selected_scores else "",
        "selection_max_p_defect": max(selected_scores) if selected_scores else "",
        "fold_base_counts": fold_counts_for_rows(selected_rows_all + kept_rows_all, oof_by_key),
        "fold_selected_counts": fold_counts_for_rows(selected_rows_all, oof_by_key),
        "fold_replay_counts": fold_counts_for_rows(selected_rows_all, oof_by_key),
    }
    return final_rows, selection_rows, summary


def limit_rows(rows: list[dict[str, str]], count: int | None, seed_text: str) -> list[dict[str, str]]:
    if count is None or count >= len(rows):
        return list(rows)
    return sample_rows(rows, count, seed_text)


def build_run_ids(max_q: int) -> list[tuple[str, str, int]]:
    runs = [("BL-0", "BL", 0)]
    for q in range(1, max_q + 1):
        runs.append((f"HN-{q:02d}", "HN", q))
    for q in range(1, max_q + 1):
        runs.append((f"RN-{q:02d}", "RN", q))
    return runs


def selection_policy(group: str) -> str:
    if group == "BL":
        return "baseline_no_replay"
    if group == "HN":
        return "global_top_oof_p_defect_operational_normal"
    if group == "RN":
        return "global_random_normal_seeded"
    raise ValueError(f"Unknown group: {group}")


def assert_output_root_safe(output_root: Path, force: bool) -> None:
    if force or not output_root.exists():
        return
    protected = (
        "manifests",
        "runs",
        "eval",
        "workdirs",
        "pipeline_summaries",
        "pipeline_logs",
        "run_matrix.csv",
        "build_summary.json",
    )
    existing = [str(output_root / name) for name in protected if (output_root / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing phase root. "
            "Use a new --output-root or pass --force only for an intentional rebuild. "
            f"Existing paths: {existing[:6]}"
        )


def build_manifests(args: argparse.Namespace) -> Path:
    repo_root = repo_root_from_script()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT
    manifest_dir = dataset_root / "manifests"
    oof_path = Path(args.oof_predictions).resolve() if args.oof_predictions else repo_root / DEFAULT_OOF_PREDICTIONS
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root / DEFAULT_OUTPUT_ROOT
    assert_output_root_safe(output_root, args.force)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(manifest_dir / TRAIN_MANIFEST)
    normal_train_rows = read_csv(manifest_dir / NORMAL_TRAIN_MANIFEST)
    val_model_rows = read_csv(manifest_dir / VAL_MODEL_MANIFEST)
    normal_val_model_rows = read_csv(manifest_dir / NORMAL_VAL_MODEL_MANIFEST)
    oof_rows = read_csv(oof_path)

    oof_by_key = group_oof_normals(oof_rows)
    normal_by_fold = rows_by_fold(normal_train_rows, oof_by_key)
    source_files = {
        TRAIN_MANIFEST: manifest_dir / TRAIN_MANIFEST,
        NORMAL_TRAIN_MANIFEST: manifest_dir / NORMAL_TRAIN_MANIFEST,
        VAL_MODEL_MANIFEST: manifest_dir / VAL_MODEL_MANIFEST,
        NORMAL_VAL_MODEL_MANIFEST: manifest_dir / NORMAL_VAL_MODEL_MANIFEST,
        "oof_predictions_merged.csv": oof_path,
    }
    source_hashes = {name: file_sha256(path) for name, path in source_files.items()}

    defect_train_rows = limit_rows(train_rows, args.defect_slots, f"{args.seed}:defect-train")
    val_defect_rows = limit_rows(val_model_rows, args.val_defect_slots, f"{args.seed}:val-defect")
    val_normal_rows = limit_rows(normal_val_model_rows, args.val_normal_slots, f"{args.seed}:val-normal")

    run_matrix = []
    for run_id, group, q_percent in build_run_ids(args.max_q):
        run_dir = output_root / "manifests" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        normal_rows, selection_rows, normal_summary = build_normal_manifest_for_run(
            run_id=run_id,
            group=group,
            q_percent=q_percent,
            normal_rows=normal_train_rows,
            oof_by_key=oof_by_key,
            normal_slots=args.normal_slots,
            seed=args.seed,
            replay_mode=args.replay_mode,
        )

        normal_fields = list(normal_train_rows[0].keys()) + list(REPLAY_COLUMNS)
        write_csv(run_dir / NORMAL_TRAIN_MANIFEST, normal_rows, normal_fields)
        write_csv(run_dir / TRAIN_MANIFEST, defect_train_rows, train_rows[0].keys())
        write_csv(run_dir / VAL_MODEL_MANIFEST, val_defect_rows, val_model_rows[0].keys())
        write_csv(run_dir / NORMAL_VAL_MODEL_MANIFEST, val_normal_rows, normal_val_model_rows[0].keys())
        write_csv(run_dir / "selection_manifest.csv", selection_rows, SELECTION_COLUMNS)
        generated_files = {
            TRAIN_MANIFEST: run_dir / TRAIN_MANIFEST,
            NORMAL_TRAIN_MANIFEST: run_dir / NORMAL_TRAIN_MANIFEST,
            VAL_MODEL_MANIFEST: run_dir / VAL_MODEL_MANIFEST,
            NORMAL_VAL_MODEL_MANIFEST: run_dir / NORMAL_VAL_MODEL_MANIFEST,
            "selection_manifest.csv": run_dir / "selection_manifest.csv",
        }

        summary = {
            **normal_summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "budget_mode": (
                "append_duplicate_selected_keep_base_slots"
                if args.replay_mode == "append"
                else "fixed_slots_duplicate_selected_displace_unselected"
            ),
            "replay_mode": args.replay_mode,
            "selection_policy": selection_policy(group),
            "seed": args.seed,
            "dataset_root": str(dataset_root),
            "oof_predictions": str(oof_path),
            "manifest_dir": str(run_dir),
            "defect_slots": len(defect_train_rows),
            "val_defect_slots": len(val_defect_rows),
            "val_normal_slots": len(val_normal_rows),
            "source_files": {name: str(path) for name, path in source_files.items()},
            "source_hashes": source_hashes,
            "generated_files": {name: str(path) for name, path in generated_files.items()},
            "generated_hashes": {name: file_sha256(path) for name, path in generated_files.items()},
        }
        write_json(run_dir / "selection_summary.json", summary)
        run_matrix.append(
            {
                "run_id": run_id,
                "replay_mode": args.replay_mode,
                "group": group,
                "q_percent": str(q_percent),
                "manifest_dir": str(run_dir),
                "normal_slots": str(args.normal_slots),
                "defect_slots": str(len(defect_train_rows)),
                "selected_unique": str(normal_summary["selected_unique"]),
                "replay_duplicate_slots": str(normal_summary["replay_duplicate_slots"]),
                "displaced_unique": str(normal_summary["displaced_unique"]),
                "kept_unselected": str(normal_summary["kept_unselected"]),
                "final_normal_rows": str(normal_summary["final_normal_rows"]),
                "final_defect_rows": str(len(defect_train_rows)),
                "selected_actual_oof_fp": str(normal_summary["selected_actual_oof_fp"]),
                "selection_min_p_defect": str(normal_summary["selection_min_p_defect"]),
                "selection_max_p_defect": str(normal_summary["selection_max_p_defect"]),
            }
        )

    write_csv(output_root / "run_matrix.csv", run_matrix, RUN_MATRIX_COLUMNS)
    write_json(
        output_root / "build_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "max_q": args.max_q,
            "replay_mode": args.replay_mode,
            "run_count": len(run_matrix),
            "normal_slots": args.normal_slots,
            "defect_slots": len(defect_train_rows),
            "val_defect_slots": len(val_defect_rows),
            "val_normal_slots": len(val_normal_rows),
            "output_root": str(output_root),
            "run_matrix": str(output_root / "run_matrix.csv"),
            "source_files": {name: str(path) for name, path in source_files.items()},
            "source_hashes": source_hashes,
            "args": {
                "max_q": args.max_q,
                "replay_mode": args.replay_mode,
                "normal_slots": args.normal_slots,
                "defect_slots": args.defect_slots,
                "val_defect_slots": args.val_defect_slots,
                "val_normal_slots": args.val_normal_slots,
                "seed": args.seed,
                "force": args.force,
            },
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-1 HN/RN replay manifests.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--oof-predictions", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-q", type=int, default=20)
    parser.add_argument("--replay-mode", choices=("append", "fixed"), default="append")
    parser.add_argument("--normal-slots", type=int, default=60000)
    parser.add_argument("--defect-slots", type=int, default=60000)
    parser.add_argument("--val-defect-slots", type=int, default=None)
    parser.add_argument("--val-normal-slots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Allow overwriting manifests in an existing phase root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_q < 1 or args.max_q > 100:
        raise ValueError("--max-q must be in [1, 100]")
    if args.normal_slots <= 0 or args.defect_slots <= 0:
        raise ValueError("--normal-slots and --defect-slots must be positive")
    if args.normal_slots * args.max_q % 100:
        raise ValueError("--normal-slots must make max-q percent an integer slot count")
    if args.replay_mode == "fixed" and args.max_q > 50:
        raise ValueError("--replay-mode fixed requires --max-q <= 50")
    output_root = build_manifests(args)
    print(f"output_root={output_root}")
    print(f"run_matrix={output_root / 'run_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
