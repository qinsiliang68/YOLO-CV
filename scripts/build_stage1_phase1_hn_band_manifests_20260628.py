# -*- coding: utf-8 -*-
"""Build phase-1 hard-normal band replay manifests.

This is the non-cumulative follow-up to the archived 20260623 HN/RN scan.
It keeps the training/evaluation protocol unchanged, but changes the data-side
question from "top q% cumulative" to "which hard-normal confidence band helps".

Runs:
- HN1-01 ... HN1-20: one-percent bands, [0,1), [1,2), ... [19,20).
- HN2-01 ... HN2-10: two-percent bands, [0,2), [2,4), ... [18,20).
- RN1A-01 ... RN1C-20: three same-size one-percent random controls.
- RN2A-01 ... RN2C-10: three same-size two-percent random controls.

The script writes only CSV/JSON metadata. It never copies image files.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:  # Imported from tests as scripts.*, run from CLI as a sibling module.
    from scripts import build_stage1_phase1_hn_rn_manifests_20260623 as base
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    import build_stage1_phase1_hn_rn_manifests_20260623 as base


SEED = 20260606
DEFAULT_DATASET_ROOT = base.DEFAULT_DATASET_ROOT
DEFAULT_OOF_PREDICTIONS = base.DEFAULT_OOF_PREDICTIONS
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "stage1_phase1_hn_band_20260628"

BAND_SELECTION_COLUMNS = tuple(
    dict.fromkeys(
        list(base.SELECTION_COLUMNS)
        + [
            "band_index",
            "band_start_percent",
            "band_end_percent",
            "band_width_percent",
            "band_rank_start",
            "band_rank_end_exclusive",
            "paired_hn_run_id",
            "control_replicate",
            "selection_seed",
            "selection_policy",
        ]
    )
)

BAND_RUN_MATRIX_COLUMNS = tuple(
    dict.fromkeys(
        list(base.RUN_MATRIX_COLUMNS)
        + [
            "band_index",
            "band_start_percent",
            "band_end_percent",
            "band_width_percent",
            "band_rank_start",
            "band_rank_end_exclusive",
            "paired_hn_run_id",
            "control_replicate",
            "selection_seed",
            "selection_policy",
        ]
    )
)

REPRO_EXPECTED_COLUMNS = (
    "run_id",
    "experiment",
    "fixed_model",
    "seed",
    "replay_mode",
    "group",
    "q_percent",
    "band_start_percent",
    "band_end_percent",
    "band_width_percent",
    "band_rank_start",
    "band_rank_end_exclusive",
    "paired_hn_run_id",
    "control_replicate",
    "selection_seed",
    "selection_policy",
    "normal_slots",
    "defect_slots",
    "selected_unique",
    "selected_actual_oof_fp",
    "manifest_dir",
    "selection_manifest_csv",
    "selection_summary_json",
    "normal_train_queue_csv",
    "defect_train_queue_csv",
    "val_model_defect_queue_csv",
    "val_model_normal_queue_csv",
    "selected_queue_filter",
    "kept_normal_queue_filter",
    "replay_duplicate_filter",
    "oof_predictions_csv",
    "source_train_manifest_csv",
    "source_normal_train_manifest_csv",
    "source_val_model_manifest_csv",
    "source_normal_val_model_manifest_csv",
    "run_matrix_csv",
    "build_summary_json",
)

SELECTED_SAMPLE_INDEX_COLUMNS = tuple(
    dict.fromkeys(
        [
            "run_id",
            "experiment",
            "group",
            "paired_hn_run_id",
            "control_replicate",
            "selection_policy",
            "selection_seed",
            "q_percent",
            "band_index",
            "band_start_percent",
            "band_end_percent",
            "band_width_percent",
            "band_rank_start",
            "band_rank_end_exclusive",
            "role",
            "slot_count",
            "source_filename",
            "source_canonical_image_relpath",
            "source_dataset_split",
            "source_image_path",
            "human_fold",
            "oof_fold",
            "p_defect_operational",
            "operational_threshold",
            "y_pred_operational",
            "operational_correct",
            "source_dataset_root",
            "source_normal_train_manifest_csv",
            "oof_predictions_csv",
            "manifest_dir",
            "selection_manifest_csv",
            "selection_summary_json",
        ]
    )
)


@dataclass(frozen=True)
class BandRun:
    run_id: str
    group: str
    band_index: int
    start_percent: int
    end_percent: int
    paired_hn_run_id: str
    control_replicate: str

    @property
    def width_percent(self) -> int:
        return self.end_percent - self.start_percent

    @property
    def is_random_control(self) -> bool:
        return self.control_replicate != "HN"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_band_runs() -> list[BandRun]:
    runs: list[BandRun] = []
    for index in range(1, 21):
        start = index - 1
        run_id = f"HN1-{index:02d}"
        runs.append(BandRun(run_id, "HN1", index, start, start + 1, run_id, "HN"))
    for index in range(1, 11):
        start = (index - 1) * 2
        run_id = f"HN2-{index:02d}"
        runs.append(BandRun(run_id, "HN2", index, start, start + 2, run_id, "HN"))
    for replicate in ("A", "B", "C"):
        for index in range(1, 21):
            start = index - 1
            runs.append(
                BandRun(
                    f"RN1{replicate}-{index:02d}",
                    "RN1",
                    index,
                    start,
                    start + 1,
                    f"HN1-{index:02d}",
                    replicate,
                )
            )
    for replicate in ("A", "B", "C"):
        for index in range(1, 11):
            start = (index - 1) * 2
            runs.append(
                BandRun(
                    f"RN2{replicate}-{index:02d}",
                    "RN2",
                    index,
                    start,
                    start + 2,
                    f"HN2-{index:02d}",
                    replicate,
                )
            )
    return runs


def rank_bounds(normal_slots: int, band: BandRun) -> tuple[int, int]:
    start_num = normal_slots * band.start_percent
    end_num = normal_slots * band.end_percent
    if start_num % 100 or end_num % 100:
        raise ValueError(
            f"normal_slots={normal_slots} does not make integer rank bounds for "
            f"{band.run_id} [{band.start_percent},{band.end_percent})%"
        )
    return start_num // 100, end_num // 100


def band_selection_policy(band: BandRun) -> str:
    if band.is_random_control:
        return f"global_random_normal_seeded_control_{band.control_replicate}_same_size_as_{band.paired_hn_run_id}"
    return (
        "global_oof_p_defect_operational_normal_band_"
        f"{band.start_percent:02d}_{band.end_percent:02d}_percent"
    )


def selection_seed(band: BandRun, seed: int) -> str:
    if band.is_random_control:
        return f"{seed}:{band.run_id}:selected:global_random_control"
    return f"{seed}:{band.run_id}:oof_rank_band:{band.start_percent}:{band.end_percent}"


def add_band_fields(
    rows: list[dict[str, str]],
    band: BandRun,
    rank_start: int,
    rank_end: int,
    seed: int,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "band_index": str(band.band_index),
                "band_start_percent": str(band.start_percent),
                "band_end_percent": str(band.end_percent),
                "band_width_percent": str(band.width_percent),
                "band_rank_start": str(rank_start),
                "band_rank_end_exclusive": str(rank_end),
                "paired_hn_run_id": band.paired_hn_run_id,
                "control_replicate": band.control_replicate,
                "selection_seed": selection_seed(band, seed),
                "selection_policy": band_selection_policy(band),
            }
        )
        out.append(item)
    return out


def select_rows_for_band(
    band: BandRun,
    normal_rows: list[dict[str, str]],
    sorted_hn_rows: list[dict[str, str]],
    selected_total: int,
    seed: int,
    rank_start: int,
    rank_end: int,
) -> list[dict[str, str]]:
    if band.is_random_control:
        return base.sample_rows(normal_rows, selected_total, selection_seed(band, seed))
    if rank_end > len(sorted_hn_rows):
        raise ValueError(f"{band.run_id} rank_end={rank_end} exceeds sorted normal pool={len(sorted_hn_rows)}")
    return sorted_hn_rows[rank_start:rank_end]


def build_normal_manifest_for_band(
    band: BandRun,
    normal_rows: list[dict[str, str]],
    sorted_hn_rows: list[dict[str, str]],
    oof_by_key: dict[str, dict[str, str]],
    normal_slots: int,
    seed: int,
    replay_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    rank_start, rank_end = rank_bounds(normal_slots, band)
    selected_total = rank_end - rank_start
    if selected_total <= 0:
        raise ValueError(f"{band.run_id} selected_total must be positive")
    if selected_total > normal_slots:
        raise ValueError(f"{band.run_id} selected_total={selected_total} exceeds normal_slots={normal_slots}")
    if replay_mode == "fixed" and 2 * selected_total > normal_slots:
        raise ValueError(f"{band.run_id} fixed replay needs 2*selected_total <= normal_slots")

    selected_rows_all = select_rows_for_band(
        band, normal_rows, sorted_hn_rows, selected_total, seed, rank_start, rank_end
    )
    selected_keys = {base.canonical_key(row) for row in selected_rows_all}
    if len(selected_keys) != len(selected_rows_all):
        raise ValueError(f"{band.run_id} selected duplicate canonical keys")

    unselected_rows = [row for row in normal_rows if base.canonical_key(row) not in selected_keys]
    if replay_mode == "fixed":
        displaced_count = selected_total
        kept_count = normal_slots - 2 * selected_total
    elif replay_mode == "append":
        displaced_count = 0
        kept_count = normal_slots - selected_total
    else:
        raise ValueError(f"Unknown replay_mode={replay_mode}")

    displaced_rows_all = base.sample_rows(unselected_rows, displaced_count, f"{seed}:{band.run_id}:displaced:global")
    displaced_keys = {base.canonical_key(row) for row in displaced_rows_all}
    kept_pool = [row for row in unselected_rows if base.canonical_key(row) not in displaced_keys]
    kept_rows_all = base.sample_rows(kept_pool, kept_count, f"{seed}:{band.run_id}:kept:global")

    final_rows: list[dict[str, str]] = []
    slot_index = 1
    for row in selected_rows_all:
        oof = oof_by_key[base.canonical_key(row)]
        final_rows.append(
            base.enrich_row(
                row,
                oof,
                band.run_id,
                band.group,
                replay_mode,
                band.width_percent,
                "base_selected",
                slot_index,
                True,
                False,
            )
        )
        slot_index += 1
    for row in selected_rows_all:
        oof = oof_by_key[base.canonical_key(row)]
        final_rows.append(
            base.enrich_row(
                row,
                oof,
                band.run_id,
                band.group,
                replay_mode,
                band.width_percent,
                "replay_duplicate",
                slot_index,
                True,
                False,
                filename_override=base.replay_filename(band.run_id, slot_index, row.get("Filename", "")),
            )
        )
        slot_index += 1
    for row in kept_rows_all:
        oof = oof_by_key[base.canonical_key(row)]
        final_rows.append(
            base.enrich_row(
                row,
                oof,
                band.run_id,
                band.group,
                replay_mode,
                band.width_percent,
                "base_unselected",
                slot_index,
                False,
                False,
            )
        )
        slot_index += 1

    filenames = [row["Filename"] for row in final_rows]
    if len(filenames) != len(set(filenames)):
        duplicates = [name for name in filenames if filenames.count(name) > 1][:5]
        raise ValueError(f"Duplicate output Filename values in {band.run_id}: {duplicates}")
    expected_final_rows = normal_slots if replay_mode == "fixed" else normal_slots + len(selected_rows_all)
    if len(final_rows) != expected_final_rows:
        raise ValueError(f"{band.run_id} produced {len(final_rows)} normal rows, expected {expected_final_rows}")

    selection_rows = []
    selection_rows += base.build_selection_manifest_rows(
        band.run_id, band.group, replay_mode, band.width_percent, "selected", selected_rows_all, oof_by_key, 2
    )
    selection_rows += base.build_selection_manifest_rows(
        band.run_id, band.group, replay_mode, band.width_percent, "displaced", displaced_rows_all, oof_by_key, 0
    )
    selection_rows += base.build_selection_manifest_rows(
        band.run_id, band.group, replay_mode, band.width_percent, "kept_unselected", kept_rows_all, oof_by_key, 1
    )
    selection_rows = add_band_fields(selection_rows, band, rank_start, rank_end, seed)

    selected_scores = [float(oof_by_key[base.canonical_key(row)]["p_defect_operational"]) for row in selected_rows_all]
    summary = {
        "run_id": band.run_id,
        "replay_mode": replay_mode,
        "group": band.group,
        "q_percent": band.width_percent,
        "paired_hn_run_id": band.paired_hn_run_id,
        "control_replicate": band.control_replicate,
        "band_index": band.band_index,
        "band_start_percent": band.start_percent,
        "band_end_percent": band.end_percent,
        "band_width_percent": band.width_percent,
        "band_rank_start": rank_start,
        "band_rank_end_exclusive": rank_end,
        "selection_seed": selection_seed(band, seed),
        "normal_slots": normal_slots,
        "selected_unique": len(selected_rows_all),
        "replay_duplicate_slots": len(selected_rows_all),
        "displaced_unique": len(displaced_rows_all),
        "kept_unselected": len(kept_rows_all),
        "final_normal_rows": len(final_rows),
        "selected_actual_oof_fp": sum(
            1 for row in selected_rows_all if oof_by_key[base.canonical_key(row)].get("y_pred_operational") == "1"
        ),
        "selection_min_p_defect": min(selected_scores),
        "selection_max_p_defect": max(selected_scores),
        "fold_base_counts": base.fold_counts_for_rows(selected_rows_all + kept_rows_all, oof_by_key),
        "fold_selected_counts": base.fold_counts_for_rows(selected_rows_all, oof_by_key),
        "fold_replay_counts": base.fold_counts_for_rows(selected_rows_all, oof_by_key),
    }
    return final_rows, selection_rows, summary


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
        "validation",
        "run_matrix.csv",
        "build_summary.json",
    )
    existing = [str(output_root / name) for name in protected if (output_root / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing HN band phase root. "
            "Use a new --output-root or pass --force only for an intentional rebuild. "
            f"Existing paths: {existing[:6]}"
        )


def build_expected_repro_rows(
    run_matrix: list[dict[str, str]],
    output_root: Path,
    source_files: dict[str, Path],
    oof_path: Path,
    seed: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in run_matrix:
        run_id = row["run_id"]
        run_dir = output_root / "manifests" / run_id
        out.append(
            {
                "run_id": run_id,
                "experiment": "stage1_phase1_hn_band_20260628",
                "fixed_model": "yolo11l-cls",
                "seed": str(seed),
                "replay_mode": row["replay_mode"],
                "group": row["group"],
                "q_percent": row["q_percent"],
                "band_start_percent": row["band_start_percent"],
                "band_end_percent": row["band_end_percent"],
                "band_width_percent": row["band_width_percent"],
                "band_rank_start": row["band_rank_start"],
                "band_rank_end_exclusive": row["band_rank_end_exclusive"],
                "paired_hn_run_id": row["paired_hn_run_id"],
                "control_replicate": row["control_replicate"],
                "selection_seed": row["selection_seed"],
                "selection_policy": row["selection_policy"],
                "normal_slots": row["normal_slots"],
                "defect_slots": row["defect_slots"],
                "selected_unique": row["selected_unique"],
                "selected_actual_oof_fp": row["selected_actual_oof_fp"],
                "manifest_dir": str(run_dir),
                "selection_manifest_csv": str(run_dir / "selection_manifest.csv"),
                "selection_summary_json": str(run_dir / "selection_summary.json"),
                "normal_train_queue_csv": str(run_dir / base.NORMAL_TRAIN_MANIFEST),
                "defect_train_queue_csv": str(run_dir / base.TRAIN_MANIFEST),
                "val_model_defect_queue_csv": str(run_dir / base.VAL_MODEL_MANIFEST),
                "val_model_normal_queue_csv": str(run_dir / base.NORMAL_VAL_MODEL_MANIFEST),
                "selected_queue_filter": "selection_manifest.csv where role=selected",
                "kept_normal_queue_filter": "selection_manifest.csv where role=kept_unselected",
                "replay_duplicate_filter": "normal_train_manifest.csv where replay_slot_type=replay_duplicate",
                "oof_predictions_csv": str(oof_path),
                "source_train_manifest_csv": str(source_files[base.TRAIN_MANIFEST]),
                "source_normal_train_manifest_csv": str(source_files[base.NORMAL_TRAIN_MANIFEST]),
                "source_val_model_manifest_csv": str(source_files[base.VAL_MODEL_MANIFEST]),
                "source_normal_val_model_manifest_csv": str(source_files[base.NORMAL_VAL_MODEL_MANIFEST]),
                "run_matrix_csv": str(output_root / "run_matrix.csv"),
                "build_summary_json": str(output_root / "build_summary.json"),
            }
        )
    return out


def build_selected_sample_index_rows(
    selection_rows: list[dict[str, str]],
    dataset_root: Path,
    source_files: dict[str, Path],
    oof_path: Path,
    run_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in selection_rows:
        if row.get("role") != "selected":
            continue
        item = dict(row)
        canonical_path = row.get("source_canonical_image_relpath", "").replace("\\", "/")
        item.update(
            {
                "experiment": "stage1_phase1_hn_band_20260628",
                "source_dataset_split": "normal_train",
                "source_image_path": str(dataset_root / Path(canonical_path)) if canonical_path else "",
                "source_dataset_root": str(dataset_root),
                "source_normal_train_manifest_csv": str(source_files[base.NORMAL_TRAIN_MANIFEST]),
                "oof_predictions_csv": str(oof_path),
                "manifest_dir": str(run_dir),
                "selection_manifest_csv": str(run_dir / "selection_manifest.csv"),
                "selection_summary_json": str(run_dir / "selection_summary.json"),
            }
        )
        rows.append(item)
    return rows


def build_manifests(args: argparse.Namespace) -> Path:
    repo_root = repo_root_from_script()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT
    manifest_dir = dataset_root / "manifests"
    oof_path = Path(args.oof_predictions).resolve() if args.oof_predictions else repo_root / DEFAULT_OOF_PREDICTIONS
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root / DEFAULT_OUTPUT_ROOT
    assert_output_root_safe(output_root, args.force)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = base.read_csv(manifest_dir / base.TRAIN_MANIFEST)
    normal_train_rows = base.read_csv(manifest_dir / base.NORMAL_TRAIN_MANIFEST)
    val_model_rows = base.read_csv(manifest_dir / base.VAL_MODEL_MANIFEST)
    normal_val_model_rows = base.read_csv(manifest_dir / base.NORMAL_VAL_MODEL_MANIFEST)
    oof_rows = base.read_csv(oof_path)

    oof_by_key = base.group_oof_normals(oof_rows)
    _ = base.rows_by_fold(normal_train_rows, oof_by_key)
    sorted_hn_rows = base.sort_hn(normal_train_rows, oof_by_key)
    source_files = {
        base.TRAIN_MANIFEST: manifest_dir / base.TRAIN_MANIFEST,
        base.NORMAL_TRAIN_MANIFEST: manifest_dir / base.NORMAL_TRAIN_MANIFEST,
        base.VAL_MODEL_MANIFEST: manifest_dir / base.VAL_MODEL_MANIFEST,
        base.NORMAL_VAL_MODEL_MANIFEST: manifest_dir / base.NORMAL_VAL_MODEL_MANIFEST,
        "oof_predictions_merged.csv": oof_path,
    }
    source_hashes = {name: base.file_sha256(path) for name, path in source_files.items()}

    defect_train_rows = base.limit_rows(train_rows, args.defect_slots, f"{args.seed}:defect-train")
    val_defect_rows = base.limit_rows(val_model_rows, args.val_defect_slots, f"{args.seed}:val-defect")
    val_normal_rows = base.limit_rows(normal_val_model_rows, args.val_normal_slots, f"{args.seed}:val-normal")

    run_matrix = []
    selected_sample_index_rows = []
    for band in build_band_runs():
        rank_start, rank_end = rank_bounds(args.normal_slots, band)
        run_dir = output_root / "manifests" / band.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        normal_rows, selection_rows, normal_summary = build_normal_manifest_for_band(
            band=band,
            normal_rows=normal_train_rows,
            sorted_hn_rows=sorted_hn_rows,
            oof_by_key=oof_by_key,
            normal_slots=args.normal_slots,
            seed=args.seed,
            replay_mode=args.replay_mode,
        )

        normal_fields = list(normal_train_rows[0].keys()) + list(base.REPLAY_COLUMNS)
        write_csv(run_dir / base.NORMAL_TRAIN_MANIFEST, normal_rows, normal_fields)
        write_csv(run_dir / base.TRAIN_MANIFEST, defect_train_rows, train_rows[0].keys())
        write_csv(run_dir / base.VAL_MODEL_MANIFEST, val_defect_rows, val_model_rows[0].keys())
        write_csv(run_dir / base.NORMAL_VAL_MODEL_MANIFEST, val_normal_rows, normal_val_model_rows[0].keys())
        write_csv(run_dir / "selection_manifest.csv", selection_rows, BAND_SELECTION_COLUMNS)
        selected_sample_index_rows.extend(
            build_selected_sample_index_rows(selection_rows, dataset_root, source_files, oof_path, run_dir)
        )
        generated_files = {
            base.TRAIN_MANIFEST: run_dir / base.TRAIN_MANIFEST,
            base.NORMAL_TRAIN_MANIFEST: run_dir / base.NORMAL_TRAIN_MANIFEST,
            base.VAL_MODEL_MANIFEST: run_dir / base.VAL_MODEL_MANIFEST,
            base.NORMAL_VAL_MODEL_MANIFEST: run_dir / base.NORMAL_VAL_MODEL_MANIFEST,
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
            "selection_policy": band_selection_policy(band),
            "selection_seed": selection_seed(band, args.seed),
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
            "generated_hashes": {name: base.file_sha256(path) for name, path in generated_files.items()},
        }
        write_json(run_dir / "selection_summary.json", summary)
        run_matrix.append(
            {
                "run_id": band.run_id,
                "replay_mode": args.replay_mode,
                "group": band.group,
                "q_percent": str(band.width_percent),
                "paired_hn_run_id": band.paired_hn_run_id,
                "control_replicate": band.control_replicate,
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
                "band_index": str(band.band_index),
                "band_start_percent": str(band.start_percent),
                "band_end_percent": str(band.end_percent),
                "band_width_percent": str(band.width_percent),
                "band_rank_start": str(rank_start),
                "band_rank_end_exclusive": str(rank_end),
                "selection_seed": selection_seed(band, args.seed),
                "selection_policy": band_selection_policy(band),
            }
        )

    write_csv(output_root / "run_matrix.csv", run_matrix, BAND_RUN_MATRIX_COLUMNS)
    write_csv(output_root / "selected_samples_index.csv", selected_sample_index_rows, SELECTED_SAMPLE_INDEX_COLUMNS)
    write_csv(
        output_root / "repro_manifest_expected.csv",
        build_expected_repro_rows(run_matrix, output_root, source_files, oof_path, args.seed),
        REPRO_EXPECTED_COLUMNS,
    )
    write_json(
        output_root / "build_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "stage1_phase1_hn_band_20260628",
            "run_count": len(run_matrix),
            "run_ids": [row["run_id"] for row in run_matrix],
            "hn_run_count": 30,
            "random_control_replicates": ["A", "B", "C"],
            "random_control_run_count": 90,
            "replay_mode": args.replay_mode,
            "normal_slots": args.normal_slots,
            "defect_slots": len(defect_train_rows),
            "val_defect_slots": len(val_defect_rows),
            "val_normal_slots": len(val_normal_rows),
            "output_root": str(output_root),
            "run_matrix": str(output_root / "run_matrix.csv"),
            "selected_samples_index": str(output_root / "selected_samples_index.csv"),
            "repro_manifest_expected": str(output_root / "repro_manifest_expected.csv"),
            "source_files": {name: str(path) for name, path in source_files.items()},
            "source_hashes": source_hashes,
            "args": {
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
    parser = argparse.ArgumentParser(description="Build phase-1 HN band replay manifests.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--oof-predictions", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--replay-mode", choices=("append", "fixed"), default="append")
    parser.add_argument("--normal-slots", type=int, default=60000)
    parser.add_argument("--defect-slots", type=int, default=60000)
    parser.add_argument("--val-defect-slots", type=int, default=None)
    parser.add_argument("--val-normal-slots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    output_root = build_manifests(parse_args())
    print(f"wrote_hn_band_manifests={output_root}")
    print("run_count=120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
