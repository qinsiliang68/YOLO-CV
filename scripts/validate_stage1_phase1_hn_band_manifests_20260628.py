# -*- coding: utf-8 -*-
"""Validate phase-1 HN band replay manifests and prepared image directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from scripts import build_stage1_phase1_hn_band_manifests_20260628 as band_builder
    from scripts import build_stage1_phase1_hn_rn_manifests_20260623 as base
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    import build_stage1_phase1_hn_band_manifests_20260628 as band_builder
    import build_stage1_phase1_hn_rn_manifests_20260623 as base


DEFAULT_PHASE_ROOT = band_builder.DEFAULT_OUTPUT_ROOT
DEFAULT_DATASET_ROOT = base.DEFAULT_DATASET_ROOT
DEFAULT_OOF_PREDICTIONS = base.DEFAULT_OOF_PREDICTIONS
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_HASH_FILES = (
    "train_manifest.csv",
    "normal_train_manifest.csv",
    "val_model_manifest.csv",
    "normal_val_model_manifest.csv",
    "oof_predictions_merged.csv",
)
REQUIRED_MANIFESTS = (
    "train_manifest.csv",
    "normal_train_manifest.csv",
    "val_model_manifest.csv",
    "normal_val_model_manifest.csv",
    "selection_manifest.csv",
    "selection_summary.json",
)
RUN_MATRIX_REQUIRED_FIELDS = (
    "run_id",
    "replay_mode",
    "group",
    "q_percent",
    "normal_slots",
    "defect_slots",
    "selected_unique",
    "replay_duplicate_slots",
    "displaced_unique",
    "kept_unselected",
    "final_normal_rows",
    "final_defect_rows",
    "selected_actual_oof_fp",
    "band_index",
    "band_start_percent",
    "band_end_percent",
    "band_width_percent",
    "band_rank_start",
    "band_rank_end_exclusive",
    "selection_policy",
)
NORMAL_REPLAY_FIELDS = (
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
SELECTION_REQUIRED_FIELDS = tuple(
    dict.fromkeys(
        list(base.SELECTION_COLUMNS)
        + [
            "band_index",
            "band_start_percent",
            "band_end_percent",
            "band_width_percent",
            "band_rank_start",
            "band_rank_end_exclusive",
        ]
    )
)
SUMMARY_REQUIRED_FIELDS = (
    "run_id",
    "replay_mode",
    "group",
    "q_percent",
    "budget_mode",
    "selection_policy",
    "seed",
    "dataset_root",
    "oof_predictions",
    "manifest_dir",
    "defect_slots",
    "val_defect_slots",
    "val_normal_slots",
    "source_files",
    "source_hashes",
    "generated_files",
    "generated_hashes",
    "band_index",
    "band_start_percent",
    "band_end_percent",
    "band_width_percent",
    "band_rank_start",
    "band_rank_end_exclusive",
)
BASE_IMAGE_FIELDS = ("Filename", "canonical_image_relpath", "source_image_path")
INVALID_FILENAME_CHARS = set('<>:"|?*')


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_ids(raw_values: list[str] | None) -> set[str] | None:
    if not raw_values:
        return None
    out: list[str] = []
    for value in raw_values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return set(out)


def source_image_exists(row: dict[str, str], dataset_root: Path) -> bool:
    rel = row.get("canonical_image_relpath", "")
    if rel and (dataset_root / Path(rel)).exists():
        return True
    source = row.get("source_image_path", "")
    return bool(source and Path(source).exists())


def count_image_files(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def is_safe_leaf_filename(value: str) -> bool:
    if not value:
        return False
    if "/" in value or "\\" in value:
        return False
    if any(char in INVALID_FILENAME_CHARS or ord(char) < 32 for char in value):
        return False
    return Path(value).name == value


def is_finite_float(value: str) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed and parsed not in (float("inf"), float("-inf"))


def count_empty(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for item in rows if item.get(field, "") == "")


def add_required_field_errors(errors: list[str], prefix: str, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    for field in fields:
        missing = count_empty(rows, field)
        if missing:
            errors.append(f"{prefix}_empty_{field}:{missing}")


def count_bad_numeric(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for item in rows if not is_finite_float(item.get(field, "")))


def count_bad_binary(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for item in rows if item.get(field, "") not in {"0", "1"})


def count_bad_positive_int(rows: list[dict[str, str]], field: str) -> int:
    bad = 0
    for item in rows:
        try:
            if int(item.get(field, "")) <= 0:
                bad += 1
        except ValueError:
            bad += 1
    return bad


def expected_selected(row: dict[str, str]) -> int:
    normal_slots = int(row["normal_slots"])
    width = int(row["band_width_percent"])
    numerator = normal_slots * width
    if numerator % 100:
        raise ValueError(
            f"{row['run_id']} normal_slots={normal_slots} band_width_percent={width} "
            "does not make an integer selected count"
        )
    return numerator // 100


def add_value_integrity_errors(
    errors: list[str],
    row: dict[str, str],
    normal_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    val_defect_rows: list[dict[str, str]],
    val_normal_rows: list[dict[str, str]],
    selection_rows: list[dict[str, str]],
    selection_summary: dict,
) -> None:
    for field in RUN_MATRIX_REQUIRED_FIELDS:
        if row.get(field, "") == "":
            errors.append(f"run_matrix_empty_{field}")
    for field in (
        "q_percent",
        "normal_slots",
        "defect_slots",
        "selected_unique",
        "final_normal_rows",
        "final_defect_rows",
        "band_index",
        "band_start_percent",
        "band_end_percent",
        "band_width_percent",
        "band_rank_start",
        "band_rank_end_exclusive",
    ):
        try:
            if int(row.get(field, "")) < 0:
                errors.append(f"run_matrix_negative_{field}")
        except ValueError:
            errors.append(f"run_matrix_bad_int_{field}")

    for prefix, rows in (
        ("normal", normal_rows),
        ("defect", train_rows),
        ("val_defect", val_defect_rows),
        ("val_normal", val_normal_rows),
    ):
        add_required_field_errors(errors, prefix, rows, BASE_IMAGE_FIELDS)
        unsafe = sum(1 for item in rows if not is_safe_leaf_filename(item.get("Filename", "")))
        if unsafe:
            errors.append(f"{prefix}_unsafe_filename:{unsafe}")

    add_required_field_errors(errors, "normal", normal_rows, NORMAL_REPLAY_FIELDS)
    for field in ("oof_p_defect_operational", "oof_operational_threshold"):
        bad = count_bad_numeric(normal_rows, field)
        if bad:
            errors.append(f"normal_bad_numeric_{field}:{bad}")
    for field in ("oof_y_pred_operational", "oof_operational_correct", "replay_selected", "replay_displaced"):
        bad = count_bad_binary(normal_rows, field)
        if bad:
            errors.append(f"normal_bad_binary_{field}:{bad}")
    bad_slot = count_bad_positive_int(normal_rows, "replay_slot_index")
    if bad_slot:
        errors.append(f"normal_bad_replay_slot_index:{bad_slot}")

    add_required_field_errors(errors, "selection", selection_rows, SELECTION_REQUIRED_FIELDS)
    for field in ("p_defect_operational", "operational_threshold"):
        bad = count_bad_numeric(selection_rows, field)
        if bad:
            errors.append(f"selection_bad_numeric_{field}:{bad}")
    for field in ("y_pred_operational", "operational_correct"):
        bad = count_bad_binary(selection_rows, field)
        if bad:
            errors.append(f"selection_bad_binary_{field}:{bad}")

    for field in SUMMARY_REQUIRED_FIELDS:
        value = selection_summary.get(field)
        if value in ("", None, {}, []):
            errors.append(f"summary_empty_{field}")


def validate_one_run(
    row: dict[str, str],
    phase_root: Path,
    dataset_root: Path,
    work_root: Path | None,
    expected_replay_mode: str | None,
    source_hash_paths: dict[str, Path],
    expected_selected_keys: list[str],
    selected_sets: dict[str, set[str]],
) -> dict[str, str]:
    run_id = row["run_id"]
    replay_mode = row.get("replay_mode") or expected_replay_mode or "append"
    normal_slots = int(row["normal_slots"])
    defect_slots = int(row["defect_slots"])
    rank_start = int(row["band_rank_start"])
    rank_end = int(row["band_rank_end_exclusive"])
    band_width = int(row["band_width_percent"])
    manifest_dir = phase_root / "manifests" / run_id

    errors: list[str] = []
    for filename in REQUIRED_MANIFESTS:
        if not (manifest_dir / filename).exists():
            errors.append(f"missing:{filename}")

    train_rows = read_csv(manifest_dir / "train_manifest.csv")
    normal_rows = read_csv(manifest_dir / "normal_train_manifest.csv")
    val_defect_rows = read_csv(manifest_dir / "val_model_manifest.csv")
    val_normal_rows = read_csv(manifest_dir / "normal_val_model_manifest.csv")
    selection_rows = read_csv(manifest_dir / "selection_manifest.csv")
    selection_summary = load_json(manifest_dir / "selection_summary.json")

    add_value_integrity_errors(
        errors,
        row,
        normal_rows,
        train_rows,
        val_defect_rows,
        val_normal_rows,
        selection_rows,
        selection_summary,
    )

    selected_expected = expected_selected(row)
    if rank_end - rank_start != selected_expected:
        errors.append(f"rank_width:{rank_end - rank_start}!=selected_expected:{selected_expected}")
    if int(row["q_percent"]) != band_width:
        errors.append(f"q_percent:{row['q_percent']}!=band_width:{band_width}")
    if int(row["selected_unique"]) != selected_expected:
        errors.append(f"run_matrix_selected_unique:{row['selected_unique']}!=expected:{selected_expected}")

    expected_normal_rows = normal_slots + selected_expected if replay_mode == "append" else normal_slots
    expected_displaced = 0 if replay_mode == "append" else selected_expected

    selected_rows = [item for item in selection_rows if item.get("role") == "selected"]
    displaced_rows = [item for item in selection_rows if item.get("role") == "displaced"]
    replay_rows = [item for item in normal_rows if item.get("replay_slot_type") == "replay_duplicate"]
    base_selected_rows = [item for item in normal_rows if item.get("replay_slot_type") == "base_selected"]
    filenames = [item.get("Filename", "") for item in normal_rows]
    canonical_counts = Counter(item.get("canonical_image_relpath", "").replace("\\", "/") for item in normal_rows)
    actual_selected_keys = [item.get("source_canonical_image_relpath", "").replace("\\", "/") for item in selected_rows]
    selected_sets[run_id] = set(actual_selected_keys)

    if actual_selected_keys != expected_selected_keys:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(actual_selected_keys, expected_selected_keys), start=1)
                if actual != expected
            ),
            None,
        )
        errors.append(f"selected_band_slice_mismatch:first_index={mismatch}")

    checks = {
        "normal_rows": len(normal_rows),
        "defect_rows": len(train_rows),
        "val_defect_rows": len(val_defect_rows),
        "val_normal_rows": len(val_normal_rows),
        "selected_rows": len(selected_rows),
        "base_selected_rows": len(base_selected_rows),
        "replay_duplicate_rows": len(replay_rows),
        "displaced_rows": len(displaced_rows),
        "duplicate_filenames": len(filenames) - len(set(filenames)),
        "duplicate_canonical_images": sum(count - 1 for count in canonical_counts.values() if count > 1),
        "missing_train_images": sum(1 for item in train_rows if not source_image_exists(item, dataset_root)),
        "missing_normal_images": sum(1 for item in normal_rows if not source_image_exists(item, dataset_root)),
        "missing_val_defect_images": sum(1 for item in val_defect_rows if not source_image_exists(item, dataset_root)),
        "missing_val_normal_images": sum(1 for item in val_normal_rows if not source_image_exists(item, dataset_root)),
    }

    generated_hash_mismatches = []
    for filename, expected_hash in selection_summary.get("generated_hashes", {}).items():
        path = manifest_dir / filename
        if not path.exists():
            generated_hash_mismatches.append(f"{filename}:missing")
            continue
        if file_sha256(path) != expected_hash:
            generated_hash_mismatches.append(f"{filename}:hash_mismatch")

    source_hash_mismatches = []
    for filename in SOURCE_HASH_FILES:
        expected_hash = selection_summary.get("source_hashes", {}).get(filename)
        path = source_hash_paths.get(filename)
        if not expected_hash:
            source_hash_mismatches.append(f"{filename}:missing_expected_hash")
            continue
        if path is None or not path.exists():
            source_hash_mismatches.append(f"{filename}:missing_source_file")
            continue
        if file_sha256(path) != expected_hash:
            source_hash_mismatches.append(f"{filename}:hash_mismatch")

    if checks["normal_rows"] != expected_normal_rows:
        errors.append(f"normal_rows:{checks['normal_rows']}!=expected:{expected_normal_rows}")
    if checks["defect_rows"] != defect_slots:
        errors.append(f"defect_rows:{checks['defect_rows']}!=expected:{defect_slots}")
    if checks["selected_rows"] != selected_expected:
        errors.append(f"selected_rows:{checks['selected_rows']}!=expected:{selected_expected}")
    if checks["base_selected_rows"] != selected_expected:
        errors.append(f"base_selected_rows:{checks['base_selected_rows']}!=expected:{selected_expected}")
    if checks["replay_duplicate_rows"] != selected_expected:
        errors.append(f"replay_duplicate_rows:{checks['replay_duplicate_rows']}!=expected:{selected_expected}")
    if checks["displaced_rows"] != expected_displaced:
        errors.append(f"displaced_rows:{checks['displaced_rows']}!=expected:{expected_displaced}")
    if checks["duplicate_filenames"]:
        errors.append(f"duplicate_filenames:{checks['duplicate_filenames']}")
    if checks["duplicate_canonical_images"] != selected_expected:
        errors.append(f"duplicate_canonical_images:{checks['duplicate_canonical_images']}!=expected:{selected_expected}")
    if generated_hash_mismatches:
        errors.append("generated_hash_mismatch:" + ",".join(generated_hash_mismatches[:5]))
    if source_hash_mismatches:
        errors.append("source_hash_mismatch:" + ",".join(source_hash_mismatches[:5]))
    for key in (
        "missing_train_images",
        "missing_normal_images",
        "missing_val_defect_images",
        "missing_val_normal_images",
    ):
        if checks[key]:
            errors.append(f"{key}:{checks[key]}")
    if int(selection_summary.get("final_normal_rows", -1)) != checks["normal_rows"]:
        errors.append("summary_final_normal_rows_mismatch")

    work_counts = {
        "work_train_normal_files": "",
        "work_train_defect_files": "",
        "work_val_normal_files": "",
        "work_val_defect_files": "",
    }
    if work_root is not None:
        run_work = work_root / run_id / "full"
        work_counts = {
            "work_train_normal_files": str(count_image_files(run_work / "train" / "no_target")),
            "work_train_defect_files": str(count_image_files(run_work / "train" / "target_defect")),
            "work_val_normal_files": str(count_image_files(run_work / "val" / "no_target")),
            "work_val_defect_files": str(count_image_files(run_work / "val" / "target_defect")),
        }
        if work_counts["work_train_normal_files"] != "-1" and int(work_counts["work_train_normal_files"]) != checks["normal_rows"]:
            errors.append("work_train_normal_count_mismatch")
        if work_counts["work_train_defect_files"] != "-1" and int(work_counts["work_train_defect_files"]) != checks["defect_rows"]:
            errors.append("work_train_defect_count_mismatch")
        if work_counts["work_val_normal_files"] != "-1" and int(work_counts["work_val_normal_files"]) != checks["val_normal_rows"]:
            errors.append("work_val_normal_count_mismatch")
        if work_counts["work_val_defect_files"] != "-1" and int(work_counts["work_val_defect_files"]) != checks["val_defect_rows"]:
            errors.append("work_val_defect_count_mismatch")

    out = {
        "run_id": run_id,
        "group": row["group"],
        "q_percent": row["q_percent"],
        "band_start_percent": row["band_start_percent"],
        "band_end_percent": row["band_end_percent"],
        "band_width_percent": row["band_width_percent"],
        "replay_mode": replay_mode,
        "manifest_dir": str(manifest_dir),
        "expected_normal_rows": str(expected_normal_rows),
        "expected_selected": str(selected_expected),
        "expected_displaced": str(expected_displaced),
        "expected_rank_start": str(rank_start),
        "expected_rank_end_exclusive": str(rank_end),
        "status": "ok" if not errors else "failed",
        "errors": ";".join(errors),
    }
    out.update({key: str(value) for key, value in checks.items()})
    out.update(work_counts)
    return out


def add_cross_run_errors(rows: list[dict[str, str]], selected_sets: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    hn1_ids = [f"HN1-{index:02d}" for index in range(1, 21) if f"HN1-{index:02d}" in selected_sets]
    seen: dict[str, str] = {}
    for run_id in hn1_ids:
        for key in selected_sets[run_id]:
            if key in seen:
                errors.append(f"hn1_overlap:{run_id}:{seen[key]}:{key}")
                break
            seen[key] = run_id

    for index in range(1, 11):
        hn2_id = f"HN2-{index:02d}"
        hn1_a = f"HN1-{2 * index - 1:02d}"
        hn1_b = f"HN1-{2 * index:02d}"
        if hn2_id in selected_sets and hn1_a in selected_sets and hn1_b in selected_sets:
            expected = selected_sets[hn1_a] | selected_sets[hn1_b]
            if selected_sets[hn2_id] != expected:
                errors.append(f"hn2_pair_union_mismatch:{hn2_id}!={hn1_a}+{hn1_b}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate phase-1 HN band replay manifests.")
    parser.add_argument("--phase-root", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--oof-predictions", default=None)
    parser.add_argument("--work-root", default=None, help="Optional root containing run_id/full YOLO workdirs.")
    parser.add_argument("--skip-workdir", action="store_true", help="Only validate manifests and source images.")
    parser.add_argument("--run-id", action="append", help="Validate one or more run ids, comma-separated or repeated.")
    parser.add_argument("--replay-mode", choices=("append", "fixed"), default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_script()
    phase_root = Path(args.phase_root).resolve() if args.phase_root else repo_root / DEFAULT_PHASE_ROOT
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT
    manifest_root = dataset_root / "manifests"
    oof_predictions = Path(args.oof_predictions).resolve() if args.oof_predictions else repo_root / DEFAULT_OOF_PREDICTIONS
    source_hash_paths = {
        "train_manifest.csv": manifest_root / "train_manifest.csv",
        "normal_train_manifest.csv": manifest_root / "normal_train_manifest.csv",
        "val_model_manifest.csv": manifest_root / "val_model_manifest.csv",
        "normal_val_model_manifest.csv": manifest_root / "normal_val_model_manifest.csv",
        "oof_predictions_merged.csv": oof_predictions,
    }
    work_root = None if args.skip_workdir else (Path(args.work_root).resolve() if args.work_root else phase_root / "workdirs")
    run_ids = parse_run_ids(args.run_id)
    run_matrix = read_csv(phase_root / "run_matrix.csv")
    selected_rows = [row for row in run_matrix if run_ids is None or row["run_id"] in run_ids]
    if not selected_rows:
        raise ValueError("No matching run ids found")

    normal_rows = base.read_csv(manifest_root / base.NORMAL_TRAIN_MANIFEST)
    oof_rows = base.read_csv(oof_predictions)
    oof_by_key = base.group_oof_normals(oof_rows)
    sorted_hn_rows = base.sort_hn(normal_rows, oof_by_key)

    selected_sets: dict[str, set[str]] = {}
    summary_rows = []
    for row in selected_rows:
        rank_start = int(row["band_rank_start"])
        rank_end = int(row["band_rank_end_exclusive"])
        expected_keys = [base.canonical_key(item) for item in sorted_hn_rows[rank_start:rank_end]]
        summary_rows.append(
            validate_one_run(
                row,
                phase_root,
                dataset_root,
                work_root,
                args.replay_mode,
                source_hash_paths,
                expected_keys,
                selected_sets,
            )
        )

    cross_errors = add_cross_run_errors(summary_rows, selected_sets)
    if cross_errors:
        for row in summary_rows:
            row["cross_run_errors"] = ";".join(cross_errors[:10])
            row["status"] = "failed"
            row["errors"] = ";".join(part for part in (row.get("errors", ""), row["cross_run_errors"]) if part)
    else:
        for row in summary_rows:
            row["cross_run_errors"] = ""

    output_csv = Path(args.output_csv).resolve() if args.output_csv else phase_root / "validation_summary.csv"
    output_json = Path(args.output_json).resolve() if args.output_json else phase_root / "validation_summary.json"
    fieldnames = list(dict.fromkeys(key for row in summary_rows for key in row.keys()))
    write_csv(output_csv, summary_rows, fieldnames)
    failures = [row for row in summary_rows if row["status"] != "ok"]
    write_json(
        output_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase_root": str(phase_root),
            "dataset_root": str(dataset_root),
            "work_root": str(work_root),
            "source_hash_paths": {name: str(path) for name, path in source_hash_paths.items()},
            "checked_runs": len(summary_rows),
            "failed_runs": len(failures),
            "cross_run_error_count": len(cross_errors),
            "cross_run_errors": cross_errors[:20],
            "output_csv": str(output_csv),
        },
    )
    print(f"checked_runs={len(summary_rows)}")
    print(f"failed_runs={len(failures)}")
    print(f"validation_csv={output_csv}")
    if failures:
        print("first_failures=")
        for row in failures[:5]:
            print(f"{row['run_id']}: {row['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
