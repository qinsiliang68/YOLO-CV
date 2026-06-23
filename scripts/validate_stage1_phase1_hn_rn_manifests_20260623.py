# -*- coding: utf-8 -*-
"""Validate phase-1 HN/RN replay manifests and prepared image directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_PHASE_ROOT = Path("artifacts") / "stage1_phase1_hn_rn_20260623"
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OOF_PREDICTIONS = (
    Path("artifacts")
    / "stage1_oof_predictions_calop_20260621"
    / "merged_10fold_20260622"
    / "oof_predictions_merged.csv"
)
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


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def parse_run_ids(raw_values: list[str] | None) -> set[str] | None:
    if not raw_values:
        return None
    out: list[str] = []
    for value in raw_values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return set(out)


def expected_selected(normal_slots: int, q_percent: int) -> int:
    numerator = normal_slots * q_percent
    if numerator % 100:
        raise ValueError(f"normal_slots={normal_slots} q_percent={q_percent} does not make an integer selected count")
    return numerator // 100


def validate_one_run(
    row: dict[str, str],
    phase_root: Path,
    dataset_root: Path,
    work_root: Path | None,
    expected_replay_mode: str | None,
    source_hash_paths: dict[str, Path],
) -> dict[str, str]:
    run_id = row["run_id"]
    group = row["group"]
    q_percent = int(row["q_percent"])
    normal_slots = int(row["normal_slots"])
    defect_slots = int(row["defect_slots"])
    replay_mode = row.get("replay_mode") or expected_replay_mode or "append"
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
    expected_generated_hashes = selection_summary.get("generated_hashes", {})
    expected_source_hashes = selection_summary.get("source_hashes", {})

    selected_expected = 0 if group == "BL" else expected_selected(normal_slots, q_percent)
    expected_normal_rows = normal_slots + selected_expected if replay_mode == "append" else normal_slots
    expected_displaced = 0 if replay_mode == "append" else selected_expected

    selected_rows = [item for item in selection_rows if item.get("role") == "selected"]
    displaced_rows = [item for item in selection_rows if item.get("role") == "displaced"]
    replay_rows = [item for item in normal_rows if item.get("replay_slot_type") == "replay_duplicate"]
    base_selected_rows = [item for item in normal_rows if item.get("replay_slot_type") == "base_selected"]
    filenames = [item.get("Filename", "") for item in normal_rows]
    canonical_counts = Counter(item.get("canonical_image_relpath", "").replace("\\", "/") for item in normal_rows)

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
    for filename, expected_hash in expected_generated_hashes.items():
        path = manifest_dir / filename
        if not path.exists():
            generated_hash_mismatches.append(f"{filename}:missing")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            generated_hash_mismatches.append(f"{filename}:hash_mismatch")

    source_hash_mismatches = []
    for filename in SOURCE_HASH_FILES:
        expected_hash = expected_source_hashes.get(filename)
        path = source_hash_paths.get(filename)
        if not expected_hash:
            source_hash_mismatches.append(f"{filename}:missing_expected_hash")
            continue
        if path is None or not path.exists():
            source_hash_mismatches.append(f"{filename}:missing_source_file")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
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
        "group": group,
        "q_percent": str(q_percent),
        "replay_mode": replay_mode,
        "manifest_dir": str(manifest_dir),
        "expected_normal_rows": str(expected_normal_rows),
        "expected_selected": str(selected_expected),
        "expected_displaced": str(expected_displaced),
        "status": "ok" if not errors else "failed",
        "errors": ";".join(errors),
    }
    out.update({key: str(value) for key, value in checks.items()})
    out.update(work_counts)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate phase-1 HN/RN replay manifests.")
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

    summary_rows = [
        validate_one_run(row, phase_root, dataset_root, work_root, args.replay_mode, source_hash_paths)
        for row in selected_rows
    ]
    output_csv = Path(args.output_csv).resolve() if args.output_csv else phase_root / "validation_summary.csv"
    output_json = Path(args.output_json).resolve() if args.output_json else phase_root / "validation_summary.json"
    fieldnames = list(summary_rows[0].keys())
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
