from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import ValidationError
from .util import atomic_write_bytes, atomic_write_json, sha256_file


SELECTION_COLUMNS = {"run_slot", "rank", "sample_id", "y_true", "replay_role"}
QUEUE_ROOT_FILES = (
    "frozen_experiment_matrix.csv", "selection_index.csv", "overlap_decisions.json",
    "QUEUE_VALIDATION.json", "CROSS_TRIAD_AUDIT.json", "PREPARATION_COMPLETE.json",
    "MACHINE_ALLOCATION_AUDIT.json", "site_asset_binding.json",
)


def write_frozen_queue_manifest(generated_root: str | Path, output_path: str | Path) -> Path:
    generated_root = Path(generated_root).resolve()
    output_path = Path(output_path).resolve()
    files = [generated_root / name for name in QUEUE_ROOT_FILES if (generated_root / name).exists()]
    for dirname in ("selections", "machine_shards"):
        directory = generated_root / dirname
        if directory.exists(): files.extend(path for path in directory.rglob("*") if path.is_file())
    rows = [
        {"relative_path": path.relative_to(generated_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(set(files)) if path.resolve() != output_path
    ]
    return atomic_write_bytes(output_path, pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), overwrite=True)


def validate_frozen_queues(
    matrix_path: str | Path,
    selection_index_path: str | Path,
    artifact_root: str | Path,
    shard_dir: str | Path,
    output_path: str | Path,
    *,
    expected_runs: int = 240,
    expected_triads: int = 80,
    main_machine_count: int = 10,
    reserve_machine_count: int = 2,
) -> dict:
    matrix_path = Path(matrix_path)
    selection_index_path = Path(selection_index_path)
    artifact_root = Path(artifact_root).resolve()
    shard_dir = Path(shard_dir)
    issues: list[str] = []
    matrix = pd.read_csv(matrix_path, dtype={"run_slot": "string", "triad_id": "string", "arm": "string"})
    if len(matrix) != expected_runs: issues.append(f"Matrix rows {len(matrix)} != {expected_runs}")
    if matrix.run_slot.nunique() != len(matrix): issues.append("Matrix run_slot values are not unique")
    if matrix.triad_id.nunique() != expected_triads: issues.append(f"Matrix triads {matrix.triad_id.nunique()} != {expected_triads}")
    triad_sizes = matrix.groupby("triad_id").size()
    if not (triad_sizes == 3).all(): issues.append("Every triad must contain exactly three rows")
    arm_sets = matrix.groupby("triad_id").arm.apply(lambda values: set(values.astype(str)))
    if not all(value == {"T", "R1", "R2"} for value in arm_sets): issues.append("Every triad must contain T/R1/R2")

    index = pd.read_csv(selection_index_path, dtype="string")
    if len(index) != expected_runs or index.run_slot.nunique() != expected_runs:
        issues.append("Selection index must contain one unique row per run")
    if set(index.run_slot.astype(str)) != set(matrix.run_slot.astype(str)):
        issues.append("Selection index run slots do not equal matrix run slots")
    selection_rows = 0
    for row in index.itertuples(index=False):
        path = (artifact_root / str(row.selection_manifest)).resolve()
        try: path.relative_to(artifact_root)
        except ValueError: issues.append(f"Selection path escapes artifact root: {path}"); continue
        if not path.exists(): issues.append(f"Missing selection manifest: {path}"); continue
        if sha256_file(path).upper() != str(row.sha256).upper(): issues.append(f"Selection SHA mismatch: {row.run_slot}")
        frame = pd.read_csv(path, dtype={"run_slot": "string", "sample_id": "string"})
        missing = SELECTION_COLUMNS - set(frame.columns)
        if missing: issues.append(f"{row.run_slot} missing columns: {sorted(missing)}"); continue
        matrix_row = matrix[matrix.run_slot == str(row.run_slot)]
        if len(matrix_row) != 1: issues.append(f"Matrix row missing for {row.run_slot}"); continue
        budget = int(matrix_row.iloc[0].budget)
        if len(frame) != budget: issues.append(f"{row.run_slot} rows {len(frame)} != budget {budget}")
        if frame.sample_id.duplicated().any(): issues.append(f"{row.run_slot} contains duplicate sample IDs")
        if set(frame.run_slot.astype(str)) != {str(row.run_slot)}: issues.append(f"{row.run_slot} embeds a different run_slot")
        if not set(frame.y_true.astype(int).unique()).issubset({0, 1}): issues.append(f"{row.run_slot} has invalid labels")
        selection_rows += len(frame)

    expected_machine_count = main_machine_count + reserve_machine_count
    shard_files = sorted(shard_dir.glob("machine_*_jobs.csv"))
    if len(shard_files) != expected_machine_count:
        issues.append(f"Machine shard files {len(shard_files)} != {expected_machine_count}")
    shard_frames = []
    triad_to_machine: dict[str, str] = {}
    for position, path in enumerate(shard_files, start=1):
        frame = pd.read_csv(path, dtype={"run_slot": "string", "triad_id": "string", "arm": "string"})
        if position <= main_machine_count:
            expected = expected_runs // main_machine_count
            if len(frame) != expected: issues.append(f"{path.name} rows {len(frame)} != {expected}")
        elif len(frame) != 0:
            issues.append(f"Reserve shard must be empty: {path.name}")
        for triad_id in frame.triad_id.astype(str).unique() if "triad_id" in frame else []:
            previous = triad_to_machine.setdefault(triad_id, path.name)
            if previous != path.name: issues.append(f"Triad split across machines: {triad_id}")
        shard_frames.append(frame)
    assigned = pd.concat(shard_frames, ignore_index=True) if shard_frames else matrix.iloc[:0]
    if len(assigned) != expected_runs or assigned.run_slot.nunique() != expected_runs:
        issues.append("Machine shards do not assign every run exactly once")
    if set(assigned.run_slot.astype(str)) != set(matrix.run_slot.astype(str)):
        issues.append("Machine shard run slots do not equal frozen matrix")

    report = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "matrix_rows": len(matrix),
        "triad_count": int(matrix.triad_id.nunique()),
        "selection_manifest_count": len(index),
        "selection_total_rows": int(selection_rows),
        "machine_shard_count": len(shard_files),
        "shard_run_count": len(assigned),
        "matrix_sha256": sha256_file(matrix_path),
        "selection_index_sha256": sha256_file(selection_index_path),
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"Frozen queue validation failed; see {output_path}")
    return report
