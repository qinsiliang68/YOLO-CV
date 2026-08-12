"""Reproduce current-v3 evidence for the expert review's P0 findings."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable
import uuid

import polars as pl

from .evaluation import PredictionRow, compute_raw_safety_frontier
from .matrix import build_preregistered_matrix, validate_preregistered_matrix
from .oof_reference import build_oof_epoch200_reference
from .seeds import build_seed_registry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_sources(repo_root: Path) -> tuple[Path, ...]:
    excluded = {"expert_delivery_audit.py", "review_reproduction.py"}
    return tuple(
        path
        for path in sorted((repo_root / "stage1_dynamic_replay_v3").glob("*.py"))
        if path.name not in excluded
    )


def _references(repo_root: Path, sources: Iterable[Path], token: str) -> list[str]:
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    hits: list[str] = []
    for path in sources:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(repo_root).as_posix()}:{line_number}:{line.strip()}")
    return hits


def _oof_reproduction(temp_root: Path) -> dict[str, object]:
    defect_manifest = temp_root / "defect.csv"
    normal_manifest = temp_root / "normal.csv"
    _write_csv(
        defect_manifest,
        ["canonical_image_relpath", "Defect"],
        [{"canonical_image_relpath": "defect/d.png", "Defect": 1}],
    )
    _write_csv(
        normal_manifest,
        ["canonical_image_relpath", "Defect"],
        [{"canonical_image_relpath": "normal/n.png", "Defect": 0}],
    )
    fields = ["sample_id", "y_true", "oof_fold", "epoch", "p_defect_raw"]
    # The row fold deliberately contradicts the containing fold directory.
    _write_csv(
        temp_root / "jobs" / "job-a" / "fold_00" / "epoch_200_predictions.csv",
        fields,
        [{"sample_id": "defect/d.png", "y_true": 1, "oof_fold": "01", "epoch": 200, "p_defect_raw": 0.8}],
    )
    _write_csv(
        temp_root / "jobs" / "job-b" / "fold_01" / "epoch_200_predictions.csv",
        fields,
        [{"sample_id": "normal/n.png", "y_true": 0, "oof_fold": "00", "epoch": 200, "p_defect_raw": 0.2}],
    )
    result = build_oof_epoch200_reference(
        jobs_root=temp_root / "jobs",
        defect_manifest=defect_manifest,
        normal_manifest=normal_manifest,
        output_path=temp_root / "oof.parquet",
        expected_rows=2,
        expected_defect=1,
        expected_normal=1,
        expected_fold_count=2,
    )
    table = pl.read_parquet(result.output_path).sort("sample_id")
    return {
        "fold_is_not_a_global_trajectory_axis": result.row_count == 2 and table.height == 2,
        "path_fold_mismatch_accepted": table.get_column("oof_fold").to_list() == ["01", "00"],
        "row_count": result.row_count,
        "fold_count": result.fold_count,
        "observed_rows": table.select("sample_id", "oof_fold", "source_prediction_relpath").to_dicts(),
        "interpretation": (
            "v3 no longer creates a sample-by-fold Cartesian cube, but it does not bind each row's "
            "oof_fold to the containing job/fold lineage or prove held-out training membership."
        ),
    }


def _dual_threshold_reproduction() -> dict[str, object]:
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    probabilities = [0.95, 0.85, 0.75, 0.65, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    rows = [
        PredictionRow(str(index), label, probability, 200)
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True))
    ]
    result = compute_raw_safety_frontier(rows, max_fn=1, target_tn=5)
    fn_limit_point = result.frontier[-1]
    return {
        "independent_metric_demonstrated": result.fn_at_target_tn != fn_limit_point.actual_fn,
        "tn_at_fn_limit": fn_limit_point.tn,
        "fn_at_fn_limit_point": fn_limit_point.actual_fn,
        "fn_at_target_tn": result.fn_at_target_tn,
        "target_tn_reachable": result.target_tn_reachable,
        "frontier": [point.__dict__ for point in result.frontier],
        "interpretation": (
            "v3 computes FN at the target-TN constraint from a separate feasible candidate set; "
            "it does not reuse the FN-limit confusion matrix."
        ),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_v3_p0_reproduction(repo_root: str | Path, output_path: str | Path) -> dict[str, object]:
    repo = Path(repo_root).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    sources = _runtime_sources(repo)
    source_hashes = {
        path.relative_to(repo).as_posix(): _sha256(path)
        for path in sources
    }
    with tempfile.TemporaryDirectory(prefix="stage1-v3-p0-repro-") as temporary:
        oof = _oof_reproduction(Path(temporary))
    frontier = _dual_threshold_reproduction()
    matrix = build_preregistered_matrix(build_seed_registry())
    matrix_validation = validate_preregistered_matrix(matrix)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    payload: dict[str, object] = {
        "schema_version": "stage1.v3_p0_review_reproduction.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_AUDIT_REPRODUCTION",
        "git_head": git_head,
        "executed_expert_code": False,
        "scientific_result": False,
        "source_hashes": source_hashes,
        "p0_01_oof": oof,
        "p0_02_checkpoint": {
            "best_pt_references": _references(repo, sources, "best.pt"),
            "endpoint_epochs": sorted({row.endpoint_epoch for row in matrix}),
            "interpretation": "No v3 runtime source selects best.pt; the preregistered logical endpoint is epoch 200.",
        },
        "p0_03_data_roles": {
            "val_target_references": _references(repo, sources, "val_target"),
            "val_model_references": _references(repo, sources, "val_model"),
            "val_op_references": _references(repo, sources, "val_op"),
            "interpretation": (
                "The old val_op reuse is absent because v3 has no target-gradient/CLD method, but an independent "
                "val_target role is also not implemented."
            ),
        },
        "p0_04_test_oracle": {
            "test_oracle_references": _references(repo, sources, "test_oracle_curve"),
            "interpretation": "No v3 runtime source exposes the reviewed test-oracle canonical path.",
        },
        "p0_05_dual_threshold": frontier,
        "p0_06_critical_cld": {
            "critical_cld_references": _references(repo, sources, "critical_cld"),
            "interpretation": "The conflicting CLD implementation is absent, but target-direction A is not implemented.",
        },
        "p0_07_matrix": {
            "validation_status": matrix_validation.status,
            "run_count": len(matrix),
            "cycle_counts": matrix_validation.cycle_counts,
            "all_release_states_held": all(row.release_state == "HELD" for row in matrix),
            "gradient_arms": sum(row.gradient_collection for row in matrix),
            "defect_guard_arms": sum(row.defect_guard_fraction > 0 for row in matrix),
            "interpretation": (
                "The preserved v3 matrix is internally consistent and HELD, but its 236-run RHO-only design "
                "does not implement the new Q/R/A/D evidence goal."
            ),
        },
    }
    _atomic_json(output, payload)
    return payload


__all__ = ["build_v3_p0_reproduction"]
