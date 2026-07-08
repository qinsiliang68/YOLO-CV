from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from scripts.evaluate_stage1_cls_gate import (
    EvalConfig,
    cross_entropy,
    ensure_yolo_import,
    logit,
    predict_records,
    release_torch_memory,
    uncertainty,
    write_csv,
    write_json,
)
from scripts.predict_stage1_oof_folds_20260621 import FoldJob, load_fold_holdout_records, parse_fold_spec
from scripts.stage1_sample_value_experiment_layout_20260708 import (
    ACTIVE_EXPERIMENT_ID,
    FAMILY_ROOT,
    experiment_root,
    family_root,
    initialize_experiment_family_layout,
    stage_root,
)


DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OOF_ROOT = Path("artifacts") / "stage1_oof_folds_10fold_20260617"
DEFAULT_RUNS_ROOT = Path("artifacts") / "stage1_oof_runs_20260617"
DEFAULT_YOLO_ROOT = Path("YOLOv11")

DYNAMICS_COLUMNS = (
    "sample_id",
    "checkpoint_id",
    "oof_fold",
    "human_fold",
    "epoch",
    "eval_split",
    "source_split",
    "y_true",
    "y_pred_raw",
    "raw_correct",
    "p_defect_raw",
    "p_normal_raw",
    "raw_logit",
    "raw_margin_signed",
    "raw_abs_margin",
    "raw_cross_entropy",
    "raw_uncertainty",
    "risk_rank_desc_within_label",
    "risk_rank_pct_desc_within_label",
    "rank_group_n",
    "Filename",
    "canonical_image_relpath",
    "image_path",
    "checkpoint_path",
    "checkpoint_sha256",
)


@dataclass(frozen=True)
class CheckpointPlanRow:
    fold: int
    epoch: int
    checkpoint_path: Path | None
    exists: bool
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "oof_fold": f"{self.fold:02d}",
            "human_fold": str(self.fold + 1),
            "epoch": str(self.epoch),
            "checkpoint_id": checkpoint_id(self.fold, self.epoch),
            "checkpoint_path": "" if self.checkpoint_path is None else str(self.checkpoint_path),
            "exists": str(int(self.exists)),
            "source": self.source,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: Path | None, default: Path) -> Path:
    chosen = path if path is not None else default
    return chosen if chosen.is_absolute() else root / chosen


def parse_epoch_spec(value: str) -> list[int]:
    epochs: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid epoch range: {part}")
            epochs.extend(range(start, end + 1))
        else:
            epoch = int(part)
            if epoch <= 0:
                raise ValueError(f"Epoch must be positive: {epoch}")
            epochs.append(epoch)

    output: list[int] = []
    seen: set[int] = set()
    for epoch in epochs:
        if epoch not in seen:
            output.append(epoch)
            seen.add(epoch)
    if not output:
        raise ValueError("No epochs selected")
    return output


def checkpoint_id(fold: int, epoch: int) -> str:
    return f"fold_{fold:02d}_epoch_{epoch:03d}"


def job_dynamics_root(dynamics_root: Path, job_id: str) -> Path:
    if not job_id:
        raise ValueError("job_id must be non-empty")
    if any(part in job_id for part in ("/", "\\", ":", "..")):
        raise ValueError(f"Unsafe job_id: {job_id}")
    return dynamics_root / "jobs" / job_id


def validate_export_request(
    *,
    dry_run: bool,
    job_id: str,
    folds: list[int],
    fold_run_map: dict[int, Path],
    allow_auto_discover: bool,
) -> None:
    if not dry_run and not job_id:
        raise ValueError("Formal export requires --job-id so parallel jobs cannot overwrite each other.")
    if dry_run:
        return
    if allow_auto_discover:
        return
    missing = [fold for fold in folds if fold not in fold_run_map]
    if missing:
        formatted = ",".join(f"{fold:02d}" for fold in missing)
        raise ValueError(
            f"Formal export requires --fold-run-map for every fold unless --allow-auto-discover is set. Missing: {formatted}"
        )


_FOLD_ROOT_CACHE: dict[tuple[str, int], list[Path]] = {}


def fold_root_candidates(runs_root: Path, fold: int) -> list[Path]:
    cache_key = (str(runs_root.resolve()), fold)
    if cache_key in _FOLD_ROOT_CACHE:
        return _FOLD_ROOT_CACHE[cache_key]

    wanted = f"fold_{fold:02d}"
    candidates: list[Path] = []
    if runs_root.name == wanted:
        candidates.append(runs_root)
    for candidate in (runs_root / wanted, runs_root / "folds" / wanted):
        candidates.append(candidate)
    if runs_root.exists():
        candidates.extend(sorted(path for path in runs_root.rglob(wanted) if path.is_dir()))

    output: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            output.append(candidate)
            seen.add(resolved)
    _FOLD_ROOT_CACHE[cache_key] = output
    return output


def fold_uses_zero_based_epoch_names(fold_root: Path) -> bool:
    direct = fold_root / "weights" / "epoch0.pt"
    if direct.exists():
        return True
    return any(fold_root.glob("*/weights/epoch0.pt"))


def epoch_weight_names(fold_root: Path, epoch: int) -> list[str]:
    if fold_uses_zero_based_epoch_names(fold_root):
        zero_epoch = epoch - 1
        return [f"epoch{zero_epoch}.pt", f"epoch_{zero_epoch:03d}.pt"]
    return [f"epoch{epoch}.pt", f"epoch_{epoch:03d}.pt"]


def sample_id_for_row(row: dict[str, str]) -> str:
    for key in ("canonical_image_relpath", "image_path", "Filename"):
        value = row.get(key, "")
        if value:
            return value.replace("\\", "/")
    raise ValueError(f"Unable to build sample_id from row keys: {sorted(row)}")


def read_checkpoint_map(path: Path | None) -> dict[tuple[int, int], Path]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint map CSV: {path}")
    output: dict[tuple[int, int], Path] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"oof_fold", "epoch", "checkpoint_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"checkpoint map missing columns: {sorted(missing)}")
        for row in reader:
            fold = int(row["oof_fold"])
            epoch = int(row["epoch"])
            key = (fold, epoch)
            if key in output:
                raise ValueError(f"Duplicate checkpoint map key: oof_fold={fold:02d}, epoch={epoch}")
            ckpt = Path(row["checkpoint_path"])
            output[key] = ckpt if ckpt.is_absolute() else path.parent / ckpt
    return output


def read_fold_run_map(path: Path | None) -> dict[int, Path]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing fold-run map CSV: {path}")
    output: dict[int, Path] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"oof_fold", "run_dir"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"fold-run map missing columns: {sorted(missing)}")
        for row in reader:
            fold = int(row["oof_fold"])
            if fold in output:
                raise ValueError(f"Duplicate fold-run map key: oof_fold={fold:02d}")
            run_dir = Path(row["run_dir"])
            output[fold] = run_dir if run_dir.is_absolute() else path.parent / run_dir
    return output


def checkpoint_candidates_for_run_dir(run_dir: Path, epoch: int) -> list[Path]:
    candidates: list[Path] = []
    for name in epoch_weight_names(run_dir, epoch):
        candidates.append(run_dir / "weights" / name)
        candidates.extend(sorted(run_dir.glob(f"*/weights/{name}"), key=lambda item: item.stat().st_mtime, reverse=True))
        candidates.extend(sorted(run_dir.glob(f"*/{name}"), key=lambda item: item.stat().st_mtime, reverse=True))
    return candidates


def checkpoint_candidates(runs_root: Path, fold: int, epoch: int) -> list[Path]:
    candidates: list[Path] = []
    for fold_root in fold_root_candidates(runs_root, fold):
        for name in epoch_weight_names(fold_root, epoch):
            candidates.append(fold_root / "weights" / name)
            candidates.extend(
                sorted(fold_root.glob(f"*/weights/{name}"), key=lambda item: item.stat().st_mtime, reverse=True)
            )
            candidates.extend(sorted(fold_root.glob(f"*/{name}"), key=lambda item: item.stat().st_mtime, reverse=True))
    return candidates


def resolve_checkpoint_path(
    runs_root: Path,
    *,
    fold: int,
    epoch: int,
    checkpoint_map: dict[tuple[int, int], Path] | None = None,
    fold_run_map: dict[int, Path] | None = None,
    allow_auto_discover: bool = True,
) -> Path:
    if checkpoint_map and (fold, epoch) in checkpoint_map:
        mapped = checkpoint_map[(fold, epoch)]
        if not mapped.exists():
            raise FileNotFoundError(f"Mapped checkpoint does not exist: {mapped}")
        return mapped

    if fold_run_map and fold in fold_run_map:
        seen_run_candidates: set[Path] = set()
        for candidate in checkpoint_candidates_for_run_dir(fold_run_map[fold], epoch):
            if candidate in seen_run_candidates:
                continue
            seen_run_candidates.add(candidate)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing checkpoint for fold={fold:02d}, epoch={epoch} in mapped run_dir={fold_run_map[fold]}"
        )

    if not allow_auto_discover:
        raise FileNotFoundError(
            f"Auto-discovery disabled and fold={fold:02d} is not present in --fold-run-map for epoch={epoch}"
        )

    seen: set[Path] = set()
    for candidate in checkpoint_candidates(runs_root, fold, epoch):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing checkpoint for fold={fold:02d}, epoch={epoch} under {runs_root}")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_checkpoints(
    runs_root: Path,
    folds: Iterable[int],
    epochs: Iterable[int],
    checkpoint_map: dict[tuple[int, int], Path] | None = None,
    fold_run_map: dict[int, Path] | None = None,
    allow_auto_discover: bool = True,
) -> list[CheckpointPlanRow]:
    rows: list[CheckpointPlanRow] = []
    for fold in folds:
        for epoch in epochs:
            try:
                checkpoint_path = resolve_checkpoint_path(
                    runs_root,
                    fold=fold,
                    epoch=epoch,
                    checkpoint_map=checkpoint_map,
                    fold_run_map=fold_run_map,
                    allow_auto_discover=allow_auto_discover,
                )
                rows.append(
                    CheckpointPlanRow(
                        fold=fold,
                        epoch=epoch,
                        checkpoint_path=checkpoint_path,
                        exists=True,
                        source="checkpoint_map" if checkpoint_map and (fold, epoch) in checkpoint_map else "discovered",
                    )
                )
            except FileNotFoundError:
                rows.append(
                    CheckpointPlanRow(fold=fold, epoch=epoch, checkpoint_path=None, exists=False, source="missing")
                )
    return rows


def add_epoch_rank_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["y_true"], []).append(row)

    output = [dict(row) for row in rows]
    by_identity = {id(original): copied for original, copied in zip(rows, output)}
    for label_rows in groups.values():
        sorted_rows = sorted(
            label_rows,
            key=lambda row: (-float(row["p_defect_raw"]), sample_id_for_row(row)),
        )
        n = len(sorted_rows)
        for rank, row in enumerate(sorted_rows, start=1):
            target = by_identity[id(row)]
            target["risk_rank_desc_within_label"] = str(rank)
            target["risk_rank_pct_desc_within_label"] = f"{rank / n:.10f}"
            target["rank_group_n"] = str(n)
    return output


def prediction_sidecar_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".manifest.json")


def count_prediction_csv(path: Path) -> tuple[int, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"Prediction CSV missing sample_id column: {path}")
        count = 0
        sample_ids: set[str] = set()
        for row in reader:
            sample_id = row.get("sample_id", "")
            if not sample_id:
                raise ValueError(f"Prediction CSV has blank sample_id: {path}")
            sample_ids.add(sample_id)
            count += 1
    return count, len(sample_ids)


def normalized_path_text(path: Path | str | None) -> str:
    if path is None or str(path) == "":
        return ""
    return str(Path(path).expanduser().resolve(strict=False)).casefold()


def csv_metadata_matches(
    path: Path,
    *,
    expected_checkpoint_id: str | None = None,
    expected_oof_fold: str | None = None,
    expected_epoch: int | str | None = None,
    expected_checkpoint_path: Path | str | None = None,
) -> bool:
    expected_epoch_text = "" if expected_epoch is None else str(int(expected_epoch))
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required: dict[str, str] = {}
        if expected_checkpoint_id is not None:
            required["checkpoint_id"] = expected_checkpoint_id
        if expected_oof_fold is not None:
            required["oof_fold"] = expected_oof_fold
        if expected_epoch is not None:
            required["epoch"] = expected_epoch_text
        if expected_checkpoint_path is not None:
            required["checkpoint_path"] = normalized_path_text(expected_checkpoint_path)

        missing = set(required) - fieldnames
        if missing:
            return False
        for row in reader:
            for key, expected_value in required.items():
                actual = row.get(key, "")
                if key == "checkpoint_path":
                    actual = normalized_path_text(actual)
                if actual != expected_value:
                    return False
    return True


def validate_prediction_rows(rows: list[dict[str, str]], *, expected_row_count: int) -> None:
    if len(rows) != expected_row_count:
        raise ValueError(f"Prediction row count mismatch: expected={expected_row_count}, actual={len(rows)}")
    sample_ids = [row.get("sample_id", "") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Prediction rows contain blank sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Prediction rows contain duplicate sample_id values")


def write_epoch_predictions_atomic(
    output_path: Path,
    rows: list[dict[str, str]],
    columns: Iterable[str],
    *,
    expected_row_count: int,
    manifest_payload: dict[str, object],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(output_path) + ".tmp")
    sidecar_path = prediction_sidecar_path(output_path)
    tmp_sidecar_path = Path(str(sidecar_path) + ".tmp")
    tmp_path.unlink(missing_ok=True)
    tmp_sidecar_path.unlink(missing_ok=True)
    try:
        validate_prediction_rows(rows, expected_row_count=expected_row_count)
        write_csv(tmp_path, rows, columns)
        row_count, unique_count = count_prediction_csv(tmp_path)
        if row_count != expected_row_count or unique_count != expected_row_count:
            raise ValueError(
                f"Atomic write validation failed for {tmp_path}: rows={row_count}, unique_sample_ids={unique_count}, expected={expected_row_count}"
            )
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "complete",
            "output_csv": str(output_path),
            "row_count": row_count,
            "unique_sample_ids": unique_count,
            **manifest_payload,
        }
        write_json(tmp_sidecar_path, payload)
        tmp_path.replace(output_path)
        tmp_sidecar_path.replace(sidecar_path)
        return sidecar_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        tmp_sidecar_path.unlink(missing_ok=True)
        raise


def completed_epoch_is_valid(
    output_path: Path,
    *,
    expected_row_count: int,
    expected_checkpoint_id: str | None = None,
    expected_oof_fold: str | None = None,
    expected_epoch: int | str | None = None,
    expected_checkpoint_path: Path | str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> bool:
    sidecar_path = prediction_sidecar_path(output_path)
    if not output_path.exists() or not sidecar_path.exists():
        return False
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            sidecar = json.load(f)
        if sidecar.get("status") != "complete":
            return False
        if int(sidecar.get("row_count", -1)) != expected_row_count:
            return False
        if expected_checkpoint_id is not None and sidecar.get("checkpoint_id", "") != expected_checkpoint_id:
            return False
        if expected_oof_fold is not None and sidecar.get("oof_fold", "") != expected_oof_fold:
            return False
        if expected_epoch is not None and str(int(sidecar.get("epoch", -1))) != str(int(expected_epoch)):
            return False
        if expected_checkpoint_path is not None and normalized_path_text(sidecar.get("checkpoint_path", "")) != normalized_path_text(
            expected_checkpoint_path
        ):
            return False
        if expected_checkpoint_sha256 is not None and sidecar.get("checkpoint_sha256", "") != expected_checkpoint_sha256:
            return False
        row_count, unique_count = count_prediction_csv(output_path)
        if not csv_metadata_matches(
            output_path,
            expected_checkpoint_id=expected_checkpoint_id,
            expected_oof_fold=expected_oof_fold,
            expected_epoch=expected_epoch,
            expected_checkpoint_path=expected_checkpoint_path,
        ):
            return False
    except Exception:
        return False
    return row_count == expected_row_count and unique_count == expected_row_count


def resource_snapshot() -> dict[str, str]:
    snapshot = {
        "rss_mb": "",
        "cuda_allocated_mb": "",
        "cuda_reserved_mb": "",
    }
    try:
        import psutil

        snapshot["rss_mb"] = f"{psutil.Process().memory_info().rss / (1024 * 1024):.2f}"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = f"{torch.cuda.memory_allocated() / (1024 * 1024):.2f}"
            snapshot["cuda_reserved_mb"] = f"{torch.cuda.memory_reserved() / (1024 * 1024):.2f}"
    except Exception:
        pass
    return snapshot


def append_resource_log(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "created_at",
        "checkpoint_id",
        "oof_fold",
        "epoch",
        "status",
        "duration_sec",
        "rss_mb_before_cleanup",
        "cuda_allocated_mb_before_cleanup",
        "cuda_reserved_mb_before_cleanup",
        "rss_mb_after_cleanup",
        "cuda_allocated_mb_after_cleanup",
        "cuda_reserved_mb_after_cleanup",
        "rss_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "error",
    )
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def enrich_epoch_rows(
    rows: list[dict[str, str]],
    *,
    fold: int,
    epoch: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        y_true = int(row["y_true"])
        p_defect = float(row["p_defect_raw"])
        signed_margin = (2 * y_true - 1) * (p_defect - 0.5)
        item = dict(row)
        item.update(
            {
                "sample_id": sample_id_for_row(row),
                "checkpoint_id": checkpoint_id(fold, epoch),
                "oof_fold": f"{fold:02d}",
                "human_fold": str(fold + 1),
                "epoch": str(epoch),
                "raw_margin_signed": f"{signed_margin:.10f}",
                "raw_abs_margin": f"{abs(p_defect - 0.5):.10f}",
                "raw_cross_entropy": f"{cross_entropy(y_true, p_defect):.10f}",
                "raw_uncertainty": f"{uncertainty(p_defect):.10f}",
                "raw_logit": f"{logit(p_defect):.10f}",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
        enriched.append(item)
    return add_epoch_rank_columns(enriched)


def discover_manifest_dir(oof_root: Path, fold: int) -> Path:
    candidates = [
        oof_root / "folds" / f"fold_{fold:02d}" / "manifests",
        oof_root / f"fold_{fold:02d}" / "manifests",
        oof_root / f"fold_{fold + 1:02d}" / "manifests",
    ]
    for candidate in candidates:
        if (candidate / "val_model_manifest.csv").exists() and (candidate / "normal_val_model_manifest.csv").exists():
            return candidate
    raise FileNotFoundError(f"Missing holdout manifests for fold={fold:02d} under {oof_root}")


def fold_output_dir(dynamics_root: Path, fold: int) -> Path:
    return dynamics_root / f"fold_{fold:02d}"


def epoch_output_path(dynamics_root: Path, fold: int, epoch: int) -> Path:
    return fold_output_dir(dynamics_root, fold) / f"epoch_{epoch:03d}_predictions.csv"


def write_plan(plan_path: Path, plan_rows: list[CheckpointPlanRow]) -> None:
    write_csv(
        plan_path,
        [row.as_dict() for row in plan_rows],
        ("oof_fold", "human_fold", "epoch", "checkpoint_id", "checkpoint_path", "exists", "source"),
    )


def export_dynamics(args: argparse.Namespace) -> int:
    root = repo_root()
    fam_root = family_root(repo=root, override=args.family_root)
    initialize_experiment_family_layout(fam_root)
    active_root = experiment_root(fam_root, ACTIVE_EXPERIMENT_ID)
    dynamics_stage_root = stage_root(active_root, "01_oof_dynamics")

    dataset_root = resolve_path(root, args.dataset_root, DEFAULT_DATASET_ROOT)
    oof_root = resolve_path(root, args.oof_root, DEFAULT_OOF_ROOT)
    runs_root = resolve_path(root, args.runs_root, DEFAULT_RUNS_ROOT)
    yolo_root = resolve_path(root, args.yolo_root, DEFAULT_YOLO_ROOT)
    checkpoint_map = read_checkpoint_map(args.checkpoint_map)
    fold_run_map = read_fold_run_map(args.fold_run_map)
    folds = parse_fold_spec(args.folds, base=args.fold_base)
    epochs = parse_epoch_spec(args.epochs)
    job_id = args.job_id or ("dry_run" if args.dry_run else "")
    validate_export_request(
        dry_run=args.dry_run,
        job_id=job_id,
        folds=folds,
        fold_run_map=fold_run_map,
        allow_auto_discover=args.allow_auto_discover,
    )
    dynamics_root = job_dynamics_root(dynamics_stage_root, job_id)

    plan_rows = plan_checkpoints(
        runs_root,
        folds,
        epochs,
        checkpoint_map,
        fold_run_map=fold_run_map,
        allow_auto_discover=args.dry_run or args.allow_auto_discover,
    )
    write_plan(dynamics_root / "checkpoint_plan.csv", plan_rows)
    missing = [row for row in plan_rows if not row.exists]
    print(f"checkpoint_plan={dynamics_root / 'checkpoint_plan.csv'}")
    print(f"planned_checkpoints={len(plan_rows)} missing={len(missing)}")
    if args.dry_run:
        write_json(
            dynamics_root / "dry_run_summary.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_root": str(dataset_root),
                "oof_root": str(oof_root),
                "runs_root": str(runs_root),
                "folds": folds,
                "epochs": epochs,
                "planned_checkpoints": len(plan_rows),
                "missing_checkpoints": len(missing),
            },
        )
        return 0 if not missing or args.allow_missing_checkpoints else 2
    if missing and not args.allow_missing_checkpoints:
        raise FileNotFoundError(
            f"Missing {len(missing)} checkpoints. Inspect {dynamics_root / 'checkpoint_plan.csv'} or use --allow-missing-checkpoints."
        )

    YOLO = ensure_yolo_import(yolo_root)
    artifact_rows: list[dict[str, str]] = []
    for fold in folds:
        manifest_dir = discover_manifest_dir(oof_root, fold)
        placeholder_job = FoldJob(fold=fold, manifest_dir=manifest_dir, weights=Path("."), run_dir=runs_root)
        records = load_fold_holdout_records(placeholder_job, dataset_root)
        print(f"fold_{fold:02d}: holdout_records={len(records)} manifest_dir={manifest_dir}", flush=True)

        for epoch in epochs:
            try:
                checkpoint_path = resolve_checkpoint_path(
                    runs_root,
                    fold=fold,
                    epoch=epoch,
                    checkpoint_map=checkpoint_map,
                    fold_run_map=fold_run_map,
                    allow_auto_discover=args.allow_auto_discover,
                )
            except FileNotFoundError:
                if args.allow_missing_checkpoints:
                    print(f"skip missing fold={fold:02d} epoch={epoch}", flush=True)
                    continue
                raise
            output_path = epoch_output_path(dynamics_root, fold, epoch)
            expected_row_count = len(records)
            if args.resume and (output_path.exists() or prediction_sidecar_path(output_path).exists()):
                if completed_epoch_is_valid(
                    output_path,
                    expected_row_count=expected_row_count,
                    expected_checkpoint_id=checkpoint_id(fold, epoch),
                    expected_oof_fold=f"{fold:02d}",
                    expected_epoch=epoch,
                    expected_checkpoint_path=checkpoint_path,
                ):
                    print(f"resume skip complete {output_path}", flush=True)
                    artifact_rows.append(
                        {
                            "relative_path": output_path.relative_to(active_root).as_posix(),
                            "fold": f"{fold:02d}",
                            "epoch": str(epoch),
                            "row_count": str(expected_row_count),
                            "checkpoint_path": str(checkpoint_path),
                            "duration_sec": "0.000",
                            "status": "skipped_resume",
                        }
                    )
                    continue
                raise ValueError(f"Cannot resume from incomplete or invalid epoch output: {output_path}")
            if output_path.exists() and not args.exist_ok:
                raise FileExistsError(f"Output already exists: {output_path}. Use --exist-ok to overwrite.")

            started = time.time()
            print(f"predict {checkpoint_id(fold, epoch)} weights={checkpoint_path}", flush=True)
            model = None
            prediction_rows: list[dict[str, str]] | None = None
            enriched: list[dict[str, str]] | None = None
            status = "failed"
            error = ""
            try:
                model = YOLO(str(checkpoint_path))
                cfg = EvalConfig(
                    weights=checkpoint_path,
                    run_name=checkpoint_id(fold, epoch),
                    splits=("oof_holdout",),
                    seed=args.seed,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    limit_per_class=None,
                    target_recall=0.995,
                    deployment_defect_prevalence=0.1,
                    dry_run=False,
                    exist_ok=True,
                )
                prediction_rows = predict_records(model, records, cfg)
                sha = file_sha256(checkpoint_path) if args.hash_checkpoints else ""
                enriched = enrich_epoch_rows(
                    prediction_rows,
                    fold=fold,
                    epoch=epoch,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=sha,
                )
                write_epoch_predictions_atomic(
                    output_path,
                    enriched,
                    DYNAMICS_COLUMNS,
                    expected_row_count=expected_row_count,
                    manifest_payload={
                        "checkpoint_id": checkpoint_id(fold, epoch),
                        "oof_fold": f"{fold:02d}",
                        "epoch": epoch,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha,
                    },
                )
                status = "complete"
                artifact_rows.append(
                    {
                        "relative_path": output_path.relative_to(active_root).as_posix(),
                        "fold": f"{fold:02d}",
                        "epoch": str(epoch),
                        "row_count": str(len(enriched)),
                        "checkpoint_path": str(checkpoint_path),
                        "duration_sec": f"{time.time() - started:.3f}",
                        "status": status,
                    }
                )
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                del model
                del prediction_rows
                del enriched
                before_cleanup = resource_snapshot()
                release_torch_memory()
                gc.collect()
                after_cleanup = resource_snapshot()
                append_resource_log(
                    dynamics_root / "resource_log.csv",
                    {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "checkpoint_id": checkpoint_id(fold, epoch),
                        "oof_fold": f"{fold:02d}",
                        "epoch": str(epoch),
                        "status": status,
                        "duration_sec": f"{time.time() - started:.3f}",
                        "error": error,
                        "rss_mb_before_cleanup": before_cleanup.get("rss_mb", ""),
                        "cuda_allocated_mb_before_cleanup": before_cleanup.get("cuda_allocated_mb", ""),
                        "cuda_reserved_mb_before_cleanup": before_cleanup.get("cuda_reserved_mb", ""),
                        "rss_mb_after_cleanup": after_cleanup.get("rss_mb", ""),
                        "cuda_allocated_mb_after_cleanup": after_cleanup.get("cuda_allocated_mb", ""),
                        "cuda_reserved_mb_after_cleanup": after_cleanup.get("cuda_reserved_mb", ""),
                        **after_cleanup,
                    },
                )

    write_csv(
        dynamics_root / "artifact_manifest.csv",
        artifact_rows,
        ("relative_path", "fold", "epoch", "row_count", "checkpoint_path", "duration_sec", "status"),
    )
    write_json(
        dynamics_root / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "family_root": str(fam_root),
            "active_experiment_root": str(active_root),
            "job_id": job_id,
            "job_root": str(dynamics_root),
            "dataset_root": str(dataset_root),
            "oof_root": str(oof_root),
            "runs_root": str(runs_root),
            "yolo_root": str(yolo_root),
            "folds": folds,
            "epochs": epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "hash_checkpoints": bool(args.hash_checkpoints),
            "allow_auto_discover": bool(args.allow_auto_discover),
            "resume": bool(args.resume),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export strict OOF checkpoint training dynamics for Stage-1 sample value.")
    parser.add_argument("--family-root", type=Path, default=None, help="Override grouped experiment family root.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--oof-root", type=Path, default=None, help="OOF fold manifest root.")
    parser.add_argument("--runs-root", type=Path, default=None, help="OOF fold training/checkpoint root.")
    parser.add_argument("--checkpoint-map", type=Path, default=None, help="CSV with oof_fold,epoch,checkpoint_path.")
    parser.add_argument("--fold-run-map", type=Path, default=None, help="CSV with oof_fold,run_dir for exact fold run binding.")
    parser.add_argument("--yolo-root", type=Path, default=None)
    parser.add_argument("--job-id", default="", help="Required for formal non-dry-run export; output goes under jobs/<job-id>.")
    parser.add_argument("--folds", default="0-9", help="Fold spec, e.g. 0-3 or 1-4 with --fold-base 1.")
    parser.add_argument("--fold-base", type=int, default=0, choices=(0, 1))
    parser.add_argument("--epochs", default="1-200", help="Epoch spec, e.g. 1-200 or 1,50,100,200.")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    parser.add_argument("--allow-auto-discover", action="store_true", help="Allow recursive checkpoint discovery without --fold-run-map.")
    parser.add_argument("--resume", action="store_true", help="Skip completed epoch CSVs only when CSV and sidecar validate.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true", help="Compute sha256 for each checkpoint; slower on large archives.")
    return parser.parse_args()


def main() -> int:
    try:
        return export_dynamics(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
