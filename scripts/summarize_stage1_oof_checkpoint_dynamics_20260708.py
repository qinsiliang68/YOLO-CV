from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from scripts.evaluate_stage1_cls_gate import cross_entropy, write_csv, write_json
from scripts.export_stage1_oof_checkpoint_dynamics_20260708 import (
    checkpoint_id,
    completed_epoch_is_valid,
    parse_epoch_spec,
    prediction_sidecar_path,
    sample_id_for_row,
)
from scripts.predict_stage1_oof_folds_20260621 import parse_fold_spec
from scripts.stage1_sample_value_experiment_layout_20260708 import (
    ACTIVE_EXPERIMENT_ID,
    experiment_root,
    family_root,
    initialize_experiment_family_layout,
    stage_root,
)


SUMMARY_COLUMNS = (
    "sample_id",
    "y_true",
    "oof_fold",
    "epoch_count",
    "first_epoch",
    "last_epoch",
    "p_defect_start",
    "p_defect_end",
    "p_defect_trend",
    "mean_p_defect",
    "std_p_defect",
    "min_p_defect",
    "max_p_defect",
    "mean_loss",
    "loss_auc_mean",
    "max_loss",
    "mean_margin_signed",
    "mean_abs_margin",
    "min_margin_signed",
    "correct_count",
    "correct_rate",
    "final_correct",
    "first_learned_epoch",
    "last_wrong_epoch",
    "forgetting_count",
    "dynamic_bucket",
)


@dataclass
class TrajectoryAccumulator:
    sample_id: str
    y_true: int | None = None
    oof_fold: str = ""
    epoch_count: int = 0
    first_epoch: int | None = None
    last_epoch: int | None = None
    p_start: float | None = None
    p_end: float | None = None
    sum_p: float = 0.0
    sum_p2: float = 0.0
    min_p: float = 1.0
    max_p: float = 0.0
    sum_loss: float = 0.0
    max_loss: float = 0.0
    sum_margin: float = 0.0
    sum_abs_margin: float = 0.0
    min_margin: float = math.inf
    correct_count: int = 0
    final_correct: int = 0
    first_learned_epoch: int | None = None
    last_wrong_epoch: int | None = None
    forgetting_count: int = 0
    prev_correct: int | None = None
    epochs_seen: set[int] = field(default_factory=set)

    def add(self, row: dict[str, str]) -> None:
        epoch = int(row["epoch"])
        y_true = int(row["y_true"])
        p_defect = float(row["p_defect_raw"])
        correct = int(row.get("raw_correct", "1" if (p_defect >= 0.5) == (y_true == 1) else "0"))
        loss = float(row.get("raw_cross_entropy") or cross_entropy(y_true, p_defect))
        margin = float(row.get("raw_margin_signed") or ((2 * y_true - 1) * (p_defect - 0.5)))

        if self.y_true is None:
            self.y_true = y_true
        elif self.y_true != y_true:
            raise ValueError(f"Sample {self.sample_id} has conflicting y_true values")

        fold = row.get("oof_fold", "")
        if self.oof_fold and fold and self.oof_fold != fold:
            raise ValueError(f"Sample {self.sample_id} appears in multiple OOF folds")
        if fold:
            self.oof_fold = fold

        if epoch in self.epochs_seen:
            raise ValueError(f"Sample {self.sample_id} has duplicate epoch {epoch}")
        self.epochs_seen.add(epoch)

        if self.first_epoch is None or epoch < self.first_epoch:
            self.first_epoch = epoch
            self.p_start = p_defect
        if self.last_epoch is None or epoch >= self.last_epoch:
            self.last_epoch = epoch
            self.p_end = p_defect
            self.final_correct = correct

        self.epoch_count += 1
        self.sum_p += p_defect
        self.sum_p2 += p_defect * p_defect
        self.min_p = min(self.min_p, p_defect)
        self.max_p = max(self.max_p, p_defect)
        self.sum_loss += loss
        self.max_loss = max(self.max_loss, loss)
        self.sum_margin += margin
        self.sum_abs_margin += abs(p_defect - 0.5)
        self.min_margin = min(self.min_margin, margin)
        self.correct_count += correct

        if correct:
            if self.first_learned_epoch is None:
                self.first_learned_epoch = epoch
        else:
            self.last_wrong_epoch = epoch
        if self.prev_correct == 1 and correct == 0:
            self.forgetting_count += 1
        self.prev_correct = correct

    def summary(self) -> dict[str, str]:
        if self.epoch_count == 0 or self.y_true is None:
            raise ValueError(f"Sample {self.sample_id} has no epochs")
        mean_p = self.sum_p / self.epoch_count
        variance_p = max(0.0, self.sum_p2 / self.epoch_count - mean_p * mean_p)
        mean_loss = self.sum_loss / self.epoch_count
        mean_margin = self.sum_margin / self.epoch_count
        mean_abs_margin = self.sum_abs_margin / self.epoch_count
        correct_rate = self.correct_count / self.epoch_count
        row = {
            "sample_id": self.sample_id,
            "y_true": str(self.y_true),
            "oof_fold": self.oof_fold,
            "epoch_count": str(self.epoch_count),
            "first_epoch": "" if self.first_epoch is None else str(self.first_epoch),
            "last_epoch": "" if self.last_epoch is None else str(self.last_epoch),
            "p_defect_start": f"{self.p_start or 0.0:.10f}",
            "p_defect_end": f"{self.p_end or 0.0:.10f}",
            "p_defect_trend": f"{(self.p_end or 0.0) - (self.p_start or 0.0):.10f}",
            "mean_p_defect": f"{mean_p:.10f}",
            "std_p_defect": f"{math.sqrt(variance_p):.10f}",
            "min_p_defect": f"{self.min_p:.10f}",
            "max_p_defect": f"{self.max_p:.10f}",
            "mean_loss": f"{mean_loss:.10f}",
            "loss_auc_mean": f"{mean_loss:.10f}",
            "max_loss": f"{self.max_loss:.10f}",
            "mean_margin_signed": f"{mean_margin:.10f}",
            "mean_abs_margin": f"{mean_abs_margin:.10f}",
            "min_margin_signed": f"{self.min_margin:.10f}",
            "correct_count": str(self.correct_count),
            "correct_rate": f"{correct_rate:.10f}",
            "final_correct": str(self.final_correct),
            "first_learned_epoch": "" if self.first_learned_epoch is None else str(self.first_learned_epoch),
            "last_wrong_epoch": "" if self.last_wrong_epoch is None else str(self.last_wrong_epoch),
            "forgetting_count": str(self.forgetting_count),
        }
        row["dynamic_bucket"] = bucket_for_summary(row)
        return row


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(value)


def bucket_for_summary(row: dict[str, str]) -> str:
    correct_rate = safe_float(row, "correct_rate")
    mean_loss = safe_float(row, "mean_loss")
    final_correct = safe_int(row, "final_correct")
    forgetting_count = safe_int(row, "forgetting_count")
    mean_abs_margin = safe_float(row, "mean_abs_margin", default=1.0)
    first_learned = row.get("first_learned_epoch", "")
    last_wrong = row.get("last_wrong_epoch", "")

    if correct_rate >= 0.99 and mean_loss <= 0.10 and forgetting_count == 0:
        return "easy_clean"
    if forgetting_count > 0 and mean_abs_margin <= 0.15:
        return "unstable_boundary"
    if correct_rate <= 0.10 and final_correct == 0 and mean_loss >= 1.50:
        return "possible_noise_or_label_issue"
    if final_correct == 1 and first_learned and last_wrong:
        return "learnable_hard"
    if final_correct == 0 and correct_rate < 0.50:
        return "persistent_hard"
    if mean_abs_margin <= 0.10:
        return "boundary_like"
    return "ordinary"


def summarize_trajectory(sample_id: str, rows: Iterable[dict[str, str]]) -> dict[str, str]:
    accumulator = TrajectoryAccumulator(sample_id)
    for row in sorted(rows, key=lambda item: int(item["epoch"])):
        accumulator.add(row)
    return accumulator.summary()


def read_dynamics_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "y_true", "p_defect_raw"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            yield row


def collect_prediction_files(dynamics_root: Path) -> list[Path]:
    return sorted(dynamics_root.glob("jobs/*/fold_*/epoch_*_predictions.csv"))


def prediction_file_identity(path: Path) -> tuple[int, int]:
    fold_match = re.fullmatch(r"fold_(\d+)", path.parent.name)
    epoch_match = re.fullmatch(r"epoch_(\d+)_predictions\.csv", path.name)
    if not fold_match or not epoch_match:
        raise ValueError(f"Unexpected prediction path shape: {path}")
    return int(fold_match.group(1)), int(epoch_match.group(1))


def validate_prediction_file_for_summary(path: Path) -> None:
    fold, epoch = prediction_file_identity(path)
    sidecar_path = prediction_sidecar_path(path)
    if not sidecar_path.exists():
        raise ValueError(f"Prediction CSV missing sidecar manifest: {path}")
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            sidecar = json.load(f)
        expected_row_count = int(sidecar.get("row_count", -1))
    except Exception as exc:
        raise ValueError(f"Prediction CSV sidecar is unreadable: {sidecar_path}") from exc
    if expected_row_count < 0:
        raise ValueError(f"Prediction CSV sidecar missing row_count: {sidecar_path}")
    expected_checkpoint_path = sidecar.get("checkpoint_path") or None
    if not completed_epoch_is_valid(
        path,
        expected_row_count=expected_row_count,
        expected_checkpoint_id=checkpoint_id(fold, epoch),
        expected_oof_fold=f"{fold:02d}",
        expected_epoch=epoch,
        expected_checkpoint_path=expected_checkpoint_path,
    ):
        raise ValueError(f"Prediction CSV metadata mismatch or incomplete sidecar: {path}")


def write_summary_validation(output_root: Path, payload: dict[str, object]) -> None:
    write_json(output_root / "summary_validation.json", payload)


def validate_summary_inputs(
    files: list[Path],
    *,
    expected_folds: list[int],
    expected_epochs: list[int],
    output_root: Path,
    summaries: list[dict[str, str]] | None = None,
    sample_epoch_sets: dict[str, set[int]] | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    pairs: set[tuple[int, int]] = set()
    duplicates: list[str] = []
    for path in files:
        pair = prediction_file_identity(path)
        if pair in pairs:
            duplicates.append(f"fold_{pair[0]:02d}/epoch_{pair[1]:03d}")
        pairs.add(pair)
        try:
            validate_prediction_file_for_summary(path)
        except ValueError as exc:
            errors.append(str(exc))
    if duplicates:
        errors.append(f"Duplicate fold/epoch prediction files: {duplicates}")

    missing_pairs: list[str] = []
    if expected_folds and expected_epochs:
        expected_pairs = {(fold, epoch) for fold in expected_folds for epoch in expected_epochs}
        missing_pairs = [f"fold_{fold:02d}/epoch_{epoch:03d}" for fold, epoch in sorted(expected_pairs - pairs)]
        if missing_pairs:
            errors.append(f"Missing expected fold/epoch prediction files: {missing_pairs[:20]}")

    bad_epoch_count: list[str] = []
    bad_epoch_sets: list[str] = []
    if summaries is not None and expected_epochs:
        expected_count = len(expected_epochs)
        expected_set = set(expected_epochs)
        for row in summaries:
            if int(row["epoch_count"]) != expected_count:
                bad_epoch_count.append(f"{row['sample_id']}:{row['epoch_count']}")
            if sample_epoch_sets is not None:
                actual_set = sample_epoch_sets.get(row["sample_id"], set())
                if actual_set != expected_set:
                    preview = ",".join(str(epoch) for epoch in sorted(actual_set)[:10])
                    bad_epoch_sets.append(f"{row['sample_id']}:[{preview}]")
        if bad_epoch_count:
            errors.append(f"Samples with incomplete epoch_count: {bad_epoch_count[:20]}")
        if bad_epoch_sets:
            errors.append(f"Samples with incomplete epoch set: {bad_epoch_sets[:20]}")

    payload: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file_count": len(files),
        "expected_folds": [f"{fold:02d}" for fold in expected_folds],
        "expected_epochs": expected_epochs,
        "actual_pairs": [f"fold_{fold:02d}/epoch_{epoch:03d}" for fold, epoch in sorted(pairs)],
        "missing_pairs": missing_pairs,
        "duplicate_pairs": duplicates,
        "bad_epoch_count_samples": bad_epoch_count,
        "bad_epoch_set_samples": bad_epoch_sets,
        "sample_count": 0 if summaries is None else len(summaries),
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }
    write_summary_validation(output_root, payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def summarize_dynamics(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    fam_root = family_root(repo=root, override=args.family_root)
    initialize_experiment_family_layout(fam_root)
    active_root = experiment_root(fam_root, ACTIVE_EXPERIMENT_ID)
    dynamics_root = args.dynamics_root if args.dynamics_root is not None else stage_root(active_root, "01_oof_dynamics")
    dynamics_root = dynamics_root if dynamics_root.is_absolute() else root / dynamics_root
    output_root = args.output_root if args.output_root is not None else dynamics_root / "summary"
    output_root = output_root if output_root.is_absolute() else root / output_root
    expected_folds = parse_fold_spec(args.expected_folds, base=args.fold_base) if args.expected_folds else []
    expected_epochs = parse_epoch_spec(args.expected_epochs) if args.expected_epochs else []

    files = collect_prediction_files(dynamics_root)
    if not files:
        raise FileNotFoundError(f"No fold_*/epoch_*_predictions.csv files under {dynamics_root}")
    validate_summary_inputs(files, expected_folds=expected_folds, expected_epochs=expected_epochs, output_root=output_root)

    accumulators: dict[str, TrajectoryAccumulator] = {}
    artifact_rows: list[dict[str, str]] = []
    for path in files:
        count = 0
        for row in read_dynamics_rows(path):
            sample_id = row.get("sample_id") or sample_id_for_row(row)
            accumulators.setdefault(sample_id, TrajectoryAccumulator(sample_id)).add(row)
            count += 1
        artifact_rows.append({"relative_path": path.relative_to(dynamics_root).as_posix(), "row_count": str(count)})
        print(f"read {count} rows from {path}", flush=True)

    summaries = [acc.summary() for acc in accumulators.values()]
    summaries.sort(key=lambda row: (row["y_true"], row["sample_id"]))
    validate_summary_inputs(
        files,
        expected_folds=expected_folds,
        expected_epochs=expected_epochs,
        output_root=output_root,
        summaries=summaries,
        sample_epoch_sets={sample_id: set(acc.epochs_seen) for sample_id, acc in accumulators.items()},
    )
    write_csv(output_root / "sample_dynamics_summary.csv", summaries, SUMMARY_COLUMNS)

    bucket_counts = Counter(row["dynamic_bucket"] for row in summaries)
    bucket_rows = [
        {"dynamic_bucket": bucket, "sample_count": str(count)}
        for bucket, count in sorted(bucket_counts.items(), key=lambda item: item[0])
    ]
    write_csv(output_root / "dynamic_bucket_summary.csv", bucket_rows, ("dynamic_bucket", "sample_count"))
    write_csv(output_root / "summary_input_manifest.csv", artifact_rows, ("relative_path", "row_count"))
    write_json(
        output_root / "summary_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dynamics_root": str(dynamics_root),
            "output_root": str(output_root),
            "input_file_count": len(files),
            "sample_count": len(summaries),
            "expected_folds": expected_folds,
            "expected_epochs": expected_epochs,
            "bucket_counts": dict(bucket_counts),
        },
    )
    print(f"sample_summary={output_root / 'sample_dynamics_summary.csv'}")
    print(f"sample_count={len(summaries)} input_file_count={len(files)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage-1 OOF checkpoint dynamics into per-sample training signals.")
    parser.add_argument("--family-root", type=Path, default=None)
    parser.add_argument("--dynamics-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--expected-folds", default="", help="Expected fold spec, e.g. 0-9.")
    parser.add_argument("--expected-epochs", default="", help="Expected epoch spec, e.g. 1-200.")
    parser.add_argument("--fold-base", type=int, default=0, choices=(0, 1))
    return parser.parse_args()


def main() -> int:
    try:
        return summarize_dynamics(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
