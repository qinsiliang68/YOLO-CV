"""Canonical file and field coverage for the 240-run mechanism analysis.

The transfer inventory is the only authority for selecting attempts.  This
module deliberately inventories every file inside those attempts, including
operator recovery evidence and unregistered temporary remnants, so that later
analysis cannot silently narrow "all material" to an arbitrary required-file
subset.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import yaml


class DataCoverageError(RuntimeError):
    """Raised when canonical evidence coverage is incomplete or ambiguous."""


@dataclass(frozen=True)
class UsageRule:
    """First-match classification rule for one normalized file/field pair."""

    rule_id: str
    file_pattern: str
    field_pattern: str
    usage_role: str
    usage_status: str
    rationale: str


@dataclass(frozen=True)
class FieldCoverageResult:
    """Per-file schema signatures and aggregated file/field coverage."""

    file_schemas: pd.DataFrame
    field_coverage: pd.DataFrame


ALLOWED_USAGE_ROLES = frozenset(
    {
        "SCIENTIFIC_FEATURE",
        "OUTCOME_METRIC",
        "CONFOUND_CONTROL",
        "PAIRING_KEY",
        "LINEAGE_VALIDATION",
        "CONSTANT_PARAMETER",
        "DESCRIPTIVE_FIELD",
        "RECOMPUTED_FIELD",
        "MISSING_FIELD",
        "NOT_COLLECTED_FIELD",
    }
)

ALLOWED_USAGE_STATUSES = frozenset(
    {
        "ANALYZED",
        "AUDITED",
        "DOCUMENTED_CONSTANT",
        "DESCRIPTIVE_ONLY",
        "RECOMPUTED",
        "DOCUMENTED_MISSING",
        "NOT_TESTABLE",
        "EXCLUDED_WITH_REASON",
    }
)

EPOCH_TELEMETRY_COLUMNS = (
    "epoch",
    "time",
    "train/loss",
    "metrics/accuracy_top1",
    "metrics/accuracy_top5",
    "val/loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    "lr/pg3",
    "lr/pg4",
    "lr/pg5",
    "lr/pg6",
    "lr/pg7",
)

_TIMESTAMP = r"\d{8}T\d{4,6}"
_TOKEN = r"[0-9A-Za-z_]+"


def normalize_canonical_relative_path(relative_path: str | Path) -> str:
    """Collapse run-specific timestamps/tokens without hiding evidence types."""

    value = str(relative_path).replace("\\", "/")
    rules: tuple[tuple[str, str], ...] = (
        (
            rf"^02_logs/train_{_TIMESTAMP}_[0-9a-f]+(\.log(?:\.result\.json)?)$",
            r"02_logs/train_{TIMESTAMP}_{TOKEN}\1",
        ),
        (
            rf"^02_logs/gpu_usage_{_TIMESTAMP}_[0-9a-f]+\.csv$",
            "02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv",
        ),
        (
            rf"^02_logs/segment_{_TIMESTAMP}_[0-9a-f]+_(?:val_cal|val_op)(\.log(?:\.result\.json)?)$",
            r"02_logs/segment_{TIMESTAMP}_{TOKEN}_{SPLIT}\1",
        ),
        (
            r"^02_logs/checkpoint_probe_(?:best|last)_[0-9a-f]+(\.log(?:\.result\.json)?)$",
            r"02_logs/checkpoint_probe_{CHECKPOINT}_{TOKEN}\1",
        ),
        (
            r"^07_validation/checkpoint_probe_(?:best|last)_[0-9a-f]+\.json$",
            "07_validation/checkpoint_probe_{CHECKPOINT}_{TOKEN}.json",
        ),
        (
            rf"^02_logs/checkpoint_probe_manual_recovery_{_TIMESTAMP}\.(?:stderr|stdout)\.log$",
            "02_logs/checkpoint_probe_manual_recovery_{TIMESTAMP}.{STREAM}.log",
        ),
        (
            rf"^02_logs/checkpoint_probe_manual_recovery_{_TIMESTAMP}\.log$",
            "02_logs/checkpoint_probe_manual_recovery_{TIMESTAMP}.log",
        ),
        (
            rf"^07_validation/checkpoint_probe_manual_recovery_{_TIMESTAMP}\.json$",
            "07_validation/checkpoint_probe_manual_recovery_{TIMESTAMP}.json",
        ),
        (
            rf"^07_validation/(training_execution_audit\.pre_repair)_{_TIMESTAMP}\.json$",
            r"07_validation/\1_{TIMESTAMP}.json",
        ),
        (
            rf"^07_validation/(training_execution_audit_repair)_{_TIMESTAMP}\.json$",
            r"07_validation/\1_{TIMESTAMP}.json",
        ),
        (
            rf"^07_validation/(?:manual_recovery_audit|marker_FAILED_TRAIN_before_manual_recovery|status_before_manual_recovery)_{_TIMESTAMP}\.json$",
            lambda match: re.sub(
                _TIMESTAMP,
                "{TIMESTAMP}",
                match.group(0),
            ),
        ),
        (
            rf"^02_logs/\.train_{_TIMESTAMP}_[0-9a-f]+\.log\.result\.json\.{_TOKEN}\.tmp$",
            "02_logs/.train_{TIMESTAMP}_{TOKEN}.log.result.json.{TOKEN}.tmp",
        ),
    )
    for pattern, replacement in rules:
        if re.fullmatch(pattern, value):
            return re.sub(pattern, replacement, value)
    return value


def _canonical_attempt_path(extracted_root: Path, row: pd.Series) -> Path:
    return (
        extracted_root
        / f"stage1_gapvalue240_{row['package']}_upload"
        / "runs"
        / str(row["run_slot"])
        / str(row["attempt_id"])
    )


def build_canonical_file_inventory(
    extracted_root: str | Path,
    inventory_path: str | Path,
) -> pd.DataFrame:
    """Enumerate files only below attempts explicitly named by the inventory."""

    root = Path(extracted_root).resolve()
    inventory_file = Path(inventory_path).resolve()
    if not inventory_file.is_file():
        raise DataCoverageError(f"Missing canonical inventory: {inventory_file}")
    inventory = pd.read_csv(
        inventory_file,
        dtype={"run_slot": "string", "package": "string", "attempt_id": "string"},
        keep_default_na=False,
    )
    required = {"run_slot", "package", "attempt_id"}
    missing = required - set(inventory.columns)
    if missing:
        raise DataCoverageError(
            f"Canonical inventory missing columns: {sorted(missing)}"
        )
    if inventory["run_slot"].duplicated().any():
        duplicates = inventory.loc[
            inventory["run_slot"].duplicated(keep=False), "run_slot"
        ].tolist()
        raise DataCoverageError(f"Duplicate canonical run slots: {duplicates}")

    records: list[dict[str, Any]] = []
    for _, row in inventory.sort_values("run_slot").iterrows():
        attempt = _canonical_attempt_path(root, row)
        if not attempt.is_dir():
            raise DataCoverageError(f"Missing canonical attempt: {attempt}")
        artifact_manifest_path = attempt / "07_validation/artifact_manifest.csv"
        artifact_index: dict[str, dict[str, Any]] = {}
        if artifact_manifest_path.is_file():
            artifact_manifest = pd.read_csv(
                artifact_manifest_path,
                dtype={"relative_path": "string", "sha256": "string"},
                keep_default_na=False,
            )
            manifest_required = {"relative_path", "size_bytes", "sha256"}
            manifest_missing = manifest_required - set(artifact_manifest.columns)
            if manifest_missing:
                raise DataCoverageError(
                    f"Artifact manifest missing columns {sorted(manifest_missing)}: "
                    f"{artifact_manifest_path}"
                )
            if artifact_manifest["relative_path"].duplicated().any():
                raise DataCoverageError(
                    f"Duplicate artifact paths: {artifact_manifest_path}"
                )
            artifact_index = {
                str(item["relative_path"]): item
                for item in artifact_manifest.to_dict(orient="records")
            }
        for path in sorted(
            (candidate for candidate in attempt.rglob("*") if candidate.is_file()),
            key=lambda item: item.relative_to(attempt).as_posix(),
        ):
            relative = path.relative_to(attempt).as_posix()
            manifest_row = artifact_index.get(relative)
            actual_size = int(path.stat().st_size)
            records.append(
                {
                    "run_slot": str(row["run_slot"]),
                    "package": str(row["package"]),
                    "attempt_id": str(row["attempt_id"]),
                    "attempt_dir": str(attempt),
                    "relative_path": relative,
                    "normalized_path": normalize_canonical_relative_path(relative),
                    "size_bytes": actual_size,
                    "suffix": path.suffix.lower(),
                    "canonical_attempt": True,
                    "artifact_manifest_listed": manifest_row is not None,
                    "artifact_manifest_size_bytes": (
                        int(manifest_row["size_bytes"])
                        if manifest_row is not None
                        else pd.NA
                    ),
                    "artifact_manifest_sha256": (
                        str(manifest_row["sha256"]).upper()
                        if manifest_row is not None
                        else ""
                    ),
                    "artifact_manifest_size_match": (
                        actual_size == int(manifest_row["size_bytes"])
                        if manifest_row is not None
                        else pd.NA
                    ),
                }
            )
    columns = [
        "run_slot",
        "package",
        "attempt_id",
        "attempt_dir",
        "relative_path",
        "normalized_path",
        "size_bytes",
        "suffix",
        "canonical_attempt",
        "artifact_manifest_listed",
        "artifact_manifest_size_bytes",
        "artifact_manifest_sha256",
        "artifact_manifest_size_match",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _flatten_field_paths(value: Any, prefix: str = "") -> set[str]:
    fields: set[str] = set()
    if isinstance(value, dict):
        if not value and prefix:
            fields.add(prefix + "{}")
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            fields.update(_flatten_field_paths(item, child))
        return fields
    if isinstance(value, list):
        list_prefix = prefix + "[]"
        if not value:
            fields.add(list_prefix)
            return fields
        for item in value:
            if isinstance(item, (dict, list)):
                fields.update(_flatten_field_paths(item, list_prefix))
            else:
                fields.add(list_prefix)
        return fields
    fields.add(prefix or "<ROOT_SCALAR>")
    return fields


def extract_structured_field_inventory(
    path: str | Path,
    normalized_path: str,
) -> pd.DataFrame:
    """Extract schema paths without loading large CSV bodies into memory."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                fields = [str(column) for column in next(csv.reader(handle))]
        except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
            raise DataCoverageError(f"Cannot read CSV header {source}: {exc}") from exc
        kind = "CSV"
    elif suffix == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataCoverageError(f"Cannot read JSON {source}: {exc}") from exc
        fields = sorted(_flatten_field_paths(value))
        kind = "JSON"
    elif suffix in {".yaml", ".yml"}:
        try:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise DataCoverageError(f"Cannot read YAML {source}: {exc}") from exc
        fields = sorted(_flatten_field_paths(value))
        kind = "YAML"
    elif suffix == ".pt":
        fields = ["<BINARY_CHECKPOINT_PAYLOAD>"]
        kind = "CHECKPOINT"
    elif suffix == ".tmp":
        fields = ["<ORPHAN_TEMP_PAYLOAD>"]
        kind = "TEMPORARY"
    elif suffix == "":
        fields = ["<STATUS_MARKER>"]
        kind = "MARKER"
    else:
        fields = ["<UNSTRUCTURED_TEXT>"]
        kind = "TEXT"
    return pd.DataFrame(
        {
            "normalized_path": [normalized_path] * len(fields),
            "field_path": fields,
            "source_kind": [kind] * len(fields),
        }
    )


def build_field_coverage(
    file_inventory: pd.DataFrame,
    *,
    max_workers: int = 8,
) -> FieldCoverageResult:
    """Scan every inventoried file and aggregate schema coverage.

    Large CSV bodies are never loaded: :func:`extract_structured_field_inventory`
    reads only the first CSV record.  Threading is used only for filesystem I/O;
    results are aggregated deterministically after all evidence has been read.
    """

    required = {
        "run_slot",
        "attempt_dir",
        "relative_path",
        "normalized_path",
    }
    missing = required - set(file_inventory.columns)
    if missing:
        raise DataCoverageError(
            f"File inventory missing columns: {sorted(missing)}"
        )
    if max_workers < 1:
        raise DataCoverageError("max_workers must be positive")

    source_rows = file_inventory.reset_index(drop=True).to_dict(orient="records")

    def inspect(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        path = Path(str(row["attempt_dir"])) / str(row["relative_path"])
        fields = extract_structured_field_inventory(
            path,
            str(row["normalized_path"]),
        )
        field_names = sorted(fields["field_path"].astype(str).tolist())
        signature = hashlib.sha256("\0".join(field_names).encode("utf-8")).hexdigest().upper()
        source_kind = (
            str(fields["source_kind"].iloc[0]) if not fields.empty else "EMPTY"
        )
        file_key = f"{row['run_slot']}::{row['relative_path']}"
        file_record = {
            "file_key": file_key,
            "run_slot": str(row["run_slot"]),
            "relative_path": str(row["relative_path"]),
            "normalized_path": str(row["normalized_path"]),
            "source_kind": source_kind,
            "field_count": len(field_names),
            "schema_signature": signature,
        }
        instances = [
            {
                "file_key": file_key,
                "run_slot": str(row["run_slot"]),
                "normalized_path": str(row["normalized_path"]),
                "field_path": field,
                "source_kind": source_kind,
            }
            for field in field_names
        ]
        return file_record, instances

    if max_workers == 1:
        inspected = [inspect(row) for row in source_rows]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            inspected = list(executor.map(inspect, source_rows))

    file_records = [item[0] for item in inspected]
    field_records = [record for item in inspected for record in item[1]]
    file_schemas = pd.DataFrame(file_records).sort_values(
        ["run_slot", "relative_path"], ignore_index=True
    )
    instances = pd.DataFrame(field_records)
    if instances.empty:
        field_coverage = pd.DataFrame(
            columns=[
                "normalized_path",
                "field_path",
                "source_kind",
                "files_observed",
                "runs_observed",
                "files_expected",
                "runs_expected",
                "schema_variant_count",
            ]
        )
    else:
        observed = (
            instances.groupby(
                ["normalized_path", "field_path", "source_kind"],
                dropna=False,
                sort=True,
            )
            .agg(
                files_observed=("file_key", "nunique"),
                runs_observed=("run_slot", "nunique"),
            )
            .reset_index()
        )
        expected = (
            file_schemas.groupby("normalized_path", sort=True)
            .agg(
                files_expected=("file_key", "nunique"),
                runs_expected=("run_slot", "nunique"),
                schema_variant_count=("schema_signature", "nunique"),
            )
            .reset_index()
        )
        field_coverage = observed.merge(
            expected,
            on="normalized_path",
            how="left",
            validate="many_to_one",
        ).sort_values(["normalized_path", "field_path"], ignore_index=True)
    return FieldCoverageResult(
        file_schemas=file_schemas,
        field_coverage=field_coverage,
    )


def audit_epoch_grid(
    file_inventory: pd.DataFrame,
    canonical_inventory: pd.DataFrame,
    *,
    expected_runs: int = 240,
    expected_epochs: int = 200,
) -> pd.DataFrame:
    """Load and validate the complete saved 240x200 telemetry grid."""

    required_file = {
        "run_slot",
        "attempt_dir",
        "relative_path",
        "normalized_path",
    }
    missing_file = required_file - set(file_inventory.columns)
    if missing_file:
        raise DataCoverageError(
            f"File inventory missing columns: {sorted(missing_file)}"
        )
    if "run_slot" not in canonical_inventory.columns:
        raise DataCoverageError("Canonical inventory missing run_slot")
    if len(canonical_inventory) != expected_runs:
        raise DataCoverageError(
            f"Expected {expected_runs} canonical inventory rows, "
            f"found {len(canonical_inventory)}"
        )
    if canonical_inventory["run_slot"].duplicated().any():
        raise DataCoverageError("Canonical inventory contains duplicate run_slot")

    epoch_files = file_inventory.loc[
        file_inventory["normalized_path"].astype(str)
        == "02_logs/epoch_training_metrics.csv"
    ].copy()
    if len(epoch_files) != expected_runs:
        raise DataCoverageError(
            f"Expected {expected_runs} epoch telemetry files, found {len(epoch_files)}"
        )
    if epoch_files["run_slot"].duplicated().any():
        raise DataCoverageError("More than one epoch telemetry file for a run")
    if set(epoch_files["run_slot"].astype(str)) != set(
        canonical_inventory["run_slot"].astype(str)
    ):
        raise DataCoverageError("Epoch telemetry run-slot set differs from inventory")

    frames: list[pd.DataFrame] = []
    required_columns = set(EPOCH_TELEMETRY_COLUMNS)
    for row in epoch_files.sort_values("run_slot").itertuples(index=False):
        path = Path(str(row.attempt_dir)) / str(row.relative_path)
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise DataCoverageError(
                f"{row.run_slot} missing telemetry columns: {sorted(missing)}"
            )
        unexpected = set(frame.columns) - required_columns
        if unexpected:
            raise DataCoverageError(
                f"{row.run_slot} has unreviewed telemetry columns: {sorted(unexpected)}"
            )
        if len(frame) != expected_epochs:
            raise DataCoverageError(
                f"{row.run_slot} expected {expected_epochs} epochs, found {len(frame)}"
            )
        numeric = frame.loc[:, EPOCH_TELEMETRY_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )
        if not all(
            math.isfinite(float(value))
            for value in numeric.to_numpy().reshape(-1)
        ):
            raise DataCoverageError(f"{row.run_slot} telemetry contains NaN/Inf")
        epochs = numeric["epoch"].astype(int)
        if epochs.tolist() != list(range(1, expected_epochs + 1)):
            raise DataCoverageError(
                f"{row.run_slot} epoch identity is not exactly 1..{expected_epochs}"
            )
        numeric.insert(0, "run_slot", str(row.run_slot))
        frames.append(numeric)

    curves = pd.concat(frames, ignore_index=True)
    metadata = canonical_inventory.copy()
    metadata["run_slot"] = metadata["run_slot"].astype(str)
    duplicate_columns = [
        column for column in metadata.columns if column in curves.columns and column != "run_slot"
    ]
    if duplicate_columns:
        metadata = metadata.drop(columns=duplicate_columns)
    curves = curves.merge(
        metadata,
        on="run_slot",
        how="left",
        validate="many_to_one",
    )
    if len(curves) != expected_runs * expected_epochs:
        raise DataCoverageError(
            f"Expected {expected_runs * expected_epochs} epoch rows, found {len(curves)}"
        )
    return curves


def assert_complete_usage_ledger(ledger: pd.DataFrame) -> dict[str, int]:
    """Enforce the Goal's zero-unreviewed/zero-silent-drop completion gate."""

    required = {"normalized_path", "field_path", "usage_role", "usage_status"}
    missing = required - set(ledger.columns)
    if missing:
        raise DataCoverageError(f"Usage ledger missing columns: {sorted(missing)}")
    duplicate = ledger.duplicated(["normalized_path", "field_path"], keep=False)
    if duplicate.any():
        examples = ledger.loc[
            duplicate, ["normalized_path", "field_path"]
        ].drop_duplicates()
        raise DataCoverageError(
            "Usage ledger contains duplicate ledger identities: "
            + examples.head(10).to_dict(orient="records").__repr__()
        )

    roles = ledger["usage_role"].fillna("").astype(str).str.strip().str.upper()
    statuses = (
        ledger["usage_status"].fillna("").astype(str).str.strip().str.upper()
    )
    unreviewed = int(roles.isin({"", "UNREVIEWED"}).sum())
    unclassified = int((~roles.isin(ALLOWED_USAGE_ROLES | {"UNREVIEWED"})).sum())
    silently_dropped = int(
        (statuses.isin({"", "SILENTLY_DROPPED"}) | ~statuses.isin(
            ALLOWED_USAGE_STATUSES | {"SILENTLY_DROPPED"}
        )).sum()
    )
    summary = {
        "UNREVIEWED": unreviewed,
        "UNCLASSIFIED": unclassified,
        "SILENTLY_DROPPED": silently_dropped,
    }
    if any(summary.values()):
        raise DataCoverageError(
            "Incomplete DATA_USAGE_LEDGER: "
            + ", ".join(f"{key}={value}" for key, value in summary.items())
        )
    return summary


def apply_usage_rules(
    field_inventory: pd.DataFrame,
    rules: Iterable[UsageRule],
) -> pd.DataFrame:
    """Classify every source field while keeping unmatched evidence visible.

    Rules are intentionally first-match.  A narrow field rule can therefore
    override a later file-wide fallback, while any genuinely new evidence is
    emitted as ``UNREVIEWED/SILENTLY_DROPPED`` and fails the completion gate.
    """

    required = {"normalized_path", "field_path"}
    missing = required - set(field_inventory.columns)
    if missing:
        raise DataCoverageError(
            f"Field inventory missing columns: {sorted(missing)}"
        )
    compiled: list[tuple[UsageRule, re.Pattern[str], re.Pattern[str]]] = []
    seen_rule_ids: set[str] = set()
    for rule in rules:
        if rule.rule_id in seen_rule_ids:
            raise DataCoverageError(f"Duplicate usage rule id: {rule.rule_id}")
        seen_rule_ids.add(rule.rule_id)
        role = rule.usage_role.strip().upper()
        status = rule.usage_status.strip().upper()
        if role not in ALLOWED_USAGE_ROLES:
            raise DataCoverageError(
                f"Rule {rule.rule_id} has invalid usage role: {rule.usage_role}"
            )
        if status not in ALLOWED_USAGE_STATUSES:
            raise DataCoverageError(
                f"Rule {rule.rule_id} has invalid usage status: {rule.usage_status}"
            )
        compiled.append(
            (
                rule,
                re.compile(rule.file_pattern),
                re.compile(rule.field_pattern),
            )
        )

    output = field_inventory.copy()
    output["usage_role"] = "UNREVIEWED"
    output["usage_status"] = "SILENTLY_DROPPED"
    output["matched_rule_id"] = ""
    output["rationale"] = "No usage rule matched this file/field evidence."
    for index, row in output.iterrows():
        path = str(row["normalized_path"])
        field = str(row["field_path"])
        for rule, file_pattern, field_pattern in compiled:
            if file_pattern.fullmatch(path) and field_pattern.fullmatch(field):
                output.at[index, "usage_role"] = rule.usage_role.strip().upper()
                output.at[index, "usage_status"] = rule.usage_status.strip().upper()
                output.at[index, "matched_rule_id"] = rule.rule_id
                output.at[index, "rationale"] = rule.rationale
                break
    return output


def default_usage_rules() -> list[UsageRule]:
    """Return the frozen first-pass role contract for canonical run evidence.

    File-wide fallbacks are deliberate only for provenance-only assets.  Core
    scientific tables receive field-level rules before their fallback so that
    keys, outcomes, confounders and train-dynamic features stay distinguishable.
    """

    rules: list[UsageRule] = []

    def add(
        rule_id: str,
        file_pattern: str,
        field_pattern: str,
        role: str,
        status: str,
        rationale: str,
    ) -> None:
        rules.append(
            UsageRule(
                rule_id=rule_id,
                file_pattern=file_pattern,
                field_pattern=field_pattern,
                usage_role=role,
                usage_status=status,
                rationale=rationale,
            )
        )

    epoch_file = r"02_logs/epoch_training_metrics\.csv"
    add(
        "epoch-identity",
        epoch_file,
        r"epoch",
        "PAIRING_KEY",
        "AUDITED",
        "Within-run epoch identity for the complete 240x200 grid.",
    )
    add(
        "epoch-resource-time",
        epoch_file,
        r"time",
        "CONFOUND_CONTROL",
        "ANALYZED",
        "Segment-relative epoch time; resume resets must be controlled.",
    )
    add(
        "epoch-top5-constant",
        epoch_file,
        r"metrics/accuracy_top5",
        "CONSTANT_PARAMETER",
        "DOCUMENTED_CONSTANT",
        "All 48,000 saved Top5 values are exactly 1.0.",
    )
    add(
        "epoch-scientific-telemetry",
        epoch_file,
        r"(?:train/loss|val/loss|metrics/accuracy_top1|lr/pg[0-7])",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Full saved train/validation/LR telemetry, including constants.",
    )

    selection_file = r"01_manifests/selection_manifest\.csv"
    add(
        "selection-pairing",
        selection_file,
        r"(?:run_slot|triad_id|condition_id|arm|training_seed|selection_seed|sample_id)",
        "PAIRING_KEY",
        "AUDITED",
        "Frozen run, triad, seed and sample identity.",
    )
    add(
        "selection-features",
        selection_file,
        r"(?:rank|y_true|oof_fold|dynamic_bucket|mean_p_defect|correct_rate|std_p_defect|replay_role|source_method)",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Treatment/control composition and pre-training dynamic features.",
    )

    run_manifest = r"01_manifests/run_manifest\.csv"
    add(
        "run-manifest-sample-key",
        run_manifest,
        r"sample_id",
        "PAIRING_KEY",
        "AUDITED",
        "Links frozen selection to actual training exposure.",
    )
    add(
        "run-manifest-exposure",
        run_manifest,
        r"(?:y_true|role|exposure_index)",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Actual class, replay role and exposure count evidence.",
    )

    prediction_file = r"04_predictions/(?:val_cal|val_op)_predictions\.csv"
    add(
        "prediction-sample-key",
        prediction_file,
        r"sample_id",
        "PAIRING_KEY",
        "AUDITED",
        "Frozen evaluation sample identity.",
    )
    add(
        "prediction-outcomes",
        prediction_file,
        r"(?:y_true|score|score_raw)",
        "OUTCOME_METRIC",
        "ANALYZED",
        "Saved label plus calibrated and raw model outputs.",
    )
    add(
        "threshold-frontier",
        r"05_metrics/threshold_sweep\.csv",
        r".*",
        "OUTCOME_METRIC",
        "RECOMPUTED",
        "Tie-safe full threshold frontier recomputed from saved predictions.",
    )
    add(
        "operational-metrics",
        r"05_metrics/operational_metrics\.json",
        r".*",
        "OUTCOME_METRIC",
        "RECOMPUTED",
        "Saved operational metrics are independently recomputed and audited.",
    )
    add(
        "platt-calibration",
        r"05_metrics/platt_calibration\.json",
        r".*",
        "RECOMPUTED_FIELD",
        "RECOMPUTED",
        "Calibration parameters and diagnostics are verified from val_cal.",
    )

    add(
        "manifest-summary",
        r"01_manifests/manifest_summary\.json",
        r".*",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Base/replay/epoch sample counts define treatment dose and exposure.",
    )
    manifest_file = (
        r"01_manifests/(?:normal_train_manifest|normal_val_model_manifest|"
        r"train_manifest|val_model_manifest|frozen_inputs/[^/]+)\.csv"
    )
    add(
        "sample-manifest-identity",
        manifest_file,
        r"(?:sample_version|split|sample_seed|sample_order|train_primary_class|class_sample_order|Filename|canonical_image_relpath)",
        "PAIRING_KEY",
        "AUDITED",
        "Sample/split/image identity used for joins and leakage checks.",
    )
    add(
        "sample-manifest-lineage",
        manifest_file,
        r"(?:source_csv_path|source_csv_row_number|source_csv_line_number|source_image_path)",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Original CSV/image lineage; not a predictive feature.",
    )
    add(
        "sample-manifest-composition",
        manifest_file,
        r"(?:target_labels|target_label_count|normal_definition|sample_primary_class|eval_primary_class|WaterLevel|VA|RB|OB|PF|DE|FS|IS|RO|IN|AF|BE|FO|GR|PH|PB|OS|OP|OK|ND|Defect)",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Class, source and defect-code composition for subgroup mechanisms.",
    )
    add(
        "input-checksums",
        r"00_identity/input_checksums\.csv",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Frozen input path, row count and checksum evidence.",
    )

    run_identity = r"00_identity/run_identity\.json"
    add(
        "run-identity-pairing",
        run_identity,
        r"(?:run_slot|run_row\.(?:run_slot|triad_id|condition_id|condition_slot|phase|arm|method|budget|guard_ratio|training_seed|selection_seed|discovery_or_confirmation))",
        "PAIRING_KEY",
        "AUDITED",
        "Canonical matrix and triad identity.",
    )
    add(
        "run-identity-confounds",
        run_identity,
        r"(?:machine_id|input_snapshot_id|resume_count|resume_mode|resume_segments\[\].*|last_epoch)",
        "CONFOUND_CONTROL",
        "ANALYZED",
        "Machine, snapshot and resume state used in sensitivity analysis.",
    )
    add(
        "run-identity-lineage",
        run_identity,
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Release, contract, hash and attempt provenance.",
    )
    add(
        "environment-confounds",
        r"00_identity/environment_(?:controller|training)\.json",
        r".*",
        "CONFOUND_CONTROL",
        "ANALYZED",
        "Runtime software/hardware environment and process context.",
    )
    add(
        "resolved-training-config",
        r"02_logs/(?:args\.yaml|resolved_training_args\.json)",
        r".*",
        "CONFOUND_CONTROL",
        "ANALYZED",
        "Configured optimizer, augmentation and trainer arguments; constants are proven from all runs.",
    )
    add(
        "effective-training-audit",
        r"02_logs/training_execution_audit\.json",
        r".*",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Effective batch, steps, resume segments and finite-loss audit.",
    )
    add(
        "effective-optimizer-log",
        r"02_logs/train_\{TIMESTAMP\}_\{TOKEN\}\.log(?:\.result\.json)?",
        r".*",
        "SCIENTIFIC_FEATURE",
        "ANALYZED",
        "Primary evidence for the actually resolved optimizer and training execution.",
    )
    add(
        "gpu-resource-telemetry",
        r"02_logs/gpu_usage_\{TIMESTAMP\}_\{TOKEN\}\.csv",
        r".*",
        "CONFOUND_CONTROL",
        "ANALYZED",
        "GPU utilization, memory, temperature and power confound diagnostics.",
    )
    add(
        "segment-execution-lineage",
        r"02_logs/segment_\{TIMESTAMP\}_\{TOKEN\}_\{SPLIT\}\.log(?:\.result\.json)?",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Isolated prediction segment execution and return-code evidence.",
    )
    add(
        "checkpoint-probe-log",
        r"02_logs/checkpoint_probe_\{CHECKPOINT\}_\{TOKEN\}\.log(?:\.result\.json)?",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Checkpoint load/probe execution evidence.",
    )
    add(
        "checkpoint-payload",
        r"03_checkpoints/(?:best|last)\.pt",
        r".*",
        "SCIENTIFIC_FEATURE",
        "AUDITED",
        "Binary checkpoint existence/SHA is audited here; derived layerwise drift is a separate stage.",
    )

    add(
        "artifact-manifest",
        r"07_validation/artifact_manifest\.csv",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Permanent artifact size/checksum inventory.",
    )
    add(
        "validation-standard",
        r"07_validation/(?:checkpoint_preflight|preflight_report|postflight_report|storage_preflight)\.json",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Formal input, storage, checkpoint and postflight validation evidence.",
    )
    add(
        "checkpoint-probe-validation",
        r"07_validation/checkpoint_probe_\{CHECKPOINT\}_\{TOKEN\}\.json",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Structured checkpoint compatibility and checksum evidence.",
    )
    add(
        "manual-recovery-log",
        r"02_logs/checkpoint_probe_manual_recovery_\{TIMESTAMP\}(?:\.\{STREAM\})?\.log",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Manual recovery evidence is audited but never used as a scientific feature.",
    )
    add(
        "manual-recovery-validation",
        r"07_validation/(?:checkpoint_probe_manual_recovery_\{TIMESTAMP\}|checkpoint_probe_manual_sac_repair|manual_recovery_audit_\{TIMESTAMP\}|marker_FAILED_TRAIN_before_manual_recovery_\{TIMESTAMP\}|status_before_manual_recovery_\{TIMESTAMP\}|training_execution_audit\.pre_repair_\{TIMESTAMP\}|training_execution_audit_repair_\{TIMESTAMP\})\.json",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Manual recovery and pre/post-repair provenance only.",
    )
    add(
        "orphan-temp",
        r"02_logs/\.train_\{TIMESTAMP\}_\{TOKEN\}\.log\.result\.json\.\{TOKEN\}\.tmp",
        r".*",
        "LINEAGE_VALIDATION",
        "EXCLUDED_WITH_REASON",
        "Unregistered orphan temporary file; documented and prohibited from scientific analysis.",
    )
    add(
        "status-authority",
        r"08_status/(?:status\.json|VALIDATED)",
        r".*",
        "LINEAGE_VALIDATION",
        "AUDITED",
        "Canonical VALIDATED state and compatibility marker.",
    )
    return rules
