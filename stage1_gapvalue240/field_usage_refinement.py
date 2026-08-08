"""Value-profile and role refinement for the canonical 240-run field ledger.

The module is read-only with respect to canonical run evidence.  It profiles
JSON, YAML and CSV values without placing raw paths, commands, or secrets in
the derived tables, then refines the broad first-pass field roles using both
file semantics and proved constant/variable behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .comprehensive_audit import ALLOWED_USAGE_ROLES, ALLOWED_USAGE_STATUSES


class FieldUsageRefinementError(RuntimeError):
    """Raised when a value profile or refined role cannot be proved complete."""


_STRUCTURED_KINDS = frozenset({"CSV", "JSON", "YAML"})
_SYNTHETIC_KINDS = frozenset({"MISSING_FROM_ATTEMPT", "NOT_COLLECTED"})
_SENSITIVE_FIELD = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth|credential)", re.I
)
_PATH_FIELD = re.compile(
    r"(?:^|[._/])(?:path|cwd|command|log|checkpoint|model|data|project|save_dir|filename|source)(?:$|[._/\[])" ,
    re.I,
)
_DRIVE_OR_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|[\\/].*[\\/])")
_DIGEST_FIELD = re.compile(r"(?:sha256|checksum|digest|hash)", re.I)


@dataclass
class _ValueAccumulator:
    max_exact_unique: int
    example_limit: int
    values_observed: int = 0
    non_null_count: int = 0
    null_count: int = 0
    numeric_count: int = 0
    non_numeric_count: int = 0
    numeric_min: float | None = None
    numeric_max: float | None = None
    first_non_null_key: str | None = None
    non_constant_proved: bool = False
    unique_keys: set[str] = field(default_factory=set)
    unique_exact: bool = True
    examples: list[str] = field(default_factory=list)
    files_with_field: int = 0
    runs_with_field: set[str] = field(default_factory=set)
    digest_parts: list[str] = field(default_factory=list)
    field_present: bool = False

    def add_file_contribution(
        self,
        contribution: "_ValueAccumulator",
        *,
        run_slot: str,
        file_identity: str,
    ) -> None:
        self.values_observed += contribution.values_observed
        self.non_null_count += contribution.non_null_count
        self.null_count += contribution.null_count
        self.numeric_count += contribution.numeric_count
        self.non_numeric_count += contribution.non_numeric_count
        if contribution.numeric_min is not None:
            self.numeric_min = (
                contribution.numeric_min
                if self.numeric_min is None
                else min(self.numeric_min, contribution.numeric_min)
            )
            self.numeric_max = (
                contribution.numeric_max
                if self.numeric_max is None
                else max(self.numeric_max, contribution.numeric_max)
            )
        if contribution.first_non_null_key is not None:
            if self.first_non_null_key is None:
                self.first_non_null_key = contribution.first_non_null_key
            elif self.first_non_null_key != contribution.first_non_null_key:
                self.non_constant_proved = True
        self.non_constant_proved = (
            self.non_constant_proved or contribution.non_constant_proved
        )
        for key in contribution.unique_keys:
            if len(self.unique_keys) <= self.max_exact_unique:
                self.unique_keys.add(key)
        if len(self.unique_keys) > self.max_exact_unique:
            self.unique_exact = False
        if not contribution.unique_exact:
            self.unique_exact = False
        for example in contribution.examples:
            if example not in self.examples and len(self.examples) < self.example_limit:
                self.examples.append(example)
        self.files_with_field += 1
        self.runs_with_field.add(run_slot)
        file_digest = contribution.value_digest()
        self.digest_parts.append(
            hashlib.sha256(
                f"{run_slot}\0{file_identity}\0{file_digest}".encode("utf-8")
            ).hexdigest().upper()
        )

    def add_value(self, value: Any, *, field_path: str, digest: hashlib._Hash) -> None:
        self.values_observed += 1
        key = _stable_value_key(value)
        digest.update(key.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        if key == "<NULL>":
            self.null_count += 1
            return
        self.non_null_count += 1
        if self.first_non_null_key is None:
            self.first_non_null_key = key
        elif self.first_non_null_key != key:
            self.non_constant_proved = True
        if len(self.unique_keys) <= self.max_exact_unique:
            self.unique_keys.add(key)
        if len(self.unique_keys) > self.max_exact_unique:
            self.unique_exact = False
        numeric = _as_finite_float(value)
        if numeric is None:
            self.non_numeric_count += 1
        else:
            self.numeric_count += 1
            self.numeric_min = numeric if self.numeric_min is None else min(self.numeric_min, numeric)
            self.numeric_max = numeric if self.numeric_max is None else max(self.numeric_max, numeric)
        example = _redact_example(value, field_path)
        if example not in self.examples and len(self.examples) < self.example_limit:
            self.examples.append(example)

    def add_csv_series(
        self,
        values: pd.Series,
        *,
        field_path: str,
        digest: hashlib._Hash,
    ) -> None:
        """Add one CSV chunk with vectorized counts and bounded Python work."""

        self.values_observed += len(values)
        text = values.astype(str)
        null_mask = text.eq("")
        null_count = int(null_mask.sum())
        self.null_count += null_count
        non_null = text.loc[~null_mask]
        self.non_null_count += len(non_null)

        digest.update(len(text).to_bytes(8, "little", signed=False))
        if len(text):
            hashes = pd.util.hash_pandas_object(
                text, index=False, categorize=True
            ).to_numpy(dtype="uint64", copy=False)
            digest.update(hashes.tobytes())
        if non_null.empty:
            return

        if self.unique_exact:
            distinct = pd.unique(non_null)
            limited = distinct[: self.max_exact_unique + 1]
            keys = ["str:" + str(value) for value in limited]
            if self.first_non_null_key is None:
                self.first_non_null_key = keys[0]
            if len(keys) > 1 or any(key != self.first_non_null_key for key in keys):
                self.non_constant_proved = True
            for key in keys:
                if len(self.unique_keys) <= self.max_exact_unique:
                    self.unique_keys.add(key)
            if len(distinct) > self.max_exact_unique or len(self.unique_keys) > self.max_exact_unique:
                self.unique_exact = False

            if len(self.examples) < self.example_limit:
                for value in limited:
                    example = _redact_example(value, field_path)
                    if example not in self.examples:
                        self.examples.append(example)
                    if len(self.examples) >= self.example_limit:
                        break

        numeric = pd.to_numeric(non_null, errors="coerce")
        finite_mask = numeric.notna() & numeric.map(math.isfinite)
        finite = numeric.loc[finite_mask]
        self.numeric_count += len(finite)
        self.non_numeric_count += len(non_null) - len(finite)
        if len(finite):
            chunk_min = float(finite.min())
            chunk_max = float(finite.max())
            self.numeric_min = chunk_min if self.numeric_min is None else min(self.numeric_min, chunk_min)
            self.numeric_max = chunk_max if self.numeric_max is None else max(self.numeric_max, chunk_max)

    def value_digest(self) -> str:
        if self.digest_parts:
            payload = "\0".join(sorted(self.digest_parts)).encode("ascii")
            return hashlib.sha256(payload).hexdigest().upper()
        return getattr(self, "_local_digest", hashlib.sha256(b"").hexdigest().upper())


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _stable_value_key(value: Any) -> str:
    if _is_null(value):
        return "<NULL>"
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if math.isnan(value):
            return "<NULL>"
        return f"float:{value:.17g}"
    if isinstance(value, (dict, list)):
        return "json:" + json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "str:" + str(value)


def _as_finite_float(value: Any) -> float | None:
    if _is_null(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:12].upper()


def _redact_example(value: Any, field_path: str) -> str:
    if _is_null(value):
        return "<NULL>"
    text = str(value)
    if _SENSITIVE_FIELD.search(field_path):
        return f"<SECRET:{_short_hash(text)}>"
    if _DIGEST_FIELD.search(field_path) and len(text) >= 16:
        return f"<DIGEST:{text[:12]}>"
    if _PATH_FIELD.search(field_path) or _DRIVE_OR_PATH.search(text):
        return f"<PATH:{_short_hash(text)}>"
    if len(text) > 120:
        return f"<LONG_VALUE:{len(text)}:{_short_hash(text)}>"
    return text


def _flatten_values(value: Any, prefix: str = "") -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}

    def add(path: str, item: Any) -> None:
        fields.setdefault(path, []).append(item)

    if isinstance(value, dict):
        if not value and prefix:
            add(prefix + "{}", "<EMPTY_DICT>")
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            nested = _flatten_values(item, child)
            for path, values in nested.items():
                fields.setdefault(path, []).extend(values)
        return fields
    if isinstance(value, list):
        list_prefix = prefix + "[]"
        if not value:
            add(list_prefix, "<EMPTY_LIST>")
            return fields
        for item in value:
            if isinstance(item, (dict, list)):
                nested = _flatten_values(item, list_prefix)
                for path, values in nested.items():
                    fields.setdefault(path, []).extend(values)
            else:
                add(list_prefix, item)
        return fields
    add(prefix or "<ROOT_SCALAR>", value)
    return fields


def _new_accumulator(max_exact_unique: int, example_limit: int) -> _ValueAccumulator:
    return _ValueAccumulator(max_exact_unique=max_exact_unique, example_limit=example_limit)


def _profile_structured_file(
    path: Path,
    *,
    source_kind: str,
    expected_fields: set[str],
    csv_chunksize: int,
    max_exact_unique: int,
    example_limit: int,
) -> dict[str, _ValueAccumulator]:
    profiles = {
        name: _new_accumulator(max_exact_unique, example_limit)
        for name in expected_fields
    }
    digests = {name: hashlib.sha256() for name in expected_fields}
    if source_kind == "CSV":
        for accumulator in profiles.values():
            accumulator.field_present = True
        try:
            chunks = pd.read_csv(
                path,
                usecols=sorted(expected_fields),
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                chunksize=csv_chunksize,
            )
            for chunk in chunks:
                for name in expected_fields:
                    accumulator = profiles[name]
                    digest = digests[name]
                    accumulator.add_csv_series(
                        chunk[name], field_path=name, digest=digest
                    )
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            raise FieldUsageRefinementError(f"Cannot profile structured CSV: {path.name}: {exc}") from exc
    else:
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text) if source_kind == "JSON" else yaml.safe_load(text)
        except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise FieldUsageRefinementError(f"Cannot profile structured {source_kind}: {path.name}: {exc}") from exc
        flattened = _flatten_values(value)
        for name in expected_fields:
            if name not in flattened:
                continue
            profiles[name].field_present = True
            for item in flattened[name]:
                profiles[name].add_value(item, field_path=name, digest=digests[name])
    for name, accumulator in profiles.items():
        accumulator._local_digest = digests[name].hexdigest().upper()  # type: ignore[attr-defined]
    return profiles


def _safe_source_path(row: Mapping[str, Any]) -> Path:
    root = Path(str(row["attempt_dir"])).resolve()
    path = (root / str(row["relative_path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FieldUsageRefinementError(
            f"Source path escapes canonical attempt: {row['run_slot']}::{row['relative_path']}"
        ) from exc
    if not path.is_file():
        raise FieldUsageRefinementError(
            f"Missing canonical source file: {row['run_slot']}::{row['relative_path']}"
        )
    return path


def build_field_value_profiles(
    source_file_ledger: pd.DataFrame,
    usage_ledger: pd.DataFrame,
    *,
    csv_chunksize: int = 50_000,
    max_exact_unique: int = 10_000,
    example_limit: int = 3,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Profile every ledger field without emitting raw sensitive values."""

    source_required = {"run_slot", "attempt_dir", "relative_path", "normalized_path", "source_kind"}
    usage_required = {
        "normalized_path", "field_path", "source_kind", "files_observed", "runs_observed"
    }
    missing_source = source_required - set(source_file_ledger.columns)
    missing_usage = usage_required - set(usage_ledger.columns)
    if missing_source or missing_usage:
        raise FieldUsageRefinementError(
            f"Ledger columns missing: source={sorted(missing_source)}, usage={sorted(missing_usage)}"
        )
    if max_workers < 1 or csv_chunksize < 1 or max_exact_unique < 1 or example_limit < 1:
        raise FieldUsageRefinementError("Profiling limits and worker count must be positive")
    if usage_ledger.duplicated(["normalized_path", "field_path"]).any():
        raise FieldUsageRefinementError("Usage ledger field identities must be unique")
    source = source_file_ledger.copy()
    if "canonical_attempt" in source and not source["canonical_attempt"].astype(bool).all():
        raise FieldUsageRefinementError("Source ledger contains non-canonical attempts")
    if source.duplicated(["run_slot", "relative_path"]).any():
        raise FieldUsageRefinementError("Source ledger contains duplicate canonical files")

    field_map = {
        str(path): set(group["field_path"].astype(str))
        for path, group in usage_ledger.groupby("normalized_path", sort=True)
    }
    structured = source.loc[source["source_kind"].astype(str).isin(_STRUCTURED_KINDS)].copy()
    jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    cache_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in structured.to_dict(orient="records"):
        normalized = str(row["normalized_path"])
        if normalized not in field_map:
            raise FieldUsageRefinementError(f"Structured source has no usage-ledger fields: {normalized}")
        digest = str(row.get("artifact_manifest_sha256", "") or "").upper()
        cache_token = digest if re.fullmatch(r"[0-9A-F]{64}", digest) else f"{row['run_slot']}::{row['relative_path']}"
        key = (normalized, str(row["source_kind"]), cache_token)
        cache_groups.setdefault(key, []).append(row)
    for rows in cache_groups.values():
        jobs.append((rows[0], rows))

    def inspect(job: tuple[dict[str, Any], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, _ValueAccumulator]]:
        representative, members = job
        path = _safe_source_path(representative)
        normalized = str(representative["normalized_path"])
        profiles = _profile_structured_file(
            path,
            source_kind=str(representative["source_kind"]),
            expected_fields=field_map[normalized],
            csv_chunksize=csv_chunksize,
            max_exact_unique=max_exact_unique,
            example_limit=example_limit,
        )
        return members, profiles

    if max_workers == 1:
        inspected = [inspect(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            inspected = list(executor.map(inspect, jobs))

    aggregate = {
        (str(row.normalized_path), str(row.field_path)): _new_accumulator(max_exact_unique, example_limit)
        for row in usage_ledger.itertuples(index=False)
    }
    for members, profiles in inspected:
        for member in members:
            normalized = str(member["normalized_path"])
            for field_path, contribution in profiles.items():
                if not contribution.field_present:
                    continue
                aggregate[(normalized, field_path)].add_file_contribution(
                    contribution,
                    run_slot=str(member["run_slot"]),
                    file_identity=str(member["relative_path"]),
                )

    records: list[dict[str, Any]] = []
    unprofiled: list[str] = []
    for row in usage_ledger.sort_values(["normalized_path", "field_path"], kind="stable").to_dict("records"):
        key = (str(row["normalized_path"]), str(row["field_path"]))
        accumulator = aggregate[key]
        kind = str(row["source_kind"])
        if kind in _SYNTHETIC_KINDS:
            scan_status = "DOCUMENTED_ABSENT"
        elif kind in _STRUCTURED_KINDS:
            scan_status = "PROFILED"
            expected_files = int(row["files_observed"])
            if accumulator.files_with_field != expected_files:
                unprofiled.append(
                    f"{key[0]}::{key[1]} expected_files={expected_files} profiled={accumulator.files_with_field}"
                )
        else:
            scan_status = "NOT_STRUCTURED"
        if accumulator.values_observed == 0:
            constant_state = "NO_VALUES"
        elif accumulator.non_null_count == 0:
            constant_state = "ALL_NULL"
        elif accumulator.non_constant_proved or accumulator.null_count > 0:
            constant_state = "NON_CONSTANT"
        else:
            constant_state = "CONSTANT"
        exact_unique = bool(accumulator.unique_exact)
        unique_count = len(accumulator.unique_keys) if exact_unique else pd.NA
        lower_bound = len(accumulator.unique_keys)
        records.append(
            {
                "normalized_path": key[0],
                "field_path": key[1],
                "source_kind": kind,
                "scan_status": scan_status,
                "files_profiled": accumulator.files_with_field,
                "runs_profiled": len(accumulator.runs_with_field),
                "values_observed": accumulator.values_observed,
                "non_null_count": accumulator.non_null_count,
                "null_count": accumulator.null_count,
                "unique_count": unique_count,
                "unique_count_lower_bound": lower_bound,
                "unique_count_exact": exact_unique,
                "constant_state": constant_state,
                "constant_proved": constant_state in {"CONSTANT", "ALL_NULL"},
                "numeric_count": accumulator.numeric_count,
                "non_numeric_count": accumulator.non_numeric_count,
                "numeric_min": accumulator.numeric_min,
                "numeric_max": accumulator.numeric_max,
                "redacted_examples_json": json.dumps(accumulator.examples, ensure_ascii=False),
                "value_digest_sha256": accumulator.value_digest(),
                "sensitive_values_redacted": True,
            }
        )
    if unprofiled:
        preview = "; ".join(unprofiled[:8])
        raise FieldUsageRefinementError(
            f"{len(unprofiled)} structured fields were not profiled exactly: {preview}"
        )
    return pd.DataFrame(records)


def _constant(profile: Mapping[str, Any]) -> bool:
    return str(profile.get("constant_state", "")) in {"CONSTANT", "ALL_NULL"}


def _derived_consumer(normalized_path: str) -> str:
    if normalized_path.startswith("03_checkpoints/"):
        return "checkpoint_drift"
    if normalized_path == "04_predictions/val_cal_predictions.csv":
        return "val_cal"
    if normalized_path == "05_metrics/platt_calibration.json":
        return "val_cal"
    if normalized_path.startswith("04_predictions/") or normalized_path.startswith("05_metrics/"):
        return "raw_frontier"
    if normalized_path == "01_manifests/selection_manifest.csv":
        return "selection_mechanisms"
    if "gpu_usage_" in normalized_path:
        return "resource_reliability"
    if normalized_path in {
        "01_manifests/run_manifest.csv",
        "01_manifests/manifest_summary.json",
        "02_logs/args.yaml",
        "02_logs/resolved_training_args.json",
        "02_logs/epoch_training_metrics.csv",
        "02_logs/training_execution_audit.json",
    } or "training_execution_audit" in normalized_path:
        return "training_telemetry"
    return ""


def _path_or_hash_field(field_path: str) -> bool:
    lower = field_path.lower()
    return bool(
        _PATH_FIELD.search(field_path)
        or _DIGEST_FIELD.search(field_path)
        or any(token in lower for token in ("release_ref", "release_commit", "contract_id", "snapshot_id"))
    )


def _refined_role(
    normalized_path: str,
    field_path: str,
    source_kind: str,
    original_role: str,
    original_status: str,
    profile: Mapping[str, Any],
    verified: set[str],
) -> tuple[str, str, str, str]:
    consumer = _derived_consumer(normalized_path)
    verified_consumer = consumer in verified
    lower_field = field_path.lower()
    basename = lower_field.rsplit(".", 1)[-1].replace("[]", "")

    if source_kind == "NOT_COLLECTED" or normalized_path.startswith("<NOT_COLLECTED>"):
        return "NOT_COLLECTED_FIELD", "NOT_TESTABLE", "documented-not-collected", consumer
    if source_kind == "MISSING_FROM_ATTEMPT" or normalized_path.startswith("<MISSING_FROM_ATTEMPT_SELECTION>"):
        return "MISSING_FIELD", "DOCUMENTED_MISSING", "documented-missing", consumer
    if normalized_path == "00_identity/environment_controller.json":
        if field_path == "pid":
            return "DESCRIPTIVE_FIELD", "DESCRIPTIVE_ONLY", "controller-process-id", consumer
        if _constant(profile):
            return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "controller-constant-environment", consumer
        return "CONFOUND_CONTROL", "ANALYZED", "controller-varying-environment", consumer
    if normalized_path == "00_identity/environment_training.json":
        if _constant(profile):
            return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "training-environment-constant", consumer
        return "CONFOUND_CONTROL", "ANALYZED", "training-environment-varying", consumer
    if normalized_path in {"02_logs/args.yaml", "02_logs/resolved_training_args.json"}:
        if basename == "seed":
            return "PAIRING_KEY", "AUDITED", "resolved-training-seed", consumer
        if _path_or_hash_field(field_path):
            return "LINEAGE_VALIDATION", "AUDITED", "resolved-training-path-or-hash", consumer
        if "resume" in lower_field:
            return "CONFOUND_CONTROL", "ANALYZED", "resolved-resume-control", consumer
        if _constant(profile):
            return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "proved-training-constant", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "varying-training-parameter", consumer
    if normalized_path == "02_logs/epoch_training_metrics.csv":
        if field_path == "epoch":
            return "PAIRING_KEY", "AUDITED", "epoch-key", consumer
        if field_path == "time":
            return "CONFOUND_CONTROL", "ANALYZED", "segment-relative-time", consumer
        if field_path == "metrics/accuracy_top5" and _constant(profile):
            return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "proved-top5-constant", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "epoch-training-telemetry", consumer
    if normalized_path == "01_manifests/selection_manifest.csv":
        if field_path in {"run_slot", "triad_id", "condition_id", "arm", "training_seed", "selection_seed", "sample_id"}:
            return "PAIRING_KEY", "AUDITED", "selection-pairing-key", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "selection-composition-feature", consumer
    if normalized_path == "01_manifests/run_manifest.csv":
        if field_path == "sample_id":
            return "PAIRING_KEY", "AUDITED", "training-sample-key", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "training-exposure-feature", consumer
    if normalized_path.startswith("04_predictions/"):
        if field_path == "sample_id":
            return "PAIRING_KEY", "AUDITED", "prediction-sample-key", consumer
        return "OUTCOME_METRIC", "ANALYZED", "prediction-outcome", consumer
    if normalized_path == "05_metrics/platt_calibration.json":
        return "RECOMPUTED_FIELD", "RECOMPUTED", "recomputed-calibration", consumer
    if normalized_path.startswith("05_metrics/"):
        return "OUTCOME_METRIC", "RECOMPUTED", "recomputed-operational-outcome", consumer
    if normalized_path == "07_validation/postflight_report.json":
        if field_path.startswith(("recomputed.", "training_audit.")):
            return "RECOMPUTED_FIELD", "RECOMPUTED", "postflight-recomputation", consumer
        if field_path == "expected.seed":
            return "PAIRING_KEY", "AUDITED", "postflight-expected-seed", consumer
        if field_path.startswith("expected."):
            if _path_or_hash_field(field_path):
                return "LINEAGE_VALIDATION", "AUDITED", "postflight-expected-lineage", consumer
            if _constant(profile):
                return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "postflight-expected-constant", consumer
            return "SCIENTIFIC_FEATURE", "ANALYZED", "postflight-expected-parameter", consumer
        return "LINEAGE_VALIDATION", "AUDITED", "postflight-validation-evidence", consumer
    if normalized_path == "00_identity/run_identity.json" or normalized_path == "07_validation/preflight_report.json":
        if any(
            token in lower_field
            for token in (
                "run_slot",
                "triad_id",
                "condition_id",
                "condition_slot",
                "training_seed",
                "selection_seed",
                ".arm",
            )
        ):
            return "PAIRING_KEY", "AUDITED", "run-pairing-identity", consumer
        if any(token in lower_field for token in (".budget", ".guard_ratio", ".method", ".phase", "discovery_or_confirmation")):
            return "SCIENTIFIC_FEATURE", "ANALYZED", "run-scientific-config", consumer
        if _path_or_hash_field(field_path):
            return "LINEAGE_VALIDATION", "AUDITED", "run-lineage", consumer
        if any(
            token in lower_field
            for token in (
                "resume",
                "machine_id",
                "created_at",
                "started_at",
                "ended_at",
                "duration",
                "last_epoch",
                "pid",
            )
        ):
            return "CONFOUND_CONTROL", "ANALYZED", "run-execution-confound", consumer
        return "LINEAGE_VALIDATION", "AUDITED", "run-validation-evidence", consumer
    if normalized_path == "02_logs/training_execution_audit.json" or "training_execution_audit" in normalized_path:
        if basename == "seed":
            return "PAIRING_KEY", "AUDITED", "training-audit-seed", consumer
        if _path_or_hash_field(field_path) or "repair" in lower_field:
            return "LINEAGE_VALIDATION", "AUDITED", "training-audit-lineage", consumer
        if "resume" in lower_field or any(token in lower_field for token in ("started_at", "ended_at", "error")):
            return "CONFOUND_CONTROL", "ANALYZED", "training-audit-resume-confound", consumer
        if lower_field.startswith("configured_args.") and _constant(profile):
            return "CONSTANT_PARAMETER", "DOCUMENTED_CONSTANT", "training-audit-config-constant", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "training-execution-feature", consumer
    if "gpu_usage_" in normalized_path:
        if field_path == "timestamp_unix":
            return "PAIRING_KEY", "AUDITED", "resource-sample-time-key", consumer
        if field_path == "status":
            return "LINEAGE_VALIDATION", "AUDITED", "resource-sample-status", consumer
        return "CONFOUND_CONTROL", "ANALYZED", "resource-usage-confound", consumer
    if normalized_path.startswith("03_checkpoints/"):
        status = "ANALYZED" if verified_consumer else (
            original_status if original_status in ALLOWED_USAGE_STATUSES else "AUDITED"
        )
        return "SCIENTIFIC_FEATURE", status, "checkpoint-derived-drift", consumer
    if normalized_path.endswith("manifest_summary.json"):
        if _path_or_hash_field(field_path):
            return "LINEAGE_VALIDATION", "AUDITED", "manifest-summary-lineage", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "replay-exposure-summary", consumer
    if normalized_path in {
        "00_identity/input_checksums.csv",
        "07_validation/artifact_manifest.csv",
    }:
        return "LINEAGE_VALIDATION", "AUDITED", "artifact-lineage", consumer
    if "manifest" in normalized_path and source_kind == "CSV":
        if field_path in {
            "sample_id", "canonical_image_relpath", "Filename", "sample_order",
            "class_sample_order", "sample_seed", "sample_version", "split",
        }:
            return "PAIRING_KEY", "AUDITED", "sample-manifest-identity", consumer
        if _path_or_hash_field(field_path) or field_path.startswith("source_"):
            return "LINEAGE_VALIDATION", "AUDITED", "sample-manifest-lineage", consumer
        return "SCIENTIFIC_FEATURE", "ANALYZED", "sample-manifest-composition", consumer
    if normalized_path.startswith("07_validation/"):
        return "LINEAGE_VALIDATION", "AUDITED", "validation-lineage", consumer
    if normalized_path.endswith(".log.result.json"):
        if any(token in lower_field for token in ("duration", "pid", "started_at", "ended_at")):
            return "DESCRIPTIVE_FIELD", "DESCRIPTIVE_ONLY", "process-descriptive", consumer
        return "LINEAGE_VALIDATION", "AUDITED", "process-execution-lineage", consumer

    role = original_role if original_role in ALLOWED_USAGE_ROLES else "DESCRIPTIVE_FIELD"
    status = original_status if original_status in ALLOWED_USAGE_STATUSES else "DESCRIPTIVE_ONLY"
    return role, status, "preserved-reviewed-role", consumer


def refine_usage_ledger(
    usage_ledger: pd.DataFrame,
    value_profiles: pd.DataFrame,
    *,
    verified_derived_outputs: Iterable[str] = (),
) -> pd.DataFrame:
    """Return a one-to-one role-refined ledger with attached profile proof."""

    keys = ["normalized_path", "field_path"]
    if usage_ledger.duplicated(keys).any() or value_profiles.duplicated(keys).any():
        raise FieldUsageRefinementError("Usage ledger and value profiles require unique field identities")
    if set(map(tuple, usage_ledger[keys].astype(str).to_numpy())) != set(
        map(tuple, value_profiles[keys].astype(str).to_numpy())
    ):
        raise FieldUsageRefinementError("Value-profile field identities differ from usage ledger")
    merged = usage_ledger.merge(
        value_profiles,
        on=["normalized_path", "field_path", "source_kind"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_profile"),
    )
    if merged["scan_status"].isna().any():
        raise FieldUsageRefinementError("Some usage-ledger fields lack a value profile")
    verified = {str(item) for item in verified_derived_outputs}
    records: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        original_role = str(row["usage_role"])
        original_status = str(row["usage_status"])
        role, status, rule_id, consumer = _refined_role(
            str(row["normalized_path"]),
            str(row["field_path"]),
            str(row["source_kind"]),
            original_role,
            original_status,
            row,
            verified,
        )
        row["original_usage_role"] = original_role
        row["original_usage_status"] = original_status
        row["usage_role"] = role
        row["usage_status"] = status
        row["refinement_rule_id"] = rule_id
        row["derived_consumer"] = consumer
        row["derived_evidence_verified"] = bool(consumer and consumer in verified)
        row["silently_dropped"] = False
        records.append(row)
    output = pd.DataFrame(records).sort_values(keys, kind="stable").reset_index(drop=True)
    assert_refined_ledger_complete(output)
    return output


def assert_refined_ledger_complete(ledger: pd.DataFrame) -> dict[str, int]:
    required = {"normalized_path", "field_path", "usage_role", "usage_status", "silently_dropped"}
    missing = required - set(ledger.columns)
    if missing:
        raise FieldUsageRefinementError(f"Refined ledger missing columns: {sorted(missing)}")
    if ledger.duplicated(["normalized_path", "field_path"]).any():
        raise FieldUsageRefinementError("Refined ledger contains duplicate field identities")
    roles = ledger["usage_role"].fillna("").astype(str).str.strip().str.upper()
    statuses = ledger["usage_status"].fillna("").astype(str).str.strip().str.upper()
    silent_flag = ledger["silently_dropped"].fillna(True).astype(bool)
    gates = {
        "UNREVIEWED": int(roles.isin({"", "UNREVIEWED"}).sum()),
        "UNCLASSIFIED": int((~roles.isin(ALLOWED_USAGE_ROLES)).sum()),
        "SILENTLY_DROPPED": int(
            (silent_flag | statuses.isin({"", "SILENTLY_DROPPED"}) | ~statuses.isin(ALLOWED_USAGE_STATUSES)).sum()
        ),
    }
    if any(gates.values()):
        raise FieldUsageRefinementError(
            "Incomplete refined ledger: " + ", ".join(f"{key}={value}" for key, value in gates.items())
        )
    return gates


def publish_refined_field_usage(
    source_file_ledger_path: str | Path,
    usage_ledger_path: str | Path,
    output_dir: str | Path,
    *,
    csv_chunksize: int = 50_000,
    max_exact_unique: int = 10_000,
    example_limit: int = 3,
    max_workers: int = 4,
    verified_derived_outputs: Iterable[str] = (),
) -> dict[str, Any]:
    """Publish only the two derived audit tables into a supplied in-progress root.

    This deliberately does not update ``ANALYSIS_STATE.json`` or any manifest;
    the owning pipeline must review and register the artifacts separately.
    """

    output = Path(output_dir)
    if not any(part.endswith(".inprogress") for part in output.parts):
        raise FieldUsageRefinementError("Refined field usage may only publish under an .inprogress output")
    audit_dir = output / "audit"
    profile_path = audit_dir / "FIELD_VALUE_PROFILES.csv"
    refined_path = audit_dir / "DATA_USAGE_LEDGER_REFINED.csv"
    for path in (profile_path, refined_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite derived field audit: {path.name}")
    source = pd.read_csv(source_file_ledger_path)
    usage = pd.read_csv(usage_ledger_path)
    profiles = build_field_value_profiles(
        source,
        usage,
        csv_chunksize=csv_chunksize,
        max_exact_unique=max_exact_unique,
        example_limit=example_limit,
        max_workers=max_workers,
    )
    refined = refine_usage_ledger(
        usage,
        profiles,
        verified_derived_outputs=verified_derived_outputs,
    )
    gates = assert_refined_ledger_complete(refined)
    audit_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = [
        profile_path.with_name(f".{profile_path.name}.{os.getpid()}.tmp"),
        refined_path.with_name(f".{refined_path.name}.{os.getpid()}.tmp"),
    ]
    try:
        profiles.to_csv(temporary_paths[0], index=False)
        refined.to_csv(temporary_paths[1], index=False)
        temporary_paths[0].replace(profile_path)
        temporary_paths[1].replace(refined_path)
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
    return {
        "status": "PASS",
        "profile_rows": len(profiles),
        "refined_ledger_rows": len(refined),
        "gates": gates,
        "outputs": ["audit/FIELD_VALUE_PROFILES.csv", "audit/DATA_USAGE_LEDGER_REFINED.csv"],
        "state_or_manifest_modified": False,
    }
