from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arm_spec import default_phase1_arms
from .checkpointing import load_checkpoint
from .columnar import PORTABLE_MAGIC, read_columnar, validate_columnar_file
from .epoch_transaction import validate_receipt_chain
from .errors import ErrorCode, SctsrError
from .evaluation import compute_tie_safe_frontier
from .exposure_ledger import validate_exposure_rows
from .logical_artifact_index import LogicalArtifactEntry, LogicalArtifactIndex
from .occurrence_ledger import validate_occurrence_rows
from .prediction_artifact import (
    PredictionRow,
    read_registered_prediction_artifact,
    sample_label_identity_digest,
    validate_prediction_rows,
)
from .recovery import validate_recovery_pointer
from .schedule import schedule_from_dict
from .selection_ledger import validate_selection_rows
from .serialization import load_json, sha256_file, stable_digest
from .step_ledger import validate_step_rows
from .telemetry import TelemetryRow, validate_telemetry_for_closeout


SYNTHETIC_SEMANTIC = "SYNTHETIC_NOT_SCIENTIFIC_RESULT"
_PROHIBITED_SIDE_EFFECTS = (
    "formal_training_started",
    "engineering_gate_generated",
    "assignments_generated",
    "pilot_release_generated",
    "blind_holdout_opened",
    "selector_trained",
    "method_effectiveness_claimed",
)


def _closeout_failure(message: str, *, observed: Any = None, expected: Any = None) -> SctsrError:
    return SctsrError(
        ErrorCode.CLOSEOUT_NOT_VALIDATED,
        message,
        observed=observed,
        expected=expected,
        required_action="Restore the exact required run evidence and regenerate the immutable artifact index.",
    )


def _require(condition: bool, message: str, *, observed: Any = None, expected: Any = None) -> None:
    if not condition:
        raise _closeout_failure(message, observed=observed, expected=expected)


def build_artifact_index(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.name not in {"ARTIFACT_INDEX.json", "FORMAL_COMPLETION_RECEIPT.json"}:
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    files.sort(key=lambda row: row["path"])
    return {
        "schema_version": "stage1.sctsr.artifact_index.v1",
        "files": files,
        "artifact_index_digest": stable_digest(files),
    }


def _validate_exhaustive_index(
    root: Path,
    *,
    allow_synthetic_portable_fallback: bool,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    index_path = root / "ARTIFACT_INDEX.json"
    if not index_path.is_file():
        raise _closeout_failure("Run artifact index is missing")
    index = load_json(index_path)
    listed = index.get("artifacts", index.get("files", []))
    _require(isinstance(listed, list) and listed, "Artifact index is empty")
    indexed_paths = {str(row.get("relative_path") or row.get("path")) for row in listed}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"ARTIFACT_INDEX.json", "FORMAL_COMPLETION_RECEIPT.json"}
    }
    _require(
        indexed_paths == actual_paths,
        "Artifact index is not exhaustive",
        observed={"unindexed": sorted(actual_paths - indexed_paths), "missing": sorted(indexed_paths - actual_paths)},
    )
    checked: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    for row in listed:
        rel = row.get("relative_path") or row.get("path")
        _require(isinstance(rel, str) and rel not in {"", "."}, "Artifact index entry has no path", observed=row)
        rel_path = Path(rel)
        _require(not rel_path.is_absolute() and not rel_path.drive and ".." not in rel_path.parts, "Artifact index path escapes run root", observed=rel)
        path = root / rel_path
        _require(path.is_file(), "Indexed artifact is missing", observed=rel)
        observed_sha = sha256_file(path)
        expected_sha = row.get("sha256")
        _require(observed_sha == expected_sha, "Indexed artifact SHA mismatch", observed=observed_sha, expected=expected_sha)
        _require(path.stat().st_size == int(row.get("bytes", -1)), "Indexed artifact byte count mismatch", observed=path.stat().st_size, expected=row.get("bytes"))
        in_quarantine = "09_quarantine" in rel_path.parts
        if path.suffix == ".parquet" and not in_quarantine:
            portable = path.read_bytes()[: len(PORTABLE_MAGIC)] == PORTABLE_MAGIC
            if portable and not allow_synthetic_portable_fallback:
                raise SctsrError(
                    ErrorCode.SYNTHETIC_RESULT_MISLABELLED,
                    "Synthetic portable columnar is not canonical Parquet",
                    artifact_path=str(path),
                )
            validate_columnar_file(path, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
        checked.append({"relative_path": rel, "sha256": observed_sha, "bytes": path.stat().st_size})
        canonical_rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": observed_sha})
    canonical_rows.sort(key=lambda row: row["path"])
    expected_digest = index.get("artifact_index_digest")
    _require(
        expected_digest == stable_digest(canonical_rows),
        "Artifact index digest mismatch",
        observed=stable_digest(canonical_rows),
        expected=expected_digest,
    )
    return index, checked


def _manifested_partition(
    root: Path,
    relative_path: str,
    manifest: Mapping[str, Any],
    *,
    allow_synthetic_portable_fallback: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    path = root / relative_path
    _require(path.is_file(), "Required semantic partition is missing", observed=relative_path)
    report = validate_columnar_file(
        path,
        expected_rows=int(manifest["row_count"]),
        expected_schema_version=str(manifest["schema_version"]),
        expected_schema_digest=str(manifest["schema_digest"]),
        expected_sha256=str(manifest["sha256"]),
        allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
    )
    rows = read_columnar(path, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
    _require(len(rows) == int(report["row_count"]), "Partition read count differs from its manifest")
    return path, rows


def _prediction_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[PredictionRow, ...]:
    names = {field.name for field in fields(PredictionRow)}
    return tuple(PredictionRow(**{name: row[name] for name in names}) for row in rows)


def _telemetry_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[TelemetryRow, ...]:
    names = {field.name for field in fields(TelemetryRow)}
    return tuple(TelemetryRow(**{name: row[name] for name in names}) for row in rows)


def validate_formal_endpoint_evidence(
    run_root: str | Path,
    *,
    manifest: Mapping[str, Any],
    checkpoint_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Recompute and cross-bind the immutable E200 ``val_op`` endpoint.

    A formal branch is incomplete without this evidence.  The function does
    not trust either the published frontier rows or their summary: it reloads
    the registered prediction artifact, revalidates its checkpoint and split
    identity, recomputes all 96 tie-safe points, and then compares the stored
    artifacts byte-semantically with that recomputation.
    """

    root = Path(run_root).resolve()
    run_id = str(manifest.get("run_id", ""))
    arm_id = str(manifest.get("arm_id", ""))
    if not run_id or not arm_id:
        raise _closeout_failure("Formal endpoint cannot be located without run/arm identity")
    prediction_root = root / "06_predictions" / f"run_id={run_id}" / "epoch=0200"
    evaluation_root = root / "07_evaluation" / f"run_id={run_id}" / "epoch=0200"
    paths = {
        "prediction": prediction_root / "predictions.parquet",
        "prediction_summary": prediction_root / "prediction_summary.json",
        "split_bundle": prediction_root / "split_identity_bundle.json",
        "frontier": evaluation_root / "frontier.parquet",
        "frontier_summary": evaluation_root / "frontier_summary.json",
    }
    receipt_path = root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json"
    missing = sorted(name for name, path in {**paths, "endpoint_receipt": receipt_path}.items() if not path.is_file())
    if missing:
        raise _closeout_failure(
            "Formal branch lacks required E200 val_op endpoint evidence",
            observed=missing,
            expected=sorted(paths),
        )

    receipt = load_json(receipt_path)
    receipt_core = {key: value for key, value in receipt.items() if key != "endpoint_digest"}
    expected_receipt_fields = {
        "schema_version", "status", "run_id", "arm_id", "training_seed", "split_role",
        "checkpoint_epoch", "checkpoint_sha256", "model_variant", "selection_semantic", "files",
        "formal_training_started", "blind_holdout_opened", "test_accessed",
        "method_effectiveness_claimed", "endpoint_digest",
    }
    _require(set(receipt) == expected_receipt_fields, "Formal endpoint receipt schema is not exact")
    _require(
        receipt.get("schema_version") == "stage1.sctsr.formal_endpoint_receipt.v1"
        and receipt.get("status") == "FORMAL_ENDPOINT_COMPLETE_NOT_METHOD_SELECTION"
        and receipt.get("endpoint_digest") == stable_digest(receipt_core)
        and receipt.get("formal_training_started") is True
        and receipt.get("blind_holdout_opened") is False
        and receipt.get("test_accessed") is False
        and receipt.get("method_effectiveness_claimed") is False,
        "Formal endpoint receipt state/digest is invalid",
    )
    expected_file_rows = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths.values()
        ),
        key=lambda row: row["path"],
    )
    _require(receipt.get("files") == expected_file_rows, "Formal endpoint receipt does not bind the exact endpoint bytes")

    predictions, prediction_summary, binding = read_registered_prediction_artifact(
        paths["prediction"],
        summary_path=paths["prediction_summary"],
        checkpoint_path=checkpoint_path,
        evaluation_mode="formal",
        repository_root=repository_root,
    )
    _require(binding.split_role == "val_op" and binding.checkpoint_epoch == 200, "Formal endpoint is not E200 val_op")
    _require(
        Path(binding.split_manifest_path).resolve() == paths["split_bundle"].resolve(),
        "Formal endpoint prediction binding does not use its canonical split bundle",
    )
    _require(
        predictions[0].run_id == run_id
        and predictions[0].arm_id == arm_id
        and predictions[0].training_seed == int(manifest.get("training_seed", -1))
        and predictions[0].source_tree_digest == str(manifest.get("source_tree_digest", ""))
        and prediction_summary.get("asset_registry_digest") == manifest.get("asset_registry_digest"),
        "Formal endpoint identity differs from its run manifest",
    )
    _require(
        receipt.get("run_id") == run_id
        and receipt.get("arm_id") == arm_id
        and int(receipt.get("training_seed", -1)) == binding.training_seed
        and receipt.get("split_role") == binding.split_role
        and int(receipt.get("checkpoint_epoch", -1)) == binding.checkpoint_epoch
        and receipt.get("checkpoint_sha256") == binding.checkpoint_sha256
        and receipt.get("model_variant") == binding.model_variant
        and receipt.get("selection_semantic") == "ENDPOINT_ONLY_NOT_FOR_SELECTION",
        "Formal endpoint receipt identity differs from registered predictions",
    )

    frontier_report = validate_columnar_file(
        paths["frontier"],
        expected_rows=96,
        expected_schema_version="stage1.sctsr.frontier.v1",
        expected_sha256=sha256_file(paths["frontier"]),
    )
    stored_rows = read_columnar(paths["frontier"])
    expected_points, expected_summary = compute_tie_safe_frontier(
        predictions,
        max_fn=95,
        target_tn=68_253,
        checkpoint_sha256=binding.checkpoint_sha256,
        prediction_artifact_sha256=sha256_file(paths["prediction"]),
    )
    expected_rows = [asdict(point) for point in expected_points]
    _require(
        stored_rows == expected_rows,
        "Published formal frontier differs from recomputation over registered predictions",
    )
    stored_summary = load_json(paths["frontier_summary"])
    expected_summary_payload = {
        "schema_version": "stage1.sctsr.frontier_summary.v1",
        **asdict(expected_summary),
        "evaluation_mode": "formal",
        "split_role": "val_op",
        "checkpoint_epoch": 200,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "prediction_artifact_sha256": sha256_file(paths["prediction"]),
        "frontier_artifact_sha256": sha256_file(paths["frontier"]),
        "frontier_row_count": 96,
        "selection_semantic": "ENDPOINT_ONLY_NOT_FOR_SELECTION",
        "two_anchor_thresholds_are_independent": True,
    }
    _require(
        stored_summary == expected_summary_payload,
        "Published formal frontier summary differs from endpoint recomputation",
    )
    return {
        "status": "PASS",
        "split_role": "val_op",
        "checkpoint_epoch": 200,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "prediction_rows": len(predictions),
        "prediction_sha256": sha256_file(paths["prediction"]),
        "frontier_points": int(frontier_report["row_count"]),
        "frontier_sha256": sha256_file(paths["frontier"]),
        "sample_label_identity_digest": sample_label_identity_digest(predictions),
        "selection_semantic": "ENDPOINT_ONLY_NOT_FOR_SELECTION",
        "endpoint_receipt_sha256": sha256_file(receipt_path),
    }


def _validate_synthetic_canary(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    allow_synthetic_portable_fallback: bool,
) -> dict[str, Any]:
    _require(manifest.get("schema_version") == "stage1.sctsr.synthetic_run_manifest.v1", "Synthetic run manifest schema is invalid")
    _require(manifest.get("semantic") == SYNTHETIC_SEMANTIC and manifest.get("scientific_result") is False, "Synthetic run is mislabelled")
    enabled = [field for field in _PROHIBITED_SIDE_EFFECTS if manifest.get(field) is not False]
    _require(not enabled, "Synthetic run does not explicitly prove prohibited side effects absent", observed=enabled)
    expected_arms = sorted(spec.arm_id.value for spec in default_phase1_arms())
    observed_arms = sorted(manifest.get("eight_arms", []))
    _require(observed_arms == expected_arms, "Synthetic canary does not contain the exact eight-arm matrix", observed=observed_arms, expected=expected_arms)

    required_receipts = {
        "parent": root / "02_parent" / "PARENT_RECEIPT.json",
        "canary": root / "08_receipts" / "SYNTHETIC_CANARY_RECEIPT.json",
        "mechanisms": root / "08_receipts" / "SYNTHETIC_MECHANISM_AUDIT.json",
        "failures": root / "08_receipts" / "FAILURE_INJECTION_SUMMARY.json",
        "resume": root / "08_receipts" / "RESUME_IDENTITY.json",
        "logical": root / "ARTIFACT_INDEX_LOGICAL.json",
    }
    missing = sorted(name for name, path in required_receipts.items() if not path.is_file())
    _require(not missing, "Synthetic canary is missing required receipts", observed=missing)

    parent = load_json(required_receipts["parent"])
    _require(parent.get("semantic") == SYNTHETIC_SEMANTIC and parent.get("logical_epoch") == 120, "Synthetic parent receipt is invalid")
    parent_checkpoint = root / str(parent.get("checkpoint_path", ""))
    parent_payload = load_checkpoint(parent_checkpoint, expected_sha256=str(parent.get("checkpoint_sha256")), expected_epoch=120)
    _require(int(parent_payload["training_seed"]) == int(manifest["training_seed"]), "Synthetic parent seed differs from run manifest")
    _require(parent.get("replay_occurrences") == 0, "Synthetic common parent contains replay")

    selection = manifest.get("selection_evidence")
    _require(isinstance(selection, Mapping) and set(selection) == {"T_STRESS", "R1_GLOBAL_RANDOM", "R2_MATCHED_RANDOM"}, "Selection evidence set is incomplete")
    for policy, part_manifest in selection.items():
        relative = f"04_ledgers/selection/run_id=SYNTHETIC_SELECTION/epoch=0000/{policy}.parquet"
        _, rows = _manifested_partition(root, relative, part_manifest, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
        validate_selection_rows(rows, r2=policy == "R2_MATCHED_RANDOM")

    branch_summaries = manifest.get("branch_summaries")
    _require(isinstance(branch_summaries, Mapping) and sorted(branch_summaries) == expected_arms, "Branch summary set is incomplete")
    per_arm: dict[str, Any] = {}
    for arm in expected_arms:
        branch_receipt_path = root / "03_branch" / arm / "BRANCH_RECEIPT.json"
        _require(branch_receipt_path.is_file(), "Synthetic branch receipt is missing", observed=arm)
        receipt = load_json(branch_receipt_path)
        _require(receipt == branch_summaries[arm], "Run manifest branch summary differs from branch receipt", observed=arm)
        _require(receipt.get("semantic") == SYNTHETIC_SEMANTIC and receipt.get("arm_id") == arm, "Synthetic branch identity is invalid", observed=arm)
        run_id = str(receipt["run_id"])
        checkpoint = root / "05_checkpoints" / f"{run_id}_e200.pt"
        checkpoint_payload = load_checkpoint(checkpoint, expected_sha256=str(receipt["branch_checkpoint_sha256"]), expected_epoch=200)
        _require(int(checkpoint_payload["training_seed"]) == int(manifest["training_seed"]), "Branch checkpoint seed mismatch", observed=arm)
        _require(str(checkpoint_payload["source_tree_digest"]) == str(manifest["source_tree_digest"]), "Branch checkpoint source mismatch", observed=arm)

        base = f"04_ledgers"
        occurrence_path, occurrence = _manifested_partition(
            root,
            f"{base}/occurrence/run_id={run_id}/epoch=0121/part-00000.parquet",
            receipt["occurrence_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        step_path, steps = _manifested_partition(
            root,
            f"{base}/optimizer_step/run_id={run_id}/epoch=0121/part-00000.parquet",
            receipt["step_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        telemetry_path, telemetry = _manifested_partition(
            root,
            f"{base}/telemetry/run_id={run_id}/epoch=0121/part-00000.parquet",
            receipt["telemetry_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        _, exposure = _manifested_partition(
            root,
            f"{base}/exposure/run_id={run_id}/epoch=0121/part-00000.parquet",
            receipt["exposure_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        prediction_path, prediction_raw = _manifested_partition(
            root,
            f"06_predictions/run_id={run_id}/epoch=0200/predictions.parquet",
            receipt["prediction_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        _, frontier = _manifested_partition(
            root,
            f"07_evaluation/run_id={run_id}/epoch=0200/frontier.parquet",
            receipt["frontier_partition"],
            allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
        )
        validate_occurrence_rows(occurrence)
        validate_step_rows(steps)
        validate_exposure_rows(exposure)
        validate_telemetry_for_closeout(_telemetry_rows(telemetry))
        predictions = _prediction_rows(prediction_raw)
        validate_prediction_rows(predictions)
        _require(len(frontier) == 96 and [int(row["fn_budget"]) for row in frontier] == list(range(96)), "Synthetic frontier is not the complete FN=0..95 sequence", observed=arm)
        _require(len(steps) == int(receipt["optimizer_steps"]), "Step ledger count differs from branch receipt", observed=arm)
        _require(len(occurrence) == int(receipt["base_occurrences"]) + int(receipt["replay_occurrences"]), "Occurrence ledger count differs from branch receipt", observed=arm)
        exposure_row = exposure[0]
        _require(exposure_row["occurrence_partition_sha256"] == sha256_file(occurrence_path), "Exposure-to-occurrence SHA binding failed", observed=arm)
        _require(exposure_row["step_partition_sha256"] == sha256_file(step_path), "Exposure-to-step SHA binding failed", observed=arm)
        _require(exposure_row["telemetry_partition_sha256"] == sha256_file(telemetry_path), "Exposure-to-telemetry SHA binding failed", observed=arm)
        _require(exposure_row["checkpoint_sha256"] == sha256_file(checkpoint), "Exposure-to-checkpoint SHA binding failed", observed=arm)
        summary_path = root / "07_evaluation" / f"run_id={run_id}" / "epoch=0200" / "unreachable_target_diagnostic.json"
        _require(summary_path.is_file(), "Synthetic endpoint diagnostic is missing", observed=arm)
        diagnostic = load_json(summary_path)
        prediction_summary = diagnostic.get("prediction_summary", {})
        _require(prediction_summary.get("prediction_artifact_sha256") == sha256_file(prediction_path), "Prediction summary SHA binding failed", observed=arm)
        _require(prediction_summary.get("sample_label_identity_digest") == sample_label_identity_digest(predictions), "Prediction sample-label identity digest failed", observed=arm)
        _require(diagnostic.get("unreachable_fixture_reachable") is False, "Unreachable TN diagnostic did not remain unreachable", observed=arm)
        per_arm[arm] = {
            "run_id": run_id,
            "optimizer_steps": len(steps),
            "occurrences": len(occurrence),
            "prediction_rows": len(predictions),
            "frontier_points": len(frontier),
        }

    failure_summary = load_json(required_receipts["failures"])
    expected_faults = {"KILL", "OOM", "DISK_FULL", "CORRUPT_RECEIPT", "HALF_WRITTEN_JSON", "HALF_WRITTEN_PARQUET"}
    faults = failure_summary.get("faults", [])
    _require(failure_summary.get("status") == "PASS" and {row.get("fault") for row in faults} == expected_faults and len(faults) == 6, "Failure-injection matrix is incomplete")
    for row in faults:
        quarantine = root / str(row.get("quarantine_path", ""))
        partial = root / str(row.get("partial_artifact", ""))
        _require(quarantine.is_dir() and partial.is_file(), "Failure injection was not quarantined", observed=row.get("fault"))
        _require(sha256_file(partial) == row.get("partial_artifact_sha256"), "Quarantined artifact SHA mismatch", observed=row.get("fault"))

    logical_raw = load_json(required_receipts["logical"])
    entries = [LogicalArtifactEntry(**row) for row in logical_raw.get("entries", [])]
    logical = LogicalArtifactIndex(entries)
    logical.validate()
    _require(logical.digest == logical_raw.get("digest") and len(entries) == 120 + 80 * 8, "Logical compressed timeline identity is incomplete")
    for entry in entries:
        physical = root / entry.artifact_relative_path
        _require(physical.is_file() and sha256_file(physical) == entry.artifact_sha256, "Logical timeline references invalid physical evidence", observed=entry.artifact_relative_path)

    canary = load_json(required_receipts["canary"])
    mechanisms = load_json(required_receipts["mechanisms"])
    _require(canary.get("status") == "PASS" and canary.get("semantic") == SYNTHETIC_SEMANTIC, "Terminal synthetic receipt is invalid")
    _require(mechanisms.get("status") == "PASS_SYNTHETIC_MECHANISMS_ONLY", "Synthetic mechanism audit is invalid")
    _require(mechanisms.get("checks", {}).get("full_implementation_completion") == "NOT_ASSESSED_SYNTHETIC", "Synthetic audit improperly claims implementation completion")
    return {"synthetic_semantic_status": "PASS", "arms": per_arm, "failure_injection_count": len(faults), "logical_entry_count": len(entries)}


def _validate_formal_tree(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("schema_version") == "stage1.sctsr.formal_run_manifest.v2" and manifest.get("execution_mode") == "formal", "Formal run manifest is missing the current formal execution semantic")
    enabled = [field for field in _PROHIBITED_SIDE_EFFECTS[1:] if manifest.get(field) is not False]
    _require(not enabled, "Formal run contains prohibited orchestration/test/claim side effects", observed=enabled)
    _require(manifest.get("formal_training_started") is True and manifest.get("formal_training_authorized") is True, "Completed formal run does not record its authorized training side effect")
    _require(manifest.get("test_accessed") is False and manifest.get("best_pt_used") is False, "Formal run accessed test or best.pt")
    role = manifest.get("run_role")
    _require(role in {"COMMON_PARENT", "BRANCH"}, "Formal run role is unregistered", observed=role)
    execution_id = manifest.get("execution_id")
    execution_snapshot_digest = manifest.get("execution_attempt_snapshot_digest")
    execution_job_digest = manifest.get("execution_job_binding_digest")
    execution_claim_sha = manifest.get("execution_claim_sha256")
    run_intent_snapshot_digest = manifest.get("run_intent_snapshot_digest")
    run_intent_acknowledgement_id = manifest.get("run_intent_acknowledgement_id")
    _require(
        all(
            isinstance(value, str) and bool(value)
            for value in (execution_id, execution_snapshot_digest, execution_job_digest, execution_claim_sha)
        ),
        "Formal run is missing its execution attempt evidence",
    )
    _require(
        isinstance(run_intent_snapshot_digest, str)
        and len(run_intent_snapshot_digest) == 64
        and isinstance(run_intent_acknowledgement_id, str)
        and bool(run_intent_acknowledgement_id),
        "Formal run is missing its operator run-intent evidence",
    )
    try:
        from .formal_execution import validate_execution_attempt_snapshot

        execution_evidence = validate_execution_attempt_snapshot(
            root,
            execution_id=execution_id,
            expected_snapshot_digest=execution_snapshot_digest,
            expected_job_binding_digest=execution_job_digest,
            expected_claim_sha256=execution_claim_sha,
        )
    except SctsrError as exc:
        raise _closeout_failure("Formal execution attempt evidence is invalid", observed=str(exc)) from exc
    identity_path = root / "FORMAL_IDENTITY.json"
    binding_path = root / "FORMAL_AUTHORIZATION_BINDING.json"
    trainer_binding_path = root / "PREPARED_TRAINER_BINDING.json"
    generation_index_path = root / "ARTIFACT_INDEX_GENERATIONS.json"
    _require(identity_path.is_file() and binding_path.is_file() and trainer_binding_path.is_file() and generation_index_path.is_file(), "Formal identity, authorization, trainer, or generation index is missing")
    identity = load_json(identity_path)
    bindings = load_json(binding_path)
    trainer_binding = load_json(trainer_binding_path)
    trainer_core = {key: value for key, value in trainer_binding.items() if key != "binding_digest"}
    _require(trainer_binding.get("binding_digest") == stable_digest(trainer_core), "Prepared trainer binding digest is invalid")
    _require(trainer_binding.get("formal_training_started") is False, "Prepared trainer binding improperly claims training started")
    _require(trainer_binding.get("training_seed") == manifest.get("training_seed"), "Prepared trainer seed differs from formal run")
    trainer_output = Path(str(trainer_binding.get("output_root", "")))
    _require(trainer_output.exists() and trainer_output.samefile(root), "Prepared trainer output root differs from formal run root")
    _require(bindings == manifest.get("release_expected_bindings"), "Formal authorization binding differs from the run manifest")
    lock_path = Path(str(trainer_binding.get("canonical_training_lock_path", ""))).resolve()
    try:
        repository_root = lock_path.parents[2]
    except IndexError as exc:
        raise _closeout_failure("Prepared trainer lock path cannot identify the repository root") from exc
    _require(
        (repository_root / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json").resolve() == lock_path,
        "Prepared trainer lock path is not rooted in the canonical repository layout",
    )
    from .run_intent import validate_run_intent_snapshot_chain

    run_intent_evidence = validate_run_intent_snapshot_chain(
        root,
        repository_root=repository_root,
        expected_latest_snapshot_digest=run_intent_snapshot_digest,
    )
    _require(
        run_intent_evidence["attempts"][-1]["acknowledgement_id"] == run_intent_acknowledgement_id,
        "Formal run manifest names a different run-intent acknowledgement",
    )
    from .formal_cli import (
        validate_formal_authorization_inputs_at_closeout,
        validate_prepared_trainer_external_files,
    )
    from .formal_training import validate_formal_input_snapshot

    input_snapshot = validate_formal_input_snapshot(root)
    _require(
        manifest.get("formal_input_snapshot_digest") == input_snapshot["snapshot_digest"]
        and manifest.get("formal_input_external_binding_digest") == input_snapshot["external_binding_digest"],
        "Formal run manifest does not bind its immutable authorization-input snapshot",
    )
    validate_prepared_trainer_external_files(trainer_binding)
    authorization_inputs = validate_formal_authorization_inputs_at_closeout(
        load_json(root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json")["external_binding"],
        repository_root=repository_root,
        expected_bindings=bindings,
    )
    identity_binding = {
        "source_tree_digest": identity.get("source_tree_digest"),
        "contract_digest": identity.get("contract_digest"),
        "asset_registry_digest": identity.get("asset_registry_digest"),
        "runtime_config_digest": identity.get("runtime_config_digest"),
        "seed_registry_digest": identity.get("seed_registry_digest"),
    }
    for field, observed in identity_binding.items():
        _require(observed == manifest.get(field) == bindings.get(field), "Formal identity chain differs across manifest/release binding", observed={"field": field, "identity": observed, "manifest": manifest.get(field), "binding": bindings.get(field)})
    _require(identity.get("training_seed") == manifest.get("training_seed"), "Formal identity seed differs from run manifest")
    receipt_name = "PARENT_RECEIPT.json" if role == "COMMON_PARENT" else "BRANCH_RECEIPT.json"
    receipt_path = root / receipt_name
    _require(receipt_path.is_file(), "Formal run terminal receipt is missing", observed=receipt_name)
    receipt = load_json(receipt_path)
    expected_receipt_schema = (
        "stage1.sctsr.formal_parent_receipt.v3"
        if role == "COMMON_PARENT"
        else "stage1.sctsr.formal_branch_receipt.v3"
    )
    _require(receipt.get("schema_version") == expected_receipt_schema, "Formal terminal receipt schema is stale or invalid")
    expected_pending_status = (
        "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION"
        if role == "COMMON_PARENT"
        else "FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT"
    )
    _require(
        receipt.get("status") == expected_pending_status,
        "Formal run-state receipt claims completion before the atomic completion marker",
        observed=receipt.get("status"),
        expected=expected_pending_status,
    )
    from .formal_completion import validate_formal_completion

    completion = validate_formal_completion(root, expected_run_role=str(role))
    expected = (1, 120) if role == "COMMON_PARENT" else (121, 200)
    _require((receipt.get("epoch_start"), receipt.get("epoch_end")) == expected, "Formal receipt epoch range is incomplete", observed=(receipt.get("epoch_start"), receipt.get("epoch_end")), expected=expected)
    _require(receipt.get("epoch_evidence_enabled") is True and receipt.get("best_pt_used") is False, "Formal receipt lacks mandatory evidence or used best.pt")
    _require(receipt.get("prepared_trainer_binding_digest") == trainer_binding.get("binding_digest"), "Formal receipt does not bind prepared trainer evidence")
    _require(
        receipt.get("formal_input_snapshot_digest") == input_snapshot["snapshot_digest"],
        "Formal terminal receipt does not bind its authorization-input snapshot",
    )
    _require(
        receipt.get("run_intent_snapshot_digest") == run_intent_evidence["latest_snapshot_digest"]
        and receipt.get("run_intent_acknowledgement_id") == run_intent_acknowledgement_id,
        "Formal terminal receipt does not bind its latest run-intent attempt",
    )
    transactions = sorted((root / "03_epoch_transactions").glob("epoch_*.generation_*.complete"))
    _require(len(transactions) == expected[1] - expected[0] + 1, "Formal epoch transaction count is incomplete", observed=len(transactions), expected=expected[1] - expected[0] + 1)
    generation_index = load_json(generation_index_path)
    generations = generation_index.get("epoch_generations", [])
    _require(
        generation_index.get("schema_version") == "stage1.sctsr.epoch_artifact_index.v1"
        and len(generations) == len(transactions)
        and generation_index.get("epoch_generation_index_digest") == stable_digest(generations),
        "Formal generation index is invalid",
    )
    chain = validate_receipt_chain(root / "08_receipts" / "epoch_receipts.jsonl")
    _require(chain["row_count"] == len(transactions), "Formal receipt-chain length differs from epoch count")
    _require(receipt.get("epoch_receipt_digest") == chain["receipt_chain_digest"], "Terminal receipt does not bind the append-only epoch receipt chain")
    resume_chain_path = root / "08_receipts" / "resume_bindings.jsonl"
    resumed_from_epoch = receipt.get("resumed_from_epoch")
    if resumed_from_epoch is None:
        _require(receipt.get("resume_binding_receipt_digest") is None and not resume_chain_path.exists(), "Fresh run contains unregistered resume evidence")
    else:
        _require(resume_chain_path.is_file(), "Resumed run lacks its prepared-trainer resume chain")
        previous_resume_digest = "0" * 64
        resume_rows: list[dict[str, Any]] = []
        try:
            with resume_chain_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    _require(line.endswith("\n"), "Resume receipt row lacks a newline terminator")
                    row = json.loads(line)
                    claimed = str(row.pop("receipt_digest"))
                    _require(
                        row.get("previous_receipt_digest") == previous_resume_digest and stable_digest(row) == claimed,
                        "Resume prepared-trainer receipt chain is invalid",
                    )
                    row["receipt_digest"] = claimed
                    previous_resume_digest = claimed
                    resume_rows.append(row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _closeout_failure("Resume prepared-trainer receipt chain is unreadable", observed=str(exc)) from exc
        _require(bool(resume_rows), "Resumed run has an empty prepared-trainer receipt chain")
        latest_resume = resume_rows[-1]
        _require(
            previous_resume_digest == receipt.get("resume_binding_receipt_digest")
            and int(latest_resume.get("resume_epoch", -1)) - 1 == int(resumed_from_epoch)
            and latest_resume.get("original_prepared_trainer_binding_digest") == trainer_binding.get("binding_digest")
            and latest_resume.get("run_id") == manifest.get("run_id")
            and latest_resume.get("arm_id") == manifest.get("arm_id")
            and latest_resume.get("training_seed") == manifest.get("training_seed"),
            "Terminal receipt does not bind the latest valid resume attempt",
        )
        for row in resume_rows:
            resume_binding = row.get("resume_prepared_trainer_binding")
            _require(isinstance(resume_binding, Mapping), "Resume receipt has no prepared-trainer binding")
            binding_core = {key: value for key, value in resume_binding.items() if key != "binding_digest"}
            _require(
                resume_binding.get("binding_digest") == stable_digest(binding_core)
                and row.get("resume_prepared_trainer_binding_digest") == resume_binding.get("binding_digest"),
                "Resume prepared-trainer binding digest is invalid",
            )
            for field in (
                "upstream_binding_digest",
                "canonical_training_lock_sha256",
                "initial_checkpoint_sha256",
                "scientific_overrides_digest",
                "identity_manifest_binding",
                "dataset_binding",
                "dataset_content_binding",
                "training_seed",
            ):
                _require(resume_binding.get(field) == trainer_binding.get(field), "Resume trainer changed a stable scientific binding", observed=field)
            setup_root = Path(str(resume_binding.get("output_root", ""))).resolve()
            try:
                setup_root.relative_to((root / "10_resume_setup").resolve())
            except ValueError as exc:
                raise _closeout_failure("Resume trainer setup root escapes the run-root resume namespace") from exc
            _require(setup_root.is_dir(), "Resume trainer setup evidence is missing", observed=setup_root.as_posix())
    schedule = None
    if role == "BRANCH":
        schedule_path = root / "SCHEDULE.json"
        lineage_path = root / "BRANCH_LINEAGE.json"
        pool_binding_path = root / "IDENTITY_POOL_BINDING.json"
        parent_binding_path = root / "PARENT_ARTIFACT_INDEX_BINDING.json"
        _require(
            schedule_path.is_file() and lineage_path.is_file() and pool_binding_path.is_file() and parent_binding_path.is_file(),
            "Formal branch lacks schedule, lineage, identity-pool, or parent-index snapshot",
        )
        schedule = schedule_from_dict(load_json(schedule_path))
        _require(schedule.arm_id.value == manifest.get("arm_id") and schedule.plan_digest == load_json(schedule_path).get("plan_digest"), "Formal branch schedule identity is invalid")
        pool_binding = load_json(pool_binding_path)
        parent_binding = load_json(parent_binding_path)
        pool_core = {key: value for key, value in pool_binding.items() if key != "binding_digest"}
        parent_core = {key: value for key, value in parent_binding.items() if key != "binding_digest"}
        _require(pool_binding.get("binding_digest") == stable_digest(pool_core), "Formal identity-pool binding digest is invalid")
        _require(parent_binding.get("binding_digest") == stable_digest(parent_core), "Formal parent-index binding digest is invalid")
        _require(parent_binding.get("parent_checkpoint_sha256") == receipt.get("parent_checkpoint_sha256"), "Formal parent-index binding checkpoint differs from branch receipt")
        _require(pool_binding.get("arm_id") == schedule.arm_id.value, "Formal identity-pool binding arm differs from schedule")
        _require(
            receipt.get("identity_pool_binding_digest") == pool_binding.get("binding_digest")
            and receipt.get("parent_artifact_index_binding_digest") == parent_binding.get("binding_digest"),
            "Formal branch receipt does not bind its pool/parent evidence",
        )
        from .formal_cli import (
            load_lineage,
            validate_identity_pool_artifacts,
            validate_parent_artifact_index,
            validate_training_identity_manifest,
        )
        from .formal_pool_inputs import load_formal_pool_inputs

        pool_manifest_paths = [
            row["manifest_path"]
            for _role, row in sorted(pool_binding.get("artifact_bindings", {}).items())
        ]
        recomputed_pool = validate_identity_pool_artifacts(
            pool_manifest_paths,
            schedule=schedule,
            expected_base_denominator=120_000,
            expected_base_manifest_sha256=str(identity["base_manifest_sha256"]),
        )
        _require(recomputed_pool == pool_binding, "Formal identity-pool external bytes changed after branch setup")
        recomputed_parent = validate_parent_artifact_index(
            parent_checkpoint=parent_binding.get("parent_checkpoint_path", ""),
            parent_artifact_index=parent_binding.get("artifact_index_path", ""),
        )
        _require(recomputed_parent == parent_binding, "Formal parent artifact tree changed after branch setup")
        lineage = load_lineage(lineage_path)
        lineage.validate(
            parent_sha=str(receipt["parent_checkpoint_sha256"]),
            training_seed=int(manifest["training_seed"]),
            arm_id=str(manifest["arm_id"]),
            source_digest=str(identity["source_tree_digest"]),
            contract_digest=str(identity["contract_digest"]),
        )
        _require(lineage.lineage_digest == receipt.get("lineage_digest"), "Formal lineage digest differs from terminal receipt")
        pool_inputs = load_formal_pool_inputs(authorization_inputs["asset_registry"], repository_root)
        recomputed_identity_manifest = validate_training_identity_manifest(
            trainer_binding["identity_manifest_binding"]["path"],
            base_records=pool_inputs.base_records,
            pool_manifest_paths=pool_manifest_paths,
            schedule=schedule,
            base_denominator=120_000,
            base_manifest_sha256=pool_inputs.base_manifest_sha256,
        )
        _require(
            recomputed_identity_manifest == trainer_binding["identity_manifest_binding"],
            "Formal trainer identity manifest no longer matches registered assets and pools",
        )
    else:
        from .formal_cli import validate_training_identity_manifest
        from .formal_pool_inputs import load_formal_pool_inputs

        pool_inputs = load_formal_pool_inputs(authorization_inputs["asset_registry"], repository_root)
        recomputed_identity_manifest = validate_training_identity_manifest(
            trainer_binding["identity_manifest_binding"]["path"],
            base_records=pool_inputs.base_records,
            pool_manifest_paths=(),
            schedule=None,
            base_denominator=120_000,
            base_manifest_sha256=pool_inputs.base_manifest_sha256,
        )
        _require(
            recomputed_identity_manifest == trainer_binding["identity_manifest_binding"],
            "Formal parent trainer identity manifest no longer matches registered assets",
        )
    from .dataset_content_ledger import registered_dataset_manifest_asset_ids, validate_registered_dataset_content

    recomputed_dataset_content = validate_registered_dataset_content(
        registry=authorization_inputs["asset_registry"],
        repository_root=repository_root,
        dataset_root=trainer_binding["dataset_content_binding"]["dataset_root"],
        required_manifest_asset_ids=registered_dataset_manifest_asset_ids(authorization_inputs["asset_registry"]),
        verify_physical_files=True,
    )
    _require(
        recomputed_dataset_content == trainer_binding["dataset_content_binding"],
        "Formal dataset image bytes changed after prepared-trainer preflight",
    )
    previous_checkpoint = str(identity["initial_checkpoint_sha256"]) if role == "COMMON_PARENT" else str(receipt["parent_checkpoint_sha256"])
    previous_generation = stable_digest(
        {"role": "COMMON_PARENT_START", "initial_checkpoint_sha256": identity["initial_checkpoint_sha256"]}
        if role == "COMMON_PARENT"
        else {"role": "BRANCH_START", "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"], "lineage_digest": receipt["lineage_digest"]}
    )
    final_checkpoint_sha = None
    total_occurrences = 0
    total_replay = 0
    replay_counts: dict[str, int] = {}
    replay_last_epoch: dict[str, int] = {}
    replay_history_cumulative = 0
    for offset, (epoch, transaction) in enumerate(zip(range(expected[0], expected[1] + 1), transactions, strict=True)):
        _require(transaction.name == f"epoch_{epoch:04d}.generation_1.complete", "Formal epoch generation name is noncanonical", observed=transaction.name)
        generation = transaction / "GENERATION_MANIFEST.json"
        _require(generation.is_file(), "Completed formal generation lacks its generation manifest", observed=transaction.name)
        generation_raw = load_json(generation)
        generation_without_digest = {key: value for key, value in generation_raw.items() if key != "generation_digest"}
        _require(generation_raw.get("generation_digest") == stable_digest(generation_without_digest), "Formal generation digest is invalid", observed=epoch)
        indexed = generations[offset]
        _require(
            indexed.get("run_id") == manifest.get("run_id")
            and int(indexed.get("epoch", -1)) == epoch
            and int(indexed.get("generation", -1)) == 1
            and indexed.get("generation_digest") == generation_raw.get("generation_digest")
            and indexed.get("generation_manifest_sha256") == sha256_file(generation),
            "Formal generation index row differs from its transaction",
            observed=epoch,
        )
        transaction_identity = generation_raw.get("identity", {})
        _require(
            transaction_identity.get("parent_sha256") == previous_checkpoint
            and transaction_identity.get("previous_generation_digest") == previous_generation
            and transaction_identity.get("training_seed") == identity.get("training_seed")
            and transaction_identity.get("source_tree_digest") == identity.get("source_tree_digest")
            and transaction_identity.get("contract_digest") == identity.get("contract_digest")
            and transaction_identity.get("asset_registry_digest") == identity.get("asset_registry_digest"),
            "Formal transaction identity chain is broken",
            observed=epoch,
        )
        run_id = str(manifest["run_id"])
        required_relatives = {
            "TRANSACTION_IDENTITY.json",
            f"04_ledgers/occurrence/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet",
            f"04_ledgers/optimizer_step/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet",
            f"04_ledgers/telemetry/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet",
            f"04_ledgers/exposure/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet",
            "EPOCH_EVIDENCE_SUMMARY.json",
            f"05_checkpoints/rolling_epoch_{epoch:04d}.generation_1.pt",
        }
        file_rows = generation_raw.get("files", [])
        _require({row.get("path") for row in file_rows} == required_relatives, "Formal epoch transaction artifact set is incomplete", observed=epoch)
        file_by_path = {row["path"]: row for row in file_rows}
        for relative, file_row in file_by_path.items():
            artifact = transaction / relative
            _require(artifact.is_file() and artifact.stat().st_size == int(file_row["bytes"]) and sha256_file(artifact) == file_row["sha256"], "Formal transaction file identity mismatch", observed={"epoch": epoch, "path": relative})
        occurrence_path = transaction / f"04_ledgers/occurrence/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet"
        step_path = transaction / f"04_ledgers/optimizer_step/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet"
        telemetry_path = transaction / f"04_ledgers/telemetry/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet"
        exposure_path = transaction / f"04_ledgers/exposure/run_id={run_id}/epoch={epoch:04d}/part-00000.parquet"
        checkpoint_path = transaction / f"05_checkpoints/rolling_epoch_{epoch:04d}.generation_1.pt"
        occurrence = read_columnar(occurrence_path)
        steps = read_columnar(step_path)
        telemetry = read_columnar(telemetry_path)
        exposure = read_columnar(exposure_path)
        validate_occurrence_rows(occurrence)
        validate_step_rows(steps)
        validate_telemetry_for_closeout(_telemetry_rows(telemetry))
        validate_exposure_rows(exposure)
        _require(len(steps) == 938 and len(exposure) == 1, "Formal epoch does not contain 938 steps and one exposure row", observed=epoch)
        base_count = sum(row["occurrence_role"] == "BASE" for row in occurrence)
        replay_count = len(occurrence) - base_count
        for row_index, occurrence_row in enumerate(occurrence):
            _require(
                int(occurrence_row["cumulative_replay_count_before"]) == replay_history_cumulative,
                "Formal occurrence replay history is discontinuous",
                observed={"epoch": epoch, "row": row_index},
            )
            if occurrence_row["occurrence_role"] == "REPLAY":
                sample_id = str(occurrence_row["sample_id"])
                previous_count = replay_counts.get(sample_id, 0)
                previous_epoch = replay_last_epoch.get(sample_id)
                _require(
                    int(occurrence_row["replay_count_before"]) == previous_count
                    and int(occurrence_row["replay_count_after"]) == previous_count + 1
                    and occurrence_row["last_replay_epoch"] == previous_epoch,
                    "Formal per-sample replay history is discontinuous",
                    observed={"epoch": epoch, "row": row_index, "sample_id": sample_id},
                )
                replay_counts[sample_id] = previous_count + 1
                replay_last_epoch[sample_id] = epoch
                replay_history_cumulative += 1
            _require(
                int(occurrence_row["cumulative_replay_count_after"]) == replay_history_cumulative,
                "Formal occurrence cumulative-after replay history is invalid",
                observed={"epoch": epoch, "row": row_index},
            )
        _require(base_count == 120_000, "Formal base occurrence denominator changed", observed={"epoch": epoch, "base": base_count})
        exposure_row = exposure[0]
        _require(
            int(exposure_row["base_optimizer_steps_actual"]) == 938
            and int(exposure_row["replay_numerator_actual"]) == replay_count
            and exposure_row["occurrence_partition_sha256"] == sha256_file(occurrence_path)
            and exposure_row["step_partition_sha256"] == sha256_file(step_path)
            and exposure_row["telemetry_partition_sha256"] == sha256_file(telemetry_path)
            and exposure_row["checkpoint_sha256"] == sha256_file(checkpoint_path),
            "Formal exposure cross-bindings failed",
            observed=epoch,
        )
        if role == "COMMON_PARENT":
            _require(replay_count == 0 and int(exposure_row["rate_numerator"]) == 0, "Common parent contains replay", observed=epoch)
        else:
            assert schedule is not None
            plan = schedule.epoch(epoch)
            _require(
                replay_count == plan.replay_occurrences
                and int(exposure_row["rate_numerator"]) == plan.rate.numerator
                and int(exposure_row["rate_denominator"]) == plan.rate.denominator
                and exposure_row["replay_schedule_digest"] == schedule.plan_digest
                and exposure_row["identity_pool_digest"] == schedule.identity_pool_digest,
                "Formal branch exposure differs from its frozen schedule",
                observed=epoch,
            )
        checkpoint_payload = load_checkpoint(checkpoint_path, expected_sha256=sha256_file(checkpoint_path), expected_epoch=epoch)
        _require(
            checkpoint_payload["training_seed"] == identity["training_seed"]
            and checkpoint_payload["source_tree_digest"] == identity["source_tree_digest"]
            and checkpoint_payload["runtime_config_digest"] == identity["runtime_config_digest"]
            and checkpoint_payload["asset_registry_digest"] == identity["asset_registry_digest"],
            "Formal checkpoint identity differs from the prepared run",
            observed=epoch,
        )
        summary = load_json(transaction / "EPOCH_EVIDENCE_SUMMARY.json")
        _require(
            summary.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            and summary.get("occurrence_partition", {}).get("sha256") == sha256_file(occurrence_path)
            and summary.get("step_partition", {}).get("sha256") == sha256_file(step_path)
            and summary.get("telemetry_partition", {}).get("sha256") == sha256_file(telemetry_path)
            and summary.get("exposure_partition", {}).get("sha256") == sha256_file(exposure_path),
            "Formal epoch evidence summary cross-binding failed",
            observed=epoch,
        )
        _require(
            summary.get("history_after_epoch")
            == {
                "counts": dict(sorted(replay_counts.items())),
                "last_epoch": dict(sorted(replay_last_epoch.items())),
                "cumulative_occurrences": replay_history_cumulative,
            },
            "Formal epoch history summary differs from occurrence bytes",
            observed=epoch,
        )
        rng_evidence = summary.get("rng_evidence", {})
        checkpoint_rng_digest = checkpoint_payload["rng_state"].digest()
        _require(
            rng_evidence.get("recorder_epoch_start_digest") == transaction_identity.get("rng_state_digest")
            and rng_evidence.get("runtime_epoch_start_digest") == transaction_identity.get("rng_state_digest")
            and rng_evidence.get("runtime_epoch_end_digest") == checkpoint_rng_digest
            and rng_evidence.get("finalize_entry_digest") == checkpoint_rng_digest,
            "Formal epoch RNG evidence is not closed by its checkpoint",
            observed=epoch,
        )
        previous_checkpoint = sha256_file(checkpoint_path)
        previous_generation = str(generation_raw["generation_digest"])
        final_checkpoint_sha = previous_checkpoint
        total_occurrences += len(occurrence)
        total_replay += replay_count
    pointer = validate_recovery_pointer(root / "ROLLING_RECOVERY_POINTER.json")
    _require(pointer.get("epoch") == expected[1] and pointer.get("generation") == 1, "Formal recovery pointer does not identify the final complete epoch")
    endpoint_evidence = None
    if role == "COMMON_PARENT":
        _require(receipt.get("checkpoint_sha256") == final_checkpoint_sha, "Parent terminal checkpoint differs from E120 transaction")
    else:
        fixed = receipt.get("fixed_formal_endpoint", {})
        _require(fixed.get("sha256") == final_checkpoint_sha and not str(fixed.get("path", "")).lower().endswith("best.pt"), "Branch fixed endpoint differs from E200 transaction")
        logical_raw = load_json(root / "ARTIFACT_INDEX_LOGICAL.json")
        logical_entries = [LogicalArtifactEntry(**row) for row in logical_raw.get("logical_timeline", [])]
        logical = LogicalArtifactIndex(logical_entries)
        logical.validate(require_complete_timeline=True, logical_run_id=str(manifest["run_id"]))
        _require(logical.digest == logical_raw.get("logical_timeline_digest"), "Formal logical timeline digest is invalid")
        parent_root = Path(str(logical_raw.get("physical_parent_root", ""))).resolve()
        child_root = Path(str(logical_raw.get("physical_child_root", ""))).resolve()
        _require(child_root == root, "Formal logical timeline points to a different child root")
        for entry in logical_entries:
            physical_root = parent_root if entry.physical_owner_type == "PARENT" else child_root
            physical = physical_root / entry.artifact_relative_path
            _require(physical.is_file() and sha256_file(physical) == entry.artifact_sha256, "Formal logical timeline physical binding failed", observed=entry.logical_epoch)
        endpoint_evidence = validate_formal_endpoint_evidence(
            root,
            manifest=manifest,
            checkpoint_path=checkpoint_path,
            repository_root=repository_root,
        )
    return {
        "formal_semantic_status": "PASS",
        "run_role": role,
        "execution_id": execution_id,
        "execution_attempt_snapshot_digest": execution_evidence["snapshot_digest"],
        "run_intent_attempt_count": run_intent_evidence["attempt_count"],
        "run_intent_snapshot_digest": run_intent_evidence["latest_snapshot_digest"],
        "epoch_transaction_count": len(transactions),
        "receipt_chain_digest": chain["receipt_chain_digest"],
        "total_optimizer_visible_occurrences": total_occurrences,
        "total_replay_occurrences": total_replay,
        "fixed_endpoint_checkpoint_sha256": final_checkpoint_sha,
        "formal_completion_receipt_sha256": completion["receipt_sha256"],
        "endpoint_evidence": endpoint_evidence,
    }


def validate_run_tree(run_root: str | Path, *, allow_synthetic_portable_fallback: bool = False) -> dict[str, Any]:
    from .filesystem import windows_safe_resolved_path

    root = windows_safe_resolved_path(run_root)
    if not root.is_dir():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Run root is missing", artifact_path=str(root))
    manifest_path = root / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        raise _closeout_failure("Run manifest is missing")
    manifest = load_json(manifest_path)
    semantic = manifest.get("semantic") or manifest.get("execution_mode")
    if semantic not in {SYNTHETIC_SEMANTIC, "formal"}:
        raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Run has no registered execution semantic")
    _, checked = _validate_exhaustive_index(root, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
    try:
        if semantic == SYNTHETIC_SEMANTIC:
            semantic_report = _validate_synthetic_canary(
                root,
                manifest,
                allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
            )
        else:
            semantic_report = _validate_formal_tree(root, manifest)
    except SctsrError as exc:
        if exc.code is ErrorCode.CLOSEOUT_NOT_VALIDATED:
            raise
        raise _closeout_failure(
            "Run passed file hashing but failed semantic validation",
            observed={"cause_code": exc.code.value, "cause_message": exc.message},
        ) from exc
    return {
        "status": "PASS",
        "run_root": root.as_posix(),
        "semantic": semantic,
        "artifact_count": len(checked),
        "artifact_digest": stable_digest(checked),
        "manifest_sha256": sha256_file(manifest_path),
        "index_sha256": sha256_file(root / "ARTIFACT_INDEX.json"),
        **semantic_report,
    }
