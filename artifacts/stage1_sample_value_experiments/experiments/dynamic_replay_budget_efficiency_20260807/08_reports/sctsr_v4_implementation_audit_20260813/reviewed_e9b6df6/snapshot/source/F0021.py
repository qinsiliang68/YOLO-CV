from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import _load_identity_pool_artifact, validate_identity_pool_artifacts, validate_training_identity_manifest
from stage1_sctsr_v4.formal_pool_inputs import load_formal_pool_inputs
from stage1_sctsr_v4.schedule import schedule_from_dict
from stage1_sctsr_v4.serialization import atomic_write_json, atomic_write_text, load_json, sha256_file, stable_digest


FIELDS = (
    "sample_id",
    "y_true",
    "source_path",
    "replay_role",
    "oof_fold",
    "oof_group_id",
    "historical_dynamic_bucket",
    "identity_group",
    "oof_reference_probability",
    "oof_reference_reason",
    "rho_candidate_signal",
    "rho_reason",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the canonical trainer identity manifest from registered pre-terminal assets")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--identity-pool", type=Path, action="append", default=[])
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        summary_output = arguments.summary_output or arguments.manifest_output.with_name("training_identity_manifest_summary.json")
        if arguments.manifest_output.exists() or summary_output.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Training identity manifest destination already exists")
        root = arguments.repository_root.resolve()
        registry = load_asset_registry(arguments.asset_registry)
        inputs = load_formal_pool_inputs(registry, root)
        schedule = None if arguments.schedule is None else schedule_from_dict(load_json(arguments.schedule))
        expected_groups: dict[str, str] = {}
        pool_bindings = {}
        for manifest_path in arguments.identity_pool:
            pool, groups, binding = _load_identity_pool_artifact(
                manifest_path,
                expected_base_denominator=registry.base_denominator,
                expected_base_manifest_sha256=inputs.base_manifest_sha256,
            )
            for group, records in groups.items():
                for record in records:
                    if record.sample_id in expected_groups:
                        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Multiple supplied pools annotate the same trainer identity", observed=record.sample_id)
                    expected_groups[record.sample_id] = group
            pool_bindings[pool.spec.pool_role] = dict(binding)
        if bool(arguments.identity_pool) != (schedule is not None and schedule.arm_id.value != "NR"):
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Trainer identity pools and schedule role are inconsistent")
        if schedule is not None:
            validate_identity_pool_artifacts(
                arguments.identity_pool,
                schedule=schedule,
                expected_base_denominator=registry.base_denominator,
                expected_base_manifest_sha256=inputs.base_manifest_sha256,
            )

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(inputs.base_records, key=lambda row: row.sample_id):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "y_true": record.y_true,
                    "source_path": record.sample_id,
                    "replay_role": record.replay_role,
                    "oof_fold": record.oof_fold,
                    "oof_group_id": record.oof_group_id,
                    "historical_dynamic_bucket": record.historical_dynamic_bucket,
                    "identity_group": expected_groups.get(record.sample_id, "UNASSIGNED"),
                    "oof_reference_probability": "",
                    "oof_reference_reason": "REGISTERED_NOT_AVAILABLE",
                    "rho_candidate_signal": "",
                    "rho_reason": "REGISTERED_NOT_REPORTED",
                }
            )
        atomic_write_text(arguments.manifest_output, buffer.getvalue())
        validation = validate_training_identity_manifest(
            arguments.manifest_output,
            base_records=inputs.base_records,
            pool_manifest_paths=arguments.identity_pool,
            schedule=schedule,
            base_denominator=registry.base_denominator,
            base_manifest_sha256=inputs.base_manifest_sha256,
        )
        summary = {
            "schema_version": "stage1.sctsr.training_identity_manifest_summary.v1",
            "status": "PASS",
            "manifest_path": arguments.manifest_output.resolve().as_posix(),
            "manifest_bytes": arguments.manifest_output.stat().st_size,
            "manifest_sha256": sha256_file(arguments.manifest_output),
            "row_count": registry.base_denominator,
            "base_manifest_sha256": inputs.base_manifest_sha256,
            "asset_registry_digest": registry.digest,
            "schedule_digest": None if schedule is None else schedule.plan_digest,
            "pool_bindings": pool_bindings,
            "pool_binding_digest": stable_digest(pool_bindings),
            "validation": validation,
            "formal_training_started": False,
        }
        atomic_write_json(summary_output, summary)
        return {**summary, "summary_path": summary_output.resolve().as_posix(), "summary_sha256": sha256_file(summary_output)}

    return run_cli("build_training_identity_manifest", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
