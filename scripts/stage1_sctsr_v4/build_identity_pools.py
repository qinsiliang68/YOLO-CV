from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.columnar import write_zstd_parquet
from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.identity_pool import partition_five_groups
from stage1_sctsr_v4.formal_pool_inputs import build_registered_r1, build_registered_r2, load_formal_pool_inputs
from stage1_sctsr_v4.random_controls import counter_hash
from stage1_sctsr_v4.r2_addendum import validate_r2_matching_policy_mapping
from stage1_sctsr_v4.selection_ledger import write_selection_partition
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file
from stage1_sctsr_v4.synthetic_fixture import build_synthetic_fixture
from stage1_sctsr_v4.terminal_field_guard import TerminalFieldGuard


CHOICES = ("T_STRESS", "R1_GLOBAL_RANDOM", "R2_MATCHED_RANDOM", "CURRENT_LOSS_HELD")


def _canonical_sha(value: str, *, field: str) -> str:
    token = str(value)
    if len(token) != 64 or token != token.upper() or any(character not in "0123456789ABCDEF" for character in token):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Formal pool input is not bound by a canonical SHA-256", failing_field=field, observed=value)
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description="Build T, R1, or owner-approved zero-overlap minimum-group-displacement R2 identity pools")
    parser.add_argument("--pool", choices=CHOICES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--asset-registry", type=Path)
    parser.add_argument("--base-records", type=Path)
    parser.add_argument("--t-records", type=Path)
    parser.add_argument("--base-manifest-sha", default="UNREGISTERED")
    parser.add_argument("--source-manifest-sha", default="UNREGISTERED")
    parser.add_argument("--selection-seed", type=int, default=20260812)
    parser.add_argument("--r2-policy", type=Path, help="Required canonical owner-approved policy for formal R2 only")
    parser.add_argument("--validate-only", action="store_true")
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.pool == "CURRENT_LOSS_HELD":
            if not arguments.validate_only:
                raise SctsrError(ErrorCode.CURRENT_LOSS_HELD, "CURRENT_LOSS_HELD may only be validated, never built in phase 1")
            if arguments.output_dir.exists():
                raise SctsrError(ErrorCode.CURRENT_LOSS_HELD, "HELD validation may not write a pool output directory")
            return {
                "state": "HELD_NOT_IN_PHASE1",
                "semantic": "CANDIDATE_SIGNAL_NOT_UTILITY",
                "formal_training_started": False,
            }
        if arguments.validate_only:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "--validate-only is registered only for CURRENT_LOSS_HELD")
        if arguments.output_dir.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Identity-pool destination already exists", artifact_path=str(arguments.output_dir))

        if arguments.synthetic:
            if arguments.r2_policy is not None:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Synthetic pools may not claim the formal R2 addendum policy")
            os.environ["SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK"] = "1"
            fixture = build_synthetic_fixture(training_seed=arguments.selection_seed)
            base = fixture.base_records
            t_pool = fixture.t_pool
            result = {
                "T_STRESS": fixture.t_pool,
                "R1_GLOBAL_RANDOM": fixture.r1_result.pool,
                "R2_MATCHED_RANDOM": fixture.r2_result.pool,
            }[arguments.pool]
            audit = None if arguments.pool == "T_STRESS" else asdict(
                {"R1_GLOBAL_RANDOM": fixture.r1_result, "R2_MATCHED_RANDOM": fixture.r2_result}[arguments.pool].audit
            )
            denominator = fixture.base_denominator
            source_bindings = {
                "base_records": "SYNTHETIC_IN_MEMORY",
                "base_records_sha256": "SYNTHETIC_IN_MEMORY",
                "t_records": "SYNTHETIC_IN_MEMORY",
                "t_records_sha256": "SYNTHETIC_IN_MEMORY",
            }
        else:
            base_sha = _canonical_sha(arguments.base_manifest_sha, field="base_manifest_sha")
            source_sha = _canonical_sha(arguments.source_manifest_sha, field="source_manifest_sha")
            if arguments.repository_root is None or arguments.asset_registry is None:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal pool construction requires the registered repository root and asset registry")
            if arguments.base_records is not None or arguments.t_records is not None:
                raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "Formal pool construction may not accept arbitrary base/T tables; use the registered asset projection")
            repository_root = arguments.repository_root.resolve()
            r2_policy_binding = None
            if arguments.pool == "R2_MATCHED_RANDOM":
                if arguments.r2_policy is None:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal R2 construction requires --r2-policy")
                expected_policy_path = (repository_root / "configs/stage1_sctsr_v4/r2_matching_policy_v1.json").resolve()
                policy_path = arguments.r2_policy.resolve()
                if policy_path != expected_policy_path or not policy_path.is_file():
                    raise SctsrError(
                        ErrorCode.CONFIGURATION_MISMATCH,
                        "Formal R2 policy must be the canonical repository file, not a copy or alternate path",
                        observed=policy_path.as_posix(),
                        expected=expected_policy_path.as_posix(),
                    )
                policy = validate_r2_matching_policy_mapping(load_json(policy_path))
                if arguments.selection_seed != policy["selection_seed"]:
                    raise SctsrError(
                        ErrorCode.CONFIGURATION_MISMATCH,
                        "R2 CLI selection seed differs from the owner-approved policy",
                        observed=arguments.selection_seed,
                        expected=policy["selection_seed"],
                    )
                r2_policy_binding = {
                    "path": policy_path.as_posix(),
                    "bytes": policy_path.stat().st_size,
                    "sha256": sha256_file(policy_path),
                    "policy_digest": policy["policy_digest"],
                    "policy_id": policy["policy_id"],
                }
            elif arguments.r2_policy is not None:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "--r2-policy may only be supplied for R2_MATCHED_RANDOM")
            registry = load_asset_registry(arguments.asset_registry)
            inputs = load_formal_pool_inputs(registry, repository_root)
            expected_source_sha = inputs.t_pool.spec.source_manifest_sha256 if arguments.pool == "T_STRESS" else inputs.preterminal_source_sha256
            if base_sha != inputs.base_manifest_sha256 or source_sha != expected_source_sha:
                raise SctsrError(
                    ErrorCode.IDENTITY_DIGEST_MISMATCH,
                    "Formal pool CLI bindings differ from the validated asset projection",
                    observed={"base": base_sha, "source": source_sha},
                    expected={"base": inputs.base_manifest_sha256, "source": expected_source_sha},
                )
            base = inputs.base_records
            denominator = registry.base_denominator
            base_ids = {record.sample_id for record in base}
            if len(base_ids) != denominator:
                raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Formal base record count/identity uniqueness differs from the frozen denominator", observed=len(base_ids), expected=denominator)
            t_pool = inputs.t_pool
            if arguments.pool == "T_STRESS":
                result, audit = t_pool, None
            elif arguments.pool == "R1_GLOBAL_RANDOM":
                built = build_registered_r1(inputs, base_denominator=denominator, selection_seed=arguments.selection_seed)
                result, audit = built.pool, asdict(built.audit)
            else:
                built = build_registered_r2(inputs, base_denominator=denominator, selection_seed=arguments.selection_seed)
                result, audit = built.pool, asdict(built.audit)
            result.validate(
                base_denominator=denominator,
                base_ids=base_ids,
                t_ids={record.sample_id for record in t_pool.records},
            )
            source_bindings = {
                "asset_registry": arguments.asset_registry.resolve().as_posix(),
                "asset_registry_digest": inputs.asset_registry_digest,
                "base_manifest_sha256": inputs.base_manifest_sha256,
                "preterminal_source_sha256": inputs.preterminal_source_sha256,
                "t_source_manifest_path": inputs.t_pool.spec.source_manifest_path,
                "t_source_manifest_sha256": inputs.t_pool.spec.source_manifest_sha256,
            }
            if r2_policy_binding is not None:
                source_bindings["r2_matching_policy"] = r2_policy_binding

        groups = partition_five_groups(result, base_denominator=denominator)
        membership = [
            {
                **asdict(record),
                "identity_group": group,
                "pool_id": result.spec.pool_id,
                "pool_role": result.spec.pool_role,
            }
            for group, records in groups.items()
            for record in records
        ]
        run_id = f"POOL_{arguments.pool}_{arguments.selection_seed}"
        membership_path = arguments.output_dir / f"run_id={run_id}" / "epoch=0000" / "identity_group_membership.parquet"
        membership_manifest = write_zstd_parquet(
            membership,
            membership_path,
            schema_version="stage1.sctsr.identity_group_membership.v1",
            require_run_epoch_partition=True,
            allow_synthetic_portable_fallback=arguments.synthetic,
        )
        selected_ids = {record.sample_id for record in result.records}
        t_ids = {record.sample_id for record in t_pool.records}
        required = Counter("|".join(map(str, record.stratum())) for record in t_pool.records)
        available = Counter("|".join(map(str, record.stratum())) for record in base if record.sample_id not in t_ids)
        policy = arguments.pool
        if policy == "R2_MATCHED_RANDOM":
            terminal_status = "TERMINAL_FIELDS_NOT_LOADED"
            overlap_status = "ZERO_OVERLAP"
            guard_digest = TerminalFieldGuard().digest
        elif policy == "R1_GLOBAL_RANDOM":
            terminal_status = "TERMINAL_FIELDS_NOT_USED_GLOBAL_RANDOM"
            overlap_status = "NATURAL_OVERLAP_REPORTED"
            guard_digest = "NOT_APPLICABLE_GLOBAL_RANDOM"
        else:
            terminal_status = "TERMINAL_FIELDS_NOT_USED_CANONICAL_T"
            overlap_status = "CANONICAL_T"
            guard_digest = "NOT_APPLICABLE_CANONICAL_T"
        source_row_sha = "2" * 64 if arguments.synthetic else (
            inputs.preterminal_source_sha256 if policy != "T_STRESS" else inputs.t_pool.spec.source_manifest_sha256
        )
        selection_rows = []
        for record in base:
            stratum = "|".join(map(str, record.stratum()))
            eligible = record.base_manifest_membership and not (policy == "R2_MATCHED_RANDOM" and record.sample_id in t_ids)
            selection_rows.append(
                {
                    "candidate_sample_id": record.sample_id,
                    "eligibility": eligible,
                    "exclusion_reason": "ELIGIBLE" if eligible else "EXCLUDED_T_IDENTITY_FOR_R2",
                    "allowed_strata": stratum,
                    "stratum_quota_required": int(required.get(stratum, 0)),
                    "stratum_quota_available": int(available.get(stratum, 0)) if policy == "R2_MATCHED_RANDOM" else denominator,
                    "selection_counter_hash": counter_hash(policy, arguments.selection_seed if policy != "T_STRESS" else 0, stratum, record.sample_id),
                    "selected": record.sample_id in selected_ids,
                    "selected_pool": result.spec.pool_id if record.sample_id in selected_ids else "NOT_SELECTED",
                    "terminal_field_guard_digest": guard_digest,
                    "source_row_asset_sha256": source_row_sha,
                    "duplicate_overlap_status": overlap_status,
                    "terminal_field_status": terminal_status,
                    "row_generation": 1,
                }
            )
        selection_path = arguments.output_dir / "selection" / f"run_id={run_id}" / "epoch=0000" / f"{policy}.parquet"
        selection_manifest = write_selection_partition(selection_rows, selection_path, policy=policy)
        quota_audit = {
            "schema_version": "stage1.sctsr.identity_pool_quota_audit.v1",
            "pool_role": arguments.pool,
            "pool_digest": result.spec.identity_digest,
            "group_counts": {key: len(value) for key, value in groups.items()},
            "pool_build_audit": audit,
            "source_bindings": source_bindings,
            "terminal_field_status": "TERMINAL_FIELDS_NOT_LOADED" if arguments.pool == "R2_MATCHED_RANDOM" else "NOT_APPLICABLE",
            "overlap_with_t": 0 if arguments.pool == "R2_MATCHED_RANDOM" else "NATURAL_OVERLAP_ALLOWED_OR_SELF",
        }
        quota_audit_path = arguments.output_dir / "QUOTA_AUDIT.json"
        atomic_write_json(quota_audit_path, quota_audit)
        payload = {
            "schema_version": "stage1.sctsr.identity_pool_manifest.v1",
            "pool_spec": asdict(result.spec),
            "membership": asdict(membership_manifest),
            "selection": asdict(selection_manifest),
            "quota_audit": {
                "path": quota_audit_path.resolve().as_posix(),
                "bytes": quota_audit_path.stat().st_size,
                "sha256": sha256_file(quota_audit_path),
            },
            "audit": audit,
            "group_counts": quota_audit["group_counts"],
            "pool_digest": result.spec.identity_digest,
            "semantic": "SYNTHETIC_NOT_SCIENTIFIC_RESULT" if arguments.synthetic else result.spec.selection_semantic,
            "formal_training_started": False,
        }
        atomic_write_json(arguments.output_dir / "POOL_MANIFEST.json", payload)
        return payload

    return run_cli("build_identity_pools", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
