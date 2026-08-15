from __future__ import annotations

from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


R2_MATCHING_POLICY_SCHEMA = "stage1.sctsr.r2_matching_policy.v2"
R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY = "MINIMUM_OOF_GROUP_DISPLACEMENT_CONTENT_DISJOINT_V2"
R2_STRICT_EXACT_POLICY = "STRICT_EXACT_LABEL_DYNAMIC_FOLD_OOF_GROUP_ZERO_OVERLAP_V1"
R2_APPROVED_SELECTION_SEED = 20260812
R2_APPROVED_IDENTITY_DIGEST = "A6DAA20A70F02B30D15B7C3E4079EA86903051AEED264F53E0A104A4C1AA80B6"
R2_APPROVED_CONTENT_MAP_DIGEST = "751742DA58F13A2678CB700D018B95B5FA8E38392F15366B688CCEE97BE81CA7"
R2_APPROVED_SELECTED_CONTENT_DIGEST = "A48B721CA37AD66D65B8C5972C5AE66C328C09194BA3C8C22C19B8FECE40F819"
R2_APPROVED_EXCLUDED_CONTENT_IDENTITY_DIGEST = "F0F5F857F684E214BA252360B9454E46FBCE0D9A361910E03F3390F1FCEC42B7"
R2_APPROVED_SELECTION_SEMANTIC = (
    "ZERO_OVERLAP_UNIQUE_EXACT_LABEL_DYNAMIC_FOLD_"
    "CONTENT_DISJOINT_MINIMUM_OOF_GROUP_DISPLACEMENT_RANDOM_FILL_V2"
)
R2_EXACT_FIELDS = ("y_true", "historical_dynamic_bucket", "oof_fold")
R2_RELAXED_FIELDS = ("oof_group_id",)
R2_SHARED_ARMS = ("R2_U", "R2_F", "T_TO_R2_AT_160")
R2_APPROVED_UNIQUE_COUNT = 3000
R2_APPROVED_DISPLACEMENT_COUNT = 379
R2_APPROVED_EXCLUDED_COUNT = 3011
R2_APPROVED_EXCLUDED_CONTENT_DUPLICATE_COUNT = 11
R2_APPROVED_GROUP_TOTAL_VARIATION = R2_APPROVED_DISPLACEMENT_COUNT / R2_APPROVED_UNIQUE_COUNT


def validate_r2_matching_policy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "policy_id",
        "owner_decision",
        "approved_at",
        "supersedes_taskbook_requirements",
        "selection_seed",
        "unique_count",
        "overlap_with_t",
        "overlap_with_t_content",
        "content_unique_within_r2",
        "content_ledger_asset_id",
        "exact_fields",
        "relaxed_fields",
        "construction",
        "minimum_displacement_count",
        "applies_to_arms",
        "shared_identity_pool_required",
        "expected_identity_digest",
        "expected_content_map_digest",
        "expected_selected_content_digest",
        "expected_excluded_content_duplicate_count",
        "expected_group_total_variation",
        "residual_risk",
        "formal_training_authorized",
        "policy_digest",
    }
    if set(value) != required:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "R2 addendum policy fields are not exact",
            observed={"missing": sorted(required - set(value)), "extra": sorted(set(value) - required)},
        )
    expected = {
        "schema_version": R2_MATCHING_POLICY_SCHEMA,
        "policy_id": R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
        "owner_decision": "APPROVED_CONTENT_DISJOINT_REPAIR_2026-08-16",
        "approved_at": "2026-08-16",
        "supersedes_taskbook_requirements": ["SA-051", "SA-053", "SA-054"],
        "selection_seed": R2_APPROVED_SELECTION_SEED,
        "unique_count": R2_APPROVED_UNIQUE_COUNT,
        "overlap_with_t": 0,
        "overlap_with_t_content": 0,
        "content_unique_within_r2": True,
        "content_ledger_asset_id": "dataset_content_ledger",
        "exact_fields": list(R2_EXACT_FIELDS),
        "relaxed_fields": list(R2_RELAXED_FIELDS),
        "construction": "EXCLUDE_T_CONTENT_AND_DEDUP_CONTENT_THEN_EXHAUST_EXACT_CELLS_AND_COUNTER_HASH_FILL_WITHIN_SAME_EXACT_THREE_FIELD_CELL",
        "minimum_displacement_count": R2_APPROVED_DISPLACEMENT_COUNT,
        "applies_to_arms": list(R2_SHARED_ARMS),
        "shared_identity_pool_required": True,
        "expected_identity_digest": R2_APPROVED_IDENTITY_DIGEST,
        "expected_content_map_digest": R2_APPROVED_CONTENT_MAP_DIGEST,
        "expected_selected_content_digest": R2_APPROVED_SELECTED_CONTENT_DIGEST,
        "expected_excluded_content_duplicate_count": R2_APPROVED_EXCLUDED_CONTENT_DUPLICATE_COUNT,
        "expected_group_total_variation": R2_APPROVED_GROUP_TOTAL_VARIATION,
        "residual_risk": "OOF_GROUP_IS_FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID_R1_REMAINS_CO_PRIMARY_CONTROL",
        "formal_training_authorized": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "R2 addendum policy differs from the owner-approved decision",
                failing_field=field,
                observed=value.get(field),
                expected=expected_value,
            )
    core = {key: item for key, item in value.items() if key != "policy_digest"}
    if value.get("policy_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 addendum policy digest is invalid")
    return dict(value)


def validate_approved_r2_build(
    pool_spec: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a formal R2 pool is exactly the approved addendum build."""

    expected_spec = {
        "pool_role": "R2_MATCHED_RANDOM",
        "identity_digest": R2_APPROVED_IDENTITY_DIGEST,
        "unique_count_derived": R2_APPROVED_UNIQUE_COUNT,
        "construction_seed": R2_APPROVED_SELECTION_SEED,
        "selection_semantic": R2_APPROVED_SELECTION_SEMANTIC,
    }
    for field, expected_value in expected_spec.items():
        if pool_spec.get(field) != expected_value:
            raise SctsrError(
                ErrorCode.IDENTITY_DIGEST_MISMATCH,
                "Formal R2 pool specification differs from the approved addendum",
                failing_field=f"pool_spec.{field}",
                observed=pool_spec.get(field),
                expected=expected_value,
            )

    expected_audit = {
        "policy": R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
        "selected_count": R2_APPROVED_UNIQUE_COUNT,
        "excluded_count": R2_APPROVED_EXCLUDED_COUNT,
        "overlap_with_t_count": 0,
        "overlap_with_t_rate": 0.0,
        "selection_seed": R2_APPROVED_SELECTION_SEED,
        "selected_identity_digest": R2_APPROVED_IDENTITY_DIGEST,
        "exact_joint_stratum_quota": False,
        "exact_coarse_stratum_quota": True,
        "displacement_count": R2_APPROVED_DISPLACEMENT_COUNT,
        "displacement_lower_bound": R2_APPROVED_DISPLACEMENT_COUNT,
        "content_map_digest": R2_APPROVED_CONTENT_MAP_DIGEST,
        "excluded_content_duplicate_count": R2_APPROVED_EXCLUDED_CONTENT_DUPLICATE_COUNT,
        "excluded_content_identity_digest": R2_APPROVED_EXCLUDED_CONTENT_IDENTITY_DIGEST,
        "overlap_with_t_content_count": 0,
        "selected_content_digest": R2_APPROVED_SELECTED_CONTENT_DIGEST,
    }
    for field, expected_value in expected_audit.items():
        if audit.get(field) != expected_value:
            code = ErrorCode.CONFIGURATION_MISMATCH if field in {
                "policy",
                "exact_joint_stratum_quota",
                "exact_coarse_stratum_quota",
                "displacement_count",
                "displacement_lower_bound",
            } else ErrorCode.IDENTITY_DIGEST_MISMATCH
            raise SctsrError(
                code,
                "Formal R2 construction audit differs from the approved addendum",
                failing_field=f"audit.{field}",
                observed=audit.get(field),
                expected=expected_value,
            )
    if tuple(audit.get("relaxed_fields", ())) != R2_RELAXED_FIELDS:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Formal R2 construction relaxed a field other than the sole approved surrogate",
            failing_field="audit.relaxed_fields",
            observed=audit.get("relaxed_fields"),
            expected=list(R2_RELAXED_FIELDS),
        )
    guard_digest = str(audit.get("terminal_field_guard_digest", ""))
    if len(guard_digest) != 64:
        raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "Formal R2 audit lacks a terminal-field guard digest")

    rows = audit.get("displacement_records")
    if not isinstance(rows, (list, tuple)) or len(rows) != R2_APPROVED_DISPLACEMENT_COUNT:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Formal R2 displacement ledger row count differs from the approved minimum",
            observed=0 if not isinstance(rows, (list, tuple)) else len(rows),
            expected=R2_APPROVED_DISPLACEMENT_COUNT,
        )
    row_fields = {
        "selected_sample_id",
        "y_true",
        "historical_dynamic_bucket",
        "oof_fold",
        "requested_oof_group_id",
        "selected_oof_group_id",
        "selection_counter_hash",
    }
    selected_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R2 displacement row schema is invalid", failing_field=f"displacement_records[{index}]")
        sample_id = str(row["selected_sample_id"])
        if sample_id in selected_ids:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "R2 displacement ledger repeats a selected identity", observed=sample_id)
        selected_ids.add(sample_id)
        if row["requested_oof_group_id"] == row["selected_oof_group_id"]:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "R2 displacement row does not actually displace oof_group_id", failing_field=f"displacement_records[{index}]")
        counter_digest = str(row["selection_counter_hash"])
        if len(counter_digest) != 64 or any(character not in "0123456789abcdef" for character in counter_digest):
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 displacement row counter hash is invalid", failing_field=f"displacement_records[{index}]")
    if audit.get("displacement_ledger_digest") != stable_digest(rows):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 displacement ledger digest is invalid")

    required = audit.get("quota_required")
    selected_groups = pool_spec.get("oof_group_quota")
    if not isinstance(required, Mapping) or not isinstance(selected_groups, Mapping):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R2 build lacks group quota evidence")
    requested_groups: dict[str, int] = {}
    for stratum, count in required.items():
        parts = str(stratum).split("|", 3)
        if len(parts) != 4:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R2 strict quota key is malformed", observed=stratum)
        requested_groups[parts[3]] = requested_groups.get(parts[3], 0) + int(count)
    keys = set(requested_groups) | set(selected_groups)
    total_variation = sum(abs(requested_groups.get(key, 0) - int(selected_groups.get(key, 0))) for key in keys) / (2 * R2_APPROVED_UNIQUE_COUNT)
    if total_variation != R2_APPROVED_GROUP_TOTAL_VARIATION:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "R2 group total variation differs from the approved capacity lower bound",
            observed=total_variation,
            expected=R2_APPROVED_GROUP_TOTAL_VARIATION,
        )
    return {
        "status": "PASS",
        "policy_id": R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
        "identity_digest": R2_APPROVED_IDENTITY_DIGEST,
        "selection_seed": R2_APPROVED_SELECTION_SEED,
        "unique_count": R2_APPROVED_UNIQUE_COUNT,
        "displacement_count": R2_APPROVED_DISPLACEMENT_COUNT,
        "relaxed_fields": list(R2_RELAXED_FIELDS),
        "group_total_variation": total_variation,
    }
