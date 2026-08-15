from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


def _exact_key(row: Mapping[str, Any], oof_group_id: str) -> tuple[int, str, int, str]:
    return (
        int(row["y_true"]),
        str(row["dynamic_bucket"]),
        int(row["oof_fold"]),
        str(oof_group_id),
    )


def repair_t_content_duplicates(
    t_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    oof_groups: Mapping[str, str],
    content_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace duplicate-byte T occurrences without changing four-field quotas.

    The historical T file remains immutable. For every duplicate image SHA,
    the lowest numeric rank is retained. Each later occurrence is replaced by
    the highest historical GapCritical candidate from the same label, dynamic
    bucket, OOF fold and OOF-group surrogate. Sample ID is the deterministic
    tie-breaker. Content SHA is used only as a reliability/deduplication gate.
    """

    rows = [dict(row) for row in t_rows]
    if not rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Historical T rows must have unique sample IDs")
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["sample_id"])
        content = content_by_id.get(sample_id)
        group = oof_groups.get(sample_id)
        if content is None or group is None:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "T repair input is absent from the content or OOF ledger",
                observed=sample_id,
            )
        by_sha[str(content["image_sha256"]).upper()].append(row)
    remove: list[dict[str, Any]] = []
    for members in by_sha.values():
        ordered = sorted(members, key=lambda row: (int(row["rank"]), str(row["sample_id"])))
        remove.extend(ordered[1:])
    if not remove:
        audit = {
            "replacement_count": 0,
            "content_unique_before": len(by_sha),
            "content_unique_after": len(by_sha),
            "exact_four_field_quota_preserved": True,
            "replacements": [],
        }
        return rows, {**audit, "audit_digest": stable_digest(audit)}

    original_ids = {str(row["sample_id"]) for row in rows}
    occupied_shas = set(by_sha)
    candidates = [dict(row) for row in candidate_rows]
    all_groups = dict(oof_groups)
    all_groups.update({str(row["sample_id"]): str(row["oof_group_id"]) for row in candidates})
    replacements: list[dict[str, Any]] = []
    before_quota = Counter(
        _exact_key(row, oof_groups[str(row["sample_id"])])
        for row in rows
    )
    by_rank = {int(row["rank"]): row for row in rows}
    chosen_ids: set[str] = set()
    for removed in sorted(remove, key=lambda row: (int(row["rank"]), str(row["sample_id"]))):
        removed_id = str(removed["sample_id"])
        target_key = _exact_key(removed, oof_groups[removed_id])
        eligible: list[dict[str, Any]] = []
        for candidate in candidates:
            sample_id = str(candidate["sample_id"])
            content = content_by_id.get(sample_id)
            if (
                sample_id in original_ids
                or sample_id in chosen_ids
                or content is None
                or str(content["image_sha256"]).upper() in occupied_shas
            ):
                continue
            candidate_group = str(candidate.get("oof_group_id", ""))
            if _exact_key(candidate, candidate_group) == target_key:
                eligible.append(candidate)
        if not eligible:
            raise SctsrError(
                ErrorCode.R2_QUOTA_INFEASIBLE,
                "No content-unique T replacement preserves the exact four-field quota",
                observed={"removed_sample_id": removed_id, "stratum": target_key},
            )
        selected = sorted(
            eligible,
            key=lambda row: (-float(row["gap_critical_score"]), str(row["sample_id"])),
        )[0]
        selected_id = str(selected["sample_id"])
        replacement = {
            **removed,
            "sample_id": selected_id,
            "y_true": int(selected["y_true"]),
            "oof_fold": int(selected["oof_fold"]),
            "dynamic_bucket": str(selected["dynamic_bucket"]),
            "mean_p_defect": selected["mean_p_defect"],
            "correct_rate": selected["correct_rate"],
            "std_p_defect": selected["std_p_defect"],
            "replay_role": "normal_replay" if int(selected["y_true"]) == 0 else "defect_guard_replay",
        }
        by_rank[int(removed["rank"])] = replacement
        chosen_ids.add(selected_id)
        selected_sha = str(content_by_id[selected_id]["image_sha256"]).upper()
        occupied_shas.add(selected_sha)
        replacements.append(
            {
                "rank": int(removed["rank"]),
                "removed_sample_id": removed_id,
                "removed_image_sha256": str(content_by_id[removed_id]["image_sha256"]).upper(),
                "replacement_sample_id": selected_id,
                "replacement_image_sha256": selected_sha,
                "exact_stratum": list(target_key),
                "selection_signal": "HISTORICAL_GAP_CRITICAL_SCORE_WITHIN_EXACT_STRATUM",
                "selection_value": float(selected["gap_critical_score"]),
            }
        )
    repaired = [by_rank[rank] for rank in sorted(by_rank)]
    repaired_ids = [str(row["sample_id"]) for row in repaired]
    repaired_shas = [str(content_by_id[sample_id]["image_sha256"]).upper() for sample_id in repaired_ids]
    after_quota = Counter(
        _exact_key(row, all_groups[str(row["sample_id"])])
        for row in repaired
    )
    exact_preserved = before_quota == after_quota
    if len(set(repaired_ids)) != len(rows) or len(set(repaired_shas)) != len(rows) or not exact_preserved:
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_MISMATCH,
            "Derived T repair did not preserve identity/content uniqueness and exact quota",
        )
    audit = {
        "replacement_count": len(replacements),
        "content_unique_before": len(by_sha),
        "content_unique_after": len(set(repaired_shas)),
        "exact_four_field_quota_preserved": exact_preserved,
        "replacements": replacements,
    }
    return repaired, {**audit, "audit_digest": stable_digest(audit)}
