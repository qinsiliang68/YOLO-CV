from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from extract_tdd_events import extract_tdd_events
from tdd_history_audit import REVIEW_IDENTITY, validate_tdd_history_audit


BASELINE = "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f"
SOURCE = "e9b6df61b0eb02e1d32c29175644f1c2af545afc"
ROLLOUT_ID = "019fea7a-400b-7322-b14d-252e736f7d39"
NON_BEHAVIOR = {18, 20, 30}


COMMITS = [
    "1e92350274b4b1ed1adcfcff2279efa20f0ef0fa",
    "4705464ed05a2add55cf7dad287bddef02a19cee",
    "2195c5c8cd131d017119dcebbb178e917c02773b",
    "0405764c719c08344998ef36b34cfe7dfb4417bd",
    "486c149b8c0266eb4b2a784fd39311425d7f6d0a",
    "039bcb6731befa3e75752e099ee9e4ecf5c35213",
    "a2fb3dd98ed0b923f6d0cab2892c299906ad9ebd",
    "0261d778b90a442fd7d48757b79be4efc55bfae8",
    "371fe031ac2a3c7b1276b75220df4fcb27ff2236",
    "157670f49a06952d4edfd1d36e6bf5c59e1ebb60",
    "700cebdc4129b79e7a8f10fc4c7c028c6c91354a",
    "73a60674f41a91a918975991833796ba9ad0e373",
    "f2792fe353f00f240dbe7384e181e6fdab26aa17",
    "3c2509c97e6a4279edd8f214766ce98cfd8e3108",
    "ac26e6d671cf3f8b39b64f09d1fbf17e19caaaf1",
    "d16e992435b54a5c56186b79856f8d48f614913b",
    "7be664c2830241104bbd11c89f27fe3a6a620c32",
    "b4d9e3c41250586143f697849650490195411891",
    "e353c5e1167f478332e7669a6b1054d004e4bd58",
    "1562b80def1c945d7d59a7b8f392a089cff8110e",
    "73d63b2048b33517cfe95c12797a45365079f14a",
    "a3360b293417233f23f1e8683422979419b82751",
    "18d43f8e10c677248623bd32ebdf4405964ea2dc",
    "ffe125bc47e5677519b55c4c93be0b5a6cf11a81",
    "13b1af3c7ec380e476ca0ae70cbd90d9f6a1f518",
    "a4ec9e9b32a69b69c942db1b8a41d2753ec75620",
    "bbbb0b484b354d64bb17de97cc9338ab21b1f6bc",
    "709ab5d053a3773217053806fce6407e86c5a0ff",
    "8675ebfbc25133607348f358da167d14f1a2f0eb",
    "80ae2a2dfff2140253026cdd36b9abd363c09a60",
    "bf0e4cf7ebed781422ae828cbfa4133d4d91857e",
    "f0c776cb2000f85f8779f4a009aa8fc29e1a0241",
    "cb01ff63460208da8e43aabfd9f6d23ac443e8af",
    "e9b6df61b0eb02e1d32c29175644f1c2af545afc",
]


COMMIT_EVENTS = {
    1: (11016, 11017), 2: (11379, 11380), 3: (11657, 11658), 4: (12104, 12105),
    5: (13196, 13197), 6: (13420, 13421), 7: (13511, 13512), 8: (14714, 14715),
    9: (14933, 14934), 10: (15132, 15133), 11: (15274, 15275), 12: (15344, 15345),
    13: (15621, 15622), 14: (15753, 15754), 15: (15969, 15970), 16: (16211, 16212),
    17: (16215, 16216), 18: (16225, 16226), 19: (16321, 16322), 20: (16386, 16387),
    21: (16403, 16404), 22: (16423, 16424), 23: (16534, 16535), 24: (16659, 16660),
    25: (16667, 16668), 26: (16838, 16839), 27: (18144, 18145), 28: (18147, 18148),
    29: (18150, 18151), 30: (17509, 17510), 31: (18606, 18607), 32: (18755, 18756),
    33: (18906, 18907),
    34: (19362, 19363),
}


CURRENT_EXACT = (19163, 19164)
HISTORICAL_COMMIT5_EXACT = (18317, 18318)
COMMAND_GUARD_EXACT = (18215, 18216)


def _byte_lines(lines: tuple[int, int]) -> tuple[int, int]:
    """Translate the exploratory Unicode splitline index to raw JSONL lines.

    The private rollout contains one Unicode line-separator character inside a
    JSON string before the registered events. ``str.splitlines()`` counted it;
    the fail-closed extractor correctly uses raw byte JSONL records. All
    curated exploratory coordinates therefore have one deterministic offset.
    """

    return lines[0] - 1, lines[1] - 1


def _pair(
    commit: int,
    test_id: str,
    red: tuple[int, int],
    patches: list[tuple[int, int]],
    historical_green: tuple[int, int],
    pattern: str,
    *,
    phase: str = "EXPECTED_BEHAVIOR_ASSERTION",
    suffix: str = "main",
    exact: tuple[int, int] = CURRENT_EXACT,
) -> dict[str, Any]:
    return {
        "commit_index": commit,
        "pair_id": f"C{commit:02d}_{suffix}",
        "test_id": test_id,
        "red": _byte_lines(red),
        "patches": [_byte_lines(item) for item in patches],
        "historical_green": _byte_lines(historical_green),
        "exact_green": _byte_lines(exact),
        "pattern": pattern,
        "phase": phase,
    }


PAIRS = [
    _pair(1, "tests/stage1_sctsr_v4/test_contract_hardening.py::test_sa_030_forged_nonempty_release_signature_is_rejected", (10899, 10900), [(10931, 10933)], (10936, 10937), "DID NOT RAISE"),
    _pair(2, "tests/stage1_sctsr_v4/test_asset_hardening.py::test_sa_041_asset_row_count_is_verified_from_csv", (11089, 11090), [(11098, 11100), (11115, 11117)], (11129, 11130), "DID NOT RAISE"),
    _pair(3, "tests/stage1_sctsr_v4/test_checkpoint_hardening.py::test_sa_084_windows_atomic_checkpoint_write_is_durable", (11415, 11416), [(11438, 11440), (11466, 11468), (11542, 11544)], (11547, 11548), "ValueError: can only convert an array of size 1"),
    _pair(4, "tests/stage1_sctsr_v4/test_fixed_step_hardening.py::test_sa_108_overlay_calls_upstream_optimizer_step_exactly_once", (11781, 11782), [(11785, 11787), (11795, 11797)], (11799, 11800), "assert 0 == 1"),
    _pair(5, "tests/red_commit_05/test_evidence_transaction_hardening.py::test_sa_144_selection_schema_requires_guard_quota_and_overlap", (12229, 12230), [(12240, 12242)], (13152, 13153), "FAILED tests/stage1_sctsr_v4/test_evidence_transaction_hardening.py::test_sa_144_selection_schema_requires_guard_quota_and_overlap", exact=HISTORICAL_COMMIT5_EXACT),
    _pair(6, "tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py::test_frontier_rejects_unregistered_artifact_bindings", (13265, 13266), [(13278, 13280)], (13312, 13313), "FAILED tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py::test_frontier_rejects_unregistered_artifact_bindings"),
    _pair(7, "tests/stage1_sctsr_v4/test_qrad_scaffold_hardening.py::test_nested_weighted_qrad_is_rejected_recursively", (13444, 13445), [(13449, 13451)], (13474, 13475), "DID NOT RAISE"),
    _pair(8, "tests/stage1_sctsr_v4/test_cli_contract_hardening.py::test_branch_cli_exposes_identity_pool_and_parent_artifact_index", (13956, 13957), [(13961, 13963)], (14157, 14158), "IDENTITY_DIGEST_MISMATCH"),
    _pair(9, "tests/stage1_sctsr_v4/test_formal_resume.py::test_formal_resume_restores_last_complete_checkpoint_history_and_quarantines_partial", (14879, 14880), [(14883, 14885)], (14888, 14889), "Recovery pointer is not the last contiguous complete generation"),
    _pair(10, "tests/stage1_sctsr_v4/test_schema_registry.py::test_schema_registry_rejects_stale_or_unregistered_public_schema", (15071, 15072), [(15092, 15094)], (15106, 15107), "FAILED tests/stage1_sctsr_v4/test_schema_registry.py::test_schema_registry_rejects_stale_or_unregistered_public_schema"),
    _pair(11, "tests/stage1_sctsr_v4/test_synthetic_canary.py::test_canary_exercises_real_checkpoint_resume", (15192, 15193), [(15197, 15199)], (15202, 15203), "FileNotFoundError"),
    _pair(12, "tests/stage1_sctsr_v4/test_source_identity.py::test_source_manifest_records_runtime_dependency_identity", (15324, 15325), [(15328, 15330)], (15333, 15334), "KeyError: 'runtime_environment'"),
    _pair(13, "tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py::test_formal_prediction_rejects_missing_extra_or_wrong_split_rows", (15463, 15464), [(15467, 15469)], (15472, 15473), "unexpected keyword argument 'repository_root'"),
    _pair(14, "tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py::test_formal_closeout_requires_recomputable_registered_e200_endpoint", (15723, 15724), [(15729, 15731), (15734, 15736)], (15739, 15740), "cannot import name 'validate_formal_endpoint_evidence'", phase="EXPECTED_PUBLIC_API_ABSENCE"),
    _pair(15, "tests/stage1_sctsr_v4/test_prediction_runtime.py::test_formal_endpoint_publisher_runs_real_images_and_writes_complete_evidence", (15847, 15848), [(15851, 15853)], (15855, 15856), "No module named 'stage1_sctsr_v4.prediction_runtime'", phase="EXPECTED_PUBLIC_API_ABSENCE"),
    _pair(16, "tests/stage1_sctsr_v4/test_source_identity.py::test_clean_source_manifest_revalidates_current_git_head_and_worktree", (16086, 16087), [(16094, 16096)], (16099, 16100), "DID NOT RAISE"),
    _pair(17, "tests/stage1_sctsr_v4/test_formal_input_snapshot.py::test_formal_input_snapshot_copies_and_binds_every_authorization_file", (16132, 16133), [(16141, 16143), (16146, 16148)], (16151, 16152), "cannot import name 'FORMAL_AUTHORIZATION_INPUT_ROLES'", phase="EXPECTED_PUBLIC_API_ABSENCE"),
    _pair(19, "tests/stage1_sctsr_v4/test_schema_registry.py::test_schema_registry_registers_formal_input_and_detailed_self_audit_schemas", (16261, 16262), [(16265, 16267)], (16283, 16284), "KeyError: 'external_file_binding'"),
    _pair(21, "COMMAND::git_diff_check_raw_receipt_bytes", (16390, 16391), [(16394, 16396)], (16399, 16400), "trailing whitespace.", phase="EXPECTED_COMMAND_GUARD", exact=COMMAND_GUARD_EXACT),
    _pair(22, "tests/stage1_sctsr_v4/test_implementation_self_audit.py::test_self_audit_round_trips_canonical_sorted_json", (16411, 16412), [(16415, 16417)], (16419, 16420), "FORMAL_RELEASE_NOT_AUTHORIZED"),
    _pair(23, "tests/stage1_sctsr_v4/test_repository_state_audit.py::test_repository_audit_binds_allowed_changes_and_legacy_sha_mtime", (16514, 16515), [(16526, 16528)], (16530, 16531), "Protected legacy SHA or mtime changed"),
    _pair(24, "tests/stage1_sctsr_v4/test_cli_contract_hardening.py::test_source_manifest_cli_uses_registered_source_roots", (16544, 16545), [(16547, 16549)], (16551, 16552), "No such file or directory"),
    _pair(25, "tests/stage1_sctsr_v4/test_manual_line_review.py::test_manual_line_review_requires_exact_sa280_to_sa289_and_line_digests", (16559, 16560), [(16563, 16565)], (16567, 16568), "No module named 'stage1_sctsr_v4.manual_line_review'", phase="EXPECTED_PUBLIC_API_ABSENCE", suffix="manual_review"),
    _pair(25, "tests/stage1_sctsr_v4/test_implementation_self_audit.py::test_self_audit_builder_merges_exact_taskbook_requirements_and_repository_flags", (16575, 16576), [(16579, 16581)], (16599, 16600), "cannot import name 'build_implementation_self_audit_from_plan'", phase="EXPECTED_PUBLIC_API_ABSENCE", suffix="self_audit_builder"),
    _pair(25, "tests/stage1_sctsr_v4/test_schema_registry.py::test_schema_registry_registers_formal_input_and_detailed_self_audit_schemas", (16626, 16627), [(16630, 16632)], (16635, 16636), "KeyError: 'self_audit_input_plan'", suffix="schema_registry"),
    _pair(26, "tests/stage1_sctsr_v4/test_repository_state_audit.py::test_repository_audit_uses_git_blob_identity_when_clean_worktree_line_endings_differ", (16819, 16820), [(16827, 16829)], (16831, 16832), "Protected legacy Git identity, normalized worktree content, or mtime changed"),
    _pair(27, "tests/stage1_sctsr_v4/test_repository_state_audit.py::test_repository_audit_does_not_follow_broken_historical_directory_links", (16865, 16866), [(16869, 16871)], (16874, 16875), "FileNotFoundError"),
    _pair(28, "tests/stage1_sctsr_v4/test_windows_long_paths.py::test_complete_synthetic_canary_normalizes_a_long_registered_root", (16948, 16949), [(16956, 16958)], (16960, 16961), "FileNotFoundError"),
    _pair(29, "tests/stage1_sctsr_v4/test_windows_long_paths.py::test_complete_synthetic_canary_normalizes_a_long_registered_root", (16982, 16983), [(16986, 16988)], (16993, 16994), "Run root is missing"),
    _pair(31, "tests/stage1_sctsr_v4/test_windows_long_paths.py::test_atomic_json_and_hash_survive_transaction_identity_path_beyond_max_path", (18559, 18560), [(18563, 18565), (18568, 18570)], (18573, 18574), "FileNotFoundError"),
    _pair(32, "tests/stage1_sctsr_v4/test_checkpoint_hardening.py::test_checkpoint_tensor_digest_uses_bulk_bytes_not_python_storage_iteration", (18740, 18741), [(18744, 18746)], (18748, 18749), "checkpoint hashing must not materialize bytes from Python storage iteration"),
    _pair(33, "tests/stage1_sctsr_v4/test_windows_long_paths.py::test_quarantine_accepts_unprefixed_registered_root_beyond_max_path", (18857, 18858), [(18861, 18863), (18890, 18892)], (18898, 18899), "FileNotFoundError"),
    _pair(
        34,
        "tests/stage1_sctsr_v4/test_repository_state_audit.py::test_repository_audit_records_changed_file_beyond_max_path",
        (19334, 19335),
        [(19338, 19340), (19350, 19352)],
        (19358, 19359),
        "Changed-file ledger path is missing",
        exact=(19426, 19427),
    ),
]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True, encoding="utf-8").strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(repository: Path, rollout: Path, evidence_root: Path) -> dict[str, Any]:
    observed = _git(repository, "rev-list", "--reverse", f"{BASELINE}..{SOURCE}").splitlines()
    if observed != COMMITS:
        raise ValueError("registered implementation commit inventory differs from git")

    # Merge every logical reference to one immutable rollout event. Exact-green
    # runs are intentionally shared, but their declared node IDs are the union
    # of every pair that cites the event.
    event_roles: dict[tuple[str, int, int], dict[str, Any]] = {}

    def register(kind: str, lines: tuple[int, int], *, test_id: str | None = None, phase: str = "NOT_APPLICABLE_GREEN") -> tuple[str, int, int]:
        key = (kind, lines[0], lines[1])
        row = event_roles.setdefault(key, {"kind": kind, "call_line": lines[0], "output_line": lines[1], "test_ids": set(), "failure_phase": phase})
        if test_id is not None:
            row["test_ids"].add(test_id)
        if phase != "NOT_APPLICABLE_GREEN":
            row["failure_phase"] = phase
        return key

    pair_keys: dict[str, dict[str, Any]] = {}
    for pair in PAIRS:
        red_key = register("TEST", pair["red"], test_id=pair["test_id"], phase=pair["phase"])
        historical_key = register("TEST", pair["historical_green"], test_id=pair["test_id"])
        exact_key = register("TEST", pair["exact_green"], test_id=pair["test_id"])
        patch_keys = [register("PATCH", item) for item in pair["patches"]]
        pair_keys[pair["pair_id"]] = {"red": red_key, "historical": historical_key, "exact": exact_key, "patches": patch_keys}
    commit_keys = {index: register("COMMIT", _byte_lines(lines)) for index, lines in COMMIT_EVENTS.items()}

    ordered_keys = sorted(event_roles, key=lambda item: (item[1], item[2], item[0]))
    event_ids = {key: f"EV{position:03d}" for position, key in enumerate(ordered_keys, start=1)}
    selections: list[dict[str, Any]] = []
    for key in ordered_keys:
        row = event_roles[key]
        selection: dict[str, Any] = {
            "event_id": event_ids[key],
            "kind": row["kind"],
            "call_line": row["call_line"],
            "output_line": row["output_line"],
        }
        if row["kind"] == "TEST":
            selection.update(
                {
                    "test_ids": sorted(row["test_ids"]),
                    "failure_phase": row["failure_phase"],
                    "output_path": f"tdd_history/event_outputs/{event_ids[key]}.log",
                }
            )
        elif row["kind"] == "COMMIT":
            commit_index = next(index for index, event_key in commit_keys.items() if event_key == key)
            selection["commit"] = COMMITS[commit_index - 1]
        selections.append(selection)
    selection_payload = {
        "schema_version": "stage1.sctsr.tdd_event_selection.v1",
        "rollout_id": ROLLOUT_ID,
        "selections": selections,
    }
    selection_path = evidence_root / "tdd_history/TDD_EVENT_SELECTION.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle_path = evidence_root / "tdd_history/TDD_EVENT_BUNDLE.json"
    provenance = extract_tdd_events(
        rollout_path=rollout,
        selection=selection_payload,
        evidence_root=evidence_root,
        bundle_path=bundle_path,
    )

    pairs_by_commit: dict[int, list[dict[str, Any]]] = {}
    for pair in PAIRS:
        keys = pair_keys[pair["pair_id"]]
        pairs_by_commit.setdefault(pair["commit_index"], []).append(
            {
                "pair_id": pair["pair_id"],
                "test_id": pair["test_id"],
                "red_event_id": event_ids[keys["red"]],
                "patch_event_ids": [event_ids[item] for item in keys["patches"]],
                "historical_green_event_id": event_ids[keys["historical"]],
                "exact_green_event_id": event_ids[keys["exact"]],
                "expected_failure_pattern": pair["pattern"],
            }
        )

    units: list[dict[str, Any]] = []
    for index, commit in enumerate(COMMITS, start=1):
        subject = _git(repository, "show", "-s", "--format=%s", commit)
        changed_paths = _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        pairs = pairs_by_commit.get(index, [])
        if index in NON_BEHAVIOR:
            classification = "NON_BEHAVIOR_CHANGE"
            reason = "Dependency, documentation, or evidence-only rollback unit with no claimed runtime behavior."
            claims: list[dict[str, Any]] = []
        else:
            classification = "BEHAVIOR_CHANGE"
            reason = "Runtime, contract, validation, CLI, or failure-closed behavior changed under a failing-first check."
            claims = [
                {
                    "claim_id": f"CLAIM_C{index:02d}",
                    "description": f"Rollback unit {index:02d} behavior is covered by its registered failing-first pair(s).",
                    "changed_paths": changed_paths,
                    "pair_ids": [pair["pair_id"] for pair in pairs],
                }
            ]
        units.append(
            {
                "commit": commit,
                "subject": subject,
                "classification": classification,
                "classification_reason": reason,
                "changed_paths": changed_paths,
                "behavior_claims": claims,
                "commit_event_id": event_ids[commit_keys[index]],
                "pairs": pairs,
            }
        )
    audit = {
        "schema_version": "stage1.sctsr.tdd_history_audit.v1",
        "baseline_commit": BASELINE,
        "implementation_source_commit": SOURCE,
        "rollout_provenance": {
            "rollout_id": ROLLOUT_ID,
            "reviewer_independence_claim": REVIEW_IDENTITY,
            "source_path_claim": str(provenance["source_path_claim"]),
            "prefix_line_count": int(provenance["prefix_line_count"]),
            "prefix_sha256": str(provenance["prefix_sha256"]),
            "event_bundle_path": str(provenance["event_bundle_path"]),
            "event_bundle_bytes": int(provenance["event_bundle_bytes"]),
            "event_bundle_sha256": str(provenance["event_bundle_sha256"]),
        },
        "commit_units": units,
    }
    audit_path = evidence_root / "tdd_history/TDD_HISTORY_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = validate_tdd_history_audit(audit, evidence_root=evidence_root, expected_commit_order=COMMITS)
    receipt = {
        **result,
        "audit_path": audit_path.relative_to(evidence_root).as_posix(),
        "audit_bytes": audit_path.stat().st_size,
        "audit_sha256": _sha(audit_path),
        "selection_path": selection_path.relative_to(evidence_root).as_posix(),
        "selection_bytes": selection_path.stat().st_size,
        "selection_sha256": _sha(selection_path),
        "event_bundle_path": bundle_path.relative_to(evidence_root).as_posix(),
        "event_bundle_bytes": bundle_path.stat().st_size,
        "event_bundle_sha256": _sha(bundle_path),
    }
    receipt_path = evidence_root / "tdd_history/TDD_HISTORY_AUDIT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.repository_root.resolve(), args.rollout.resolve(), args.evidence_root.resolve())
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
