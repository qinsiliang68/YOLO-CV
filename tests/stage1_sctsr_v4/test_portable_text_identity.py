from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


FORMAL_CSV_PATHS = (
    "data/final_sewerml_dataset/manifests/train_manifest.csv",
    "data/final_sewerml_dataset/manifests/normal_train_manifest.csv",
    "data/final_sewerml_dataset/manifests/val_model_manifest.csv",
    "data/final_sewerml_dataset/manifests/normal_val_model_manifest.csv",
    "data/final_sewerml_dataset/manifests/val_cal_manifest.csv",
    "data/final_sewerml_dataset/manifests/normal_val_cal_manifest.csv",
    "data/final_sewerml_dataset/manifests/val_op_manifest.csv",
    "data/final_sewerml_dataset/manifests/normal_val_op_manifest.csv",
    "artifacts/stage1_oof_folds_10fold_20260617/train_oof_assignments.csv",
)
V4_ASSET_CSV_PATHS = (
    "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/assets/T_STRESS_CONTENT_UNIQUE_v1.csv",
    "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/assets/DATASET_CONTENT_EXCLUSIONS_v1.csv",
)
R2_REPORT_ROOT = "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_r2_addendum_20260815"


def _git_blob(repository_root, path: str) -> bytes:
    return subprocess.check_output(("git", "show", f":{path}"), cwd=repository_root)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _attributes(repository_root, paths: list[str]) -> dict[str, dict[str, str]]:
    output = subprocess.check_output(
        ("git", "check-attr", "text", "eol", "--", *paths),
        cwd=repository_root,
        text=True,
    )
    result: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        path, attribute, value = line.rsplit(": ", 2)
        result.setdefault(path, {})[attribute] = value
    return result


def test_formal_csv_and_r2_evidence_bytes_are_lf_portable_git_blobs(repository_root):
    evidence_path = repository_root / R2_REPORT_ROOT / "EVIDENCE_MANIFEST.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    log_rows = [row for row in evidence["files"] if row["path"].endswith(".log")]
    log_paths = [f"{R2_REPORT_ROOT}/{row['path']}" for row in log_rows]
    all_paths = list(FORMAL_CSV_PATHS + V4_ASSET_CSV_PATHS) + log_paths

    attributes = _attributes(repository_root, all_paths)
    assert all(attributes[path] == {"text": "set", "eol": "lf"} for path in all_paths)

    asset_registry = json.loads(
        (repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json").read_text(encoding="utf-8")
    )
    registered = {row["relative_path"]: row for row in asset_registry["assets"]}
    for path in FORMAL_CSV_PATHS + V4_ASSET_CSV_PATHS:
        blob = _git_blob(repository_root, path)
        assert b"\r\n" not in blob
        assert registered[path]["bytes"] == len(blob)
        assert registered[path]["sha256"] == _sha(blob)
        assert (repository_root / path).read_bytes() == blob

    for row, path in zip(log_rows, log_paths, strict=True):
        blob = _git_blob(repository_root, path)
        assert b"\r\n" not in blob
        assert row["bytes"] == len(blob)
        assert row["sha256"] == _sha(blob)
        assert (repository_root / path).read_bytes() == blob


def test_every_registered_evidence_text_has_portable_checkout_identity(repository_root):
    manifest_paths = subprocess.check_output(
        ("git", "ls-files", "*EVIDENCE_MANIFEST.json"),
        cwd=repository_root,
        text=True,
    ).splitlines()
    text_suffixes = {".log", ".txt", ".csv", ".json", ".md", ".yaml", ".yml"}
    immutable_legacy_root = (
        "artifacts/stage1_sample_value_experiments/experiments/"
        "dynamic_replay_budget_efficiency_20260807/08_reports/"
        "sctsr_v4_implementation_audit_20260813/"
    )
    checked = 0
    legacy_checked = 0
    legacy_unmaterialized = 0
    for manifest_relative in manifest_paths:
        manifest_path = repository_root / manifest_relative
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("files", []):
            relative = row.get("relative_path", row.get("path"))
            if not isinstance(relative, str) or Path(relative).suffix.lower() not in text_suffixes:
                continue
            candidates = [manifest_path.parent / relative, repository_root / relative]
            existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
            if not existing:
                legacy_unmaterialized += 1
                continue
            path = existing[0]
            repository_relative = path.relative_to(repository_root).as_posix()
            blob = _git_blob(repository_root, repository_relative)
            worktree = path.read_bytes()
            assert int(row["bytes"]) == len(blob)
            assert str(row["sha256"]).upper() == _sha(blob)
            attributes = _attributes(repository_root, [repository_relative])[repository_relative]
            if repository_relative.startswith(immutable_legacy_root):
                assert attributes == {"text": "unset", "eol": "unset"}
                assert worktree == blob
                legacy_checked += 1
            else:
                assert attributes == {"text": "set", "eol": "lf"}
                assert b"\r\n" not in blob
                assert b"\r\n" not in worktree
                assert worktree == blob
                checked += 1
    assert checked > 0
    assert legacy_checked > 0
    assert legacy_unmaterialized > 0
