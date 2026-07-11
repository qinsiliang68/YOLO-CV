from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import load_contract
from .errors import ContractError, ValidationError
from .util import read_yaml, sha256_file, stable_hash


@dataclass(frozen=True)
class RuntimeContract:
    path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def runtime_contract_id(self) -> str:
        return str(self.data["runtime_contract_id"])


def compute_runtime_contract_hash(data: dict[str, Any]) -> str:
    body = copy.deepcopy(data)
    body.pop("runtime_contract_sha256", None)
    return stable_hash(body)


def load_runtime_contract(path: str | Path, *, verify_hash: bool = True) -> RuntimeContract:
    resolved = Path(path).resolve()
    data = read_yaml(resolved)
    required = {
        "runtime_contract_id",
        "runtime_contract_version",
        "release",
        "science_contract",
        "queue",
        "checkpoint",
        "machine_assets",
        "execution_identity",
    }
    missing = required - set(data)
    if missing:
        raise ContractError(f"Runtime contract missing keys: {sorted(missing)}")
    actual = compute_runtime_contract_hash(data)
    expected = str(data.get("runtime_contract_sha256", "")).upper()
    if verify_hash and actual != expected:
        raise ContractError(f"Runtime contract hash mismatch: expected={expected}, actual={actual}")
    if str(data["runtime_contract_version"]) != "1.2.0":
        raise ContractError(
            f"Unsupported runtime contract version: {data['runtime_contract_version']}"
        )
    return RuntimeContract(path=resolved, data=data, sha256=actual)


def _safe_repo_path(repo_root: Path, relative: str, *, label: str) -> Path:
    normalized = str(relative).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError(f"Unsafe {label} path: {relative}")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValidationError(f"{label} escapes repository root: {relative}") from exc
    return resolved


def _csv_rows(path: Path) -> int:
    try:
        return len(pd.read_csv(path, usecols=[0]))
    except Exception as exc:
        raise ValidationError(f"Unable to read CSV rows from {path}: {exc}") from exc


def _verify_link(repo_root: Path, label: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(repo_root, str(spec["path"]), label=label)
    if not path.is_file():
        raise ValidationError(f"Missing runtime-linked file {label}: {path}")
    actual = sha256_file(path)
    expected = str(spec["sha256"]).upper()
    if actual != expected:
        raise ValidationError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}")
    rows = _csv_rows(path)
    expected_rows = int(spec["rows"])
    if rows != expected_rows:
        raise ValidationError(f"{label} row count mismatch: expected={expected_rows}, actual={rows}")
    return {
        "path": str(path),
        "sha256": actual,
        "rows": rows,
    }


def validate_runtime_links(contract: RuntimeContract, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    science_spec = contract.data["science_contract"]
    science_path = _safe_repo_path(repo, str(science_spec["path"]), label="science_contract")
    if not science_path.is_file():
        raise ValidationError(f"Missing science contract: {science_path}")
    science_file_sha = sha256_file(science_path)
    expected_file_sha = str(science_spec["file_sha256"]).upper()
    if science_file_sha != expected_file_sha:
        raise ValidationError(
            f"science_contract SHA-256 mismatch: expected={expected_file_sha}, actual={science_file_sha}"
        )
    science = load_contract(science_path)
    expected_semantic = str(science_spec["semantic_sha256"]).upper()
    if science.sha256 != expected_semantic:
        raise ValidationError(
            "science_contract semantic SHA-256 mismatch: "
            f"expected={expected_semantic}, actual={science.sha256}"
        )

    queue = contract.data["queue"]
    matrix = _verify_link(repo, "frozen_matrix", queue["frozen_matrix"])
    index = _verify_link(repo, "selection_index", queue["selection_index"])
    index_frame = pd.read_csv(index["path"], dtype={"run_slot": "string", "sha256": "string"})
    required_index = {"run_slot", "selection_manifest", "sha256"}
    missing = required_index - set(index_frame.columns)
    if missing:
        raise ValidationError(f"selection_index missing columns: {sorted(missing)}")
    if index_frame["run_slot"].isna().any() or index_frame["run_slot"].duplicated().any():
        raise ValidationError("selection_index run_slot values must be non-null and unique")
    selection_count = int(queue["selection_count"])
    if len(index_frame) != selection_count:
        raise ValidationError(
            f"selection_index count mismatch: expected={selection_count}, actual={len(index_frame)}"
        )

    checkpoint_spec = contract.data["checkpoint"]
    binding_spec = checkpoint_spec["site_binding"]
    binding_path = _safe_repo_path(
        repo, str(binding_spec["path"]), label="checkpoint site_binding"
    )
    if not binding_path.is_file():
        raise ValidationError(f"Missing checkpoint site binding: {binding_path}")
    binding_file_sha = sha256_file(binding_path)
    expected_binding_sha = str(binding_spec["sha256"]).upper()
    if binding_file_sha != expected_binding_sha:
        raise ValidationError(
            "checkpoint site_binding SHA-256 mismatch: "
            f"expected={expected_binding_sha}, actual={binding_file_sha}"
        )
    import json

    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Invalid checkpoint site binding {binding_path}: {exc}") from exc
    expected_checkpoint_sha = str(checkpoint_spec["sha256"]).upper()
    if str(binding.get("full_sha256", "")).upper() != expected_checkpoint_sha:
        raise ValidationError("Checkpoint SHA-256 differs between runtime contract and site binding")
    if str(binding.get("checkpoint_filename", "")) != str(checkpoint_spec["filename"]):
        raise ValidationError("Checkpoint filename differs between runtime contract and site binding")
    return {
        "status": "PASS",
        "runtime_contract_id": contract.runtime_contract_id,
        "runtime_contract_sha256": contract.sha256,
        "science_contract": {
            "path": str(science_path),
            "file_sha256": science_file_sha,
            "semantic_sha256": science.sha256,
        },
        "queue": {"frozen_matrix": matrix, "selection_index": index},
        "checkpoint": {
            "site_binding_path": str(binding_path),
            "site_binding_sha256": binding_file_sha,
            "filename": str(checkpoint_spec["filename"]),
            "sha256": expected_checkpoint_sha,
        },
        "selection_count": selection_count,
    }


def verify_all_selections_against_index(
    contract: RuntimeContract,
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    linked = validate_runtime_links(contract, repo)
    index = pd.read_csv(linked["queue"]["selection_index"]["path"], dtype="string")
    results = [
        verify_selection_against_index(contract, repo, str(run_slot))
        for run_slot in index["run_slot"]
    ]
    return {
        "status": "PASS",
        "selection_count": len(results),
        "selection_index_sha256": linked["queue"]["selection_index"]["sha256"],
        "selection_hash_digest": stable_hash(
            [(row["run_slot"], row["selection_sha256"]) for row in results]
        ),
    }


def _artifact_root(contract: RuntimeContract, repo_root: Path) -> Path:
    return _safe_repo_path(
        repo_root,
        str(contract.data["queue"]["artifact_root"]),
        label="queue artifact_root",
    )


def verify_selection_against_index(
    contract: RuntimeContract,
    repo_root: str | Path,
    run_slot: str,
    selection_path: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    # Verify the index itself before trusting any path or checksum stored inside it.
    index_result = _verify_link(repo, "selection_index", contract.data["queue"]["selection_index"])
    frame = pd.read_csv(index_result["path"], dtype="string")
    rows = frame.loc[frame["run_slot"] == str(run_slot)]
    if len(rows) != 1:
        raise ValidationError(f"Expected exactly one selection_index row for {run_slot}, got {len(rows)}")
    row = rows.iloc[0]
    artifact_root = _artifact_root(contract, repo)
    indexed_relative = str(row["selection_manifest"]).replace("\\", "/")
    indexed = _safe_repo_path(artifact_root, indexed_relative, label="selection manifest")
    if selection_path is not None and Path(selection_path).resolve() != indexed:
        raise ValidationError(
            f"Selection path does not match frozen index for {run_slot}: {selection_path} != {indexed}"
        )
    if not indexed.is_file():
        raise ValidationError(f"Missing frozen selection for {run_slot}: {indexed}")
    actual = sha256_file(indexed)
    expected = str(row["sha256"]).upper()
    if actual != expected:
        raise ValidationError(
            f"Selection SHA-256 mismatch for {run_slot}: expected={expected}, actual={actual}"
        )
    return {
        "status": "PASS",
        "run_slot": str(run_slot),
        "selection_path": str(indexed),
        "selection_sha256": actual,
        "selection_index_sha256": index_result["sha256"],
    }


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValidationError(f"Git identity check failed: {exc.output.strip()}") from exc


def verify_release_identity(
    contract: RuntimeContract,
    repo_root: str | Path,
    *,
    test_release_ref_override: str | None = None,
    allow_test_override: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if test_release_ref_override is not None and not allow_test_override:
        raise ValidationError("Explicit test override is disabled")
    configured_tag = str(contract.data["release"]["git_tag"])
    expected_ref = test_release_ref_override or f"refs/tags/{configured_tag}^{{commit}}"
    head = _git_output(repo, "rev-parse", "HEAD")
    expected_commit = _git_output(repo, "rev-parse", expected_ref)
    if bool(contract.data["release"].get("require_tag_at_head", True)) and expected_commit != head:
        raise ValidationError(
            f"Release ref {expected_ref} does not point to HEAD: expected={expected_commit}, HEAD={head}"
        )
    return {
        "status": "PASS",
        "git_tag": configured_tag,
        "expected_ref": expected_ref,
        "expected_commit": expected_commit,
        "head": head,
        "override_used": test_release_ref_override is not None,
    }


def validation_status_for_mode(dry_run: bool, contract: RuntimeContract | None = None) -> str:
    if contract is None:
        return "DRY_RUN_VALIDATED" if dry_run else "VALIDATED"
    identity = contract.data["execution_identity"]
    return str(identity["dry_run_status"] if dry_run else identity["formal_status"])


def is_aggregatable_status(status: str, contract: RuntimeContract | None = None) -> bool:
    allowed = (
        {"VALIDATED"}
        if contract is None
        else {str(value) for value in contract.data["execution_identity"]["aggregatable_statuses"]}
    )
    return str(status) in allowed
