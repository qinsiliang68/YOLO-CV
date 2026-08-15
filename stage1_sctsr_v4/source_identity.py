from __future__ import annotations

import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import load_json, sha256_file, stable_digest


SOURCE_TREE_SCHEMA = "stage1.sctsr.source_tree_manifest.v1"
EXCLUSION_POLICY = {
    "directory_names": [".pytest_cache", "__pycache__"],
    "file_suffixes": [".pyc", ".pyo"],
}


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _unavailable(reason_code: str) -> dict[str, str]:
    return {"status": "UNAVAILABLE", "reason_code": reason_code}


def probe_runtime_environment(repository_root: str | Path) -> dict[str, Any]:
    """Probe the current process, accelerator stack, and registered adapters.

    Absolute interpreter deployment parents are intentionally excluded so two
    ``uv run --isolated`` environments with identical runtime bytes compare
    equal. Repository source origins are recorded relative to the supplied
    checkout and are paired with byte count and SHA-256.
    """

    base = Path(repository_root).resolve()
    environment: dict[str, Any] = {
        "python": {
            "status": "AVAILABLE",
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            # ``uv run --isolated`` materializes the same interpreter under a
            # fresh random temporary directory on every invocation.  Binding
            # that parent directory made identical source/runtime bytes yield
            # different source-tree and checkpoint identities.  The executable
            # name, implementation and exact version identify the interpreter
            # contract without incorporating an ephemeral deployment path.
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "status": "AVAILABLE",
            "description": platform.platform(),
            "machine": platform.machine(),
        },
    }

    try:
        import torch

        cuda_build = str(torch.version.cuda) if torch.version.cuda is not None else None
        torch_record: dict[str, Any] = {
            "status": "AVAILABLE",
            "version": str(torch.__version__),
            "cuda_build": (
                {"status": "AVAILABLE", "version": cuda_build}
                if cuda_build is not None
                else _unavailable("TORCH_CPU_ONLY_BUILD")
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_devices": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                torch_record["cuda_devices"].append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "total_memory_bytes": int(properties.total_memory),
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )
        environment["torch"] = torch_record
    except (ImportError, OSError, RuntimeError) as exc:
        environment["torch"] = {
            **_unavailable("TORCH_IMPORT_OR_CUDA_PROBE_FAILED"),
            "exception_type": type(exc).__name__,
        }

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
        environment["nvidia_driver"] = (
            {"status": "AVAILABLE", "versions": versions}
            if versions
            else _unavailable("NVIDIA_SMI_RETURNED_NO_DRIVER_VERSION")
        )
    except (OSError, subprocess.SubprocessError):
        environment["nvidia_driver"] = _unavailable("NVIDIA_SMI_UNAVAILABLE")

    local_init = base / "YOLOv11" / "ultralytics" / "__init__.py"
    local_version: str | None = None
    if local_init.is_file():
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', local_init.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            local_version = match.group(1)
    if local_version is not None:
        environment["ultralytics"] = {
            "status": "AVAILABLE",
            "version": local_version,
            "source": "FROZEN_REPOSITORY_SOURCE",
            "init_path": local_init.relative_to(base).as_posix(),
            "init_bytes": local_init.stat().st_size,
            "init_sha256": sha256_file(local_init),
        }
    else:
        try:
            installed_version = metadata.version("ultralytics")
        except metadata.PackageNotFoundError:
            environment["ultralytics"] = _unavailable("ULTRALYTICS_VERSION_NOT_DISCOVERABLE")
        else:
            environment["ultralytics"] = {
                "status": "AVAILABLE",
                "version": installed_version,
                "source": "INSTALLED_DISTRIBUTION_METADATA_ONLY",
            }
    adapter = base / "integrations" / "ultralytics" / "sctsr_classification_trainer.py"
    environment["sctsr_adapter"] = (
        {
            "status": "AVAILABLE",
            "relative_origin": adapter.relative_to(base).as_posix(),
            "bytes": adapter.stat().st_size,
            "sha256": sha256_file(adapter),
        }
        if adapter.is_file() and not adapter.is_symlink()
        else _unavailable("REGISTERED_SCTSR_ADAPTER_MISSING")
    )
    dependencies: dict[str, Mapping[str, str]] = {}
    for distribution in ("numpy", "pillow", "pyarrow", "torchvision"):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependencies[distribution] = _unavailable("DISTRIBUTION_NOT_INSTALLED")
        else:
            dependencies[distribution] = {"status": "AVAILABLE", "version": version}
    environment["registered_dependencies"] = dependencies
    return environment


def _normalize_include_path(value: object) -> str:
    token = str(value).replace("\\", "/")
    path = Path(token)
    if token in {"", "."} or path.is_absolute() or path.drive or ".." in path.parts:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree include path is empty, absolute, or escapes the repository",
            observed=token,
        )
    return path.as_posix()


def _is_excluded(relative_path: Path) -> bool:
    return bool(
        set(relative_path.parts).intersection(EXCLUSION_POLICY["directory_names"])
        or relative_path.suffix.lower() in EXCLUSION_POLICY["file_suffixes"]
    )


def _scan_included_files(base: Path, include_paths: Sequence[object]) -> tuple[list[str], list[dict[str, Any]]]:
    if not include_paths:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree include paths may not be empty")

    normalized_roots: list[str] = []
    seen_roots: set[str] = set()
    files_by_path: dict[str, dict[str, Any]] = {}
    owner_by_file: dict[str, str] = {}
    for value in include_paths:
        relative_root = _normalize_include_path(value)
        if relative_root in seen_roots:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Source-tree include path is duplicated",
                observed=relative_root,
            )
        seen_roots.add(relative_root)
        normalized_roots.append(relative_root)

        source = (base / Path(relative_root)).resolve()
        try:
            source.relative_to(base)
        except ValueError as exc:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Source-tree include path resolves outside the repository",
                observed=relative_root,
            ) from exc
        if not source.exists():
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Source-tree include path does not exist",
                artifact_path=str(source),
            )

        candidates = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(base)
            except ValueError as exc:
                raise SctsrError(
                    ErrorCode.SOURCE_TREE_MISMATCH,
                    "Source-tree candidate resolves outside the repository",
                    artifact_path=str(candidate),
                ) from exc
            if _is_excluded(relative):
                continue
            token = relative.as_posix()
            if token in files_by_path:
                raise SctsrError(
                    ErrorCode.SOURCE_TREE_MISMATCH,
                    "Source-tree include paths overlap and register a file more than once",
                    observed={"relative_path": token, "include_paths": [owner_by_file[token], relative_root]},
                )
            owner_by_file[token] = relative_root
            files_by_path[token] = {
                "relative_path": token,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }

    if not files_by_path:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree include paths contain no registered files")
    return sorted(normalized_roots), [files_by_path[key] for key in sorted(files_by_path)]


def _content_digest(
    *,
    include_paths: Sequence[str],
    files: Sequence[Mapping[str, Any]],
    external_references: Sequence[Mapping[str, Any]],
    runtime_environment: Mapping[str, Any],
) -> str:
    return stable_digest(
        {
            "include_paths": list(include_paths),
            "exclusion_policy": EXCLUSION_POLICY,
            "files": list(files),
            "external_references": list(external_references),
            "runtime_environment": dict(runtime_environment),
        }
    )


def build_source_tree_manifest(
    root: str | Path,
    include_paths: Sequence[str],
    *,
    external_references: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve()
    roots, files = _scan_included_files(base, include_paths)
    external = list(external_references or [])
    runtime_environment = probe_runtime_environment(base)
    git_status = _git(base, "status", "--porcelain")
    return {
        "schema_version": SOURCE_TREE_SCHEMA,
        "root": base.as_posix(),
        "include_paths": roots,
        "exclusion_policy": EXCLUSION_POLICY,
        "git_head": _git(base, "rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "files": files,
        "external_references": external,
        "runtime_environment": runtime_environment,
        "runtime_environment_digest": stable_digest(runtime_environment),
        "source_tree_digest": _content_digest(
            include_paths=roots,
            files=files,
            external_references=external,
            runtime_environment=runtime_environment,
        ),
    }


def validate_source_tree_manifest(
    manifest: Mapping[str, Any] | str | Path,
    repository_root: str | Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Verify registered bytes and exact file coverage under every include root.

    The stored absolute ``root`` is informational only. All paths are resolved
    under the caller-provided checkout. Rescanning the registered roots is
    essential: validating only rows already present in the manifest would let
    an untracked importable source file silently enter a formal run.
    """

    raw: Any = load_json(manifest) if not isinstance(manifest, Mapping) else manifest
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SOURCE_TREE_SCHEMA:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree manifest schema is invalid")
    if raw.get("exclusion_policy") != EXCLUSION_POLICY:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree exclusion policy is missing or changed",
            observed=raw.get("exclusion_policy"),
            expected=EXCLUSION_POLICY,
        )
    include_paths = raw.get("include_paths")
    if not isinstance(include_paths, list) or not all(isinstance(item, str) for item in include_paths):
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree include path registry is invalid")

    root = Path(repository_root).resolve()
    rescanned_roots, rescanned_files = _scan_included_files(root, include_paths)
    if rescanned_roots != include_paths:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree include paths are not canonical and sorted",
            observed=include_paths,
            expected=rescanned_roots,
        )

    rows = raw.get("files")
    if not isinstance(rows, list) or not rows:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree manifest contains no source files")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Source-tree row schema is invalid",
                failing_field=f"files[{index}]",
            )
        relative = str(row["relative_path"])
        relative_path = Path(relative)
        if (
            relative in seen
            or relative in {"", "."}
            or relative_path.is_absolute()
            or relative_path.drive
            or ".." in relative_path.parts
        ):
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Source-tree path is duplicated or escapes the repository",
                observed=relative,
            )
        seen.add(relative)
        normalized.append(
            {
                "relative_path": relative_path.as_posix(),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]).upper(),
            }
        )
    normalized.sort(key=lambda row: row["relative_path"])
    if normalized != rescanned_files:
        expected_paths = {row["relative_path"] for row in normalized}
        observed_paths = {row["relative_path"] for row in rescanned_files}
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree rows differ from the exact files currently present under include paths",
            observed={
                "unregistered_files": sorted(observed_paths - expected_paths),
                "missing_files": sorted(expected_paths - observed_paths),
                "identity_changes": sorted(observed_paths & expected_paths),
            },
            expected="exact path, byte-count, and SHA-256 equality",
        )

    external = raw.get("external_references", [])
    if not isinstance(external, list):
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree external references must be a list")
    runtime_environment = raw.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree runtime environment is missing")
    runtime_environment_digest = stable_digest(runtime_environment)
    if runtime_environment_digest != str(raw.get("runtime_environment_digest", "")).upper():
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree runtime environment digest mismatch",
            observed=runtime_environment_digest,
            expected=raw.get("runtime_environment_digest"),
        )
    live_runtime_environment = probe_runtime_environment(root)
    live_runtime_environment_digest = stable_digest(live_runtime_environment)
    if dict(live_runtime_environment) != dict(runtime_environment):
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Live runtime environment differs from the frozen source manifest",
            observed={
                "runtime_environment_digest": live_runtime_environment_digest,
                "runtime_environment": live_runtime_environment,
            },
            expected={
                "runtime_environment_digest": runtime_environment_digest,
                "runtime_environment": dict(runtime_environment),
            },
        )
    digest = _content_digest(
        include_paths=include_paths,
        files=normalized,
        external_references=external,
        runtime_environment=runtime_environment,
    )
    if digest != str(raw.get("source_tree_digest", "")).upper():
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Source-tree aggregate digest mismatch",
            observed=digest,
            expected=raw.get("source_tree_digest"),
        )
    if require_clean:
        recorded_head = raw.get("git_head")
        current_head = _git(root, "rev-parse", "HEAD")
        # Untracked files under registered import roots are already rejected by
        # the exact rescan above.  Ignore unrelated untracked run artifacts so
        # a formal output directory cannot make its own frozen source checkout
        # look dirty; all tracked modifications remain fatal.
        current_status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
        if raw.get("git_dirty") is not False or not isinstance(recorded_head, str) or not recorded_head:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Formal source-tree manifest must have been generated from a clean Git checkout",
            )
        if current_head is None or current_status is None:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Formal source-tree validation could not re-read the current Git identity",
            )
        if current_head != recorded_head or current_status:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Current Git HEAD or worktree differs from the clean source-tree freeze",
                observed={
                    "git_head": current_head,
                    "git_dirty": bool(current_status),
                    "dirty_paths": current_status.splitlines()[:100],
                },
                expected={"git_head": recorded_head, "git_dirty": False},
            )
    return {
        "status": "PASS",
        "file_count": len(normalized),
        "include_path_count": len(include_paths),
        "source_tree_digest": digest,
        "runtime_environment_digest": live_runtime_environment_digest,
        "runtime_environment": live_runtime_environment,
        "manifest_sha256": sha256_file(manifest) if not isinstance(manifest, Mapping) else None,
        "git_head": raw.get("git_head"),
        "git_dirty": raw.get("git_dirty"),
    }
