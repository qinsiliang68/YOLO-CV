from __future__ import annotations

import os,platform,subprocess,sys
from pathlib import Path
from typing import Any,Mapping,Sequence

from .errors import ErrorCode,SctsrError
from .serialization import load_json,sha256_file,stable_digest


def _git(root:Path,*args:str)->str|None:
    try:return subprocess.check_output(["git","-C",str(root),*args],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None


def build_source_tree_manifest(root:str|Path,include_paths:Sequence[str],*,external_references:Sequence[dict[str,str]]|None=None)->dict[str,Any]:
    base=Path(root).resolve(); files=[]
    for relative in include_paths:
        path=base/relative
        if path.is_file():candidates=[path]
        elif path.is_dir():candidates=[p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
        else:continue
        for file in sorted(candidates):
            files.append({"relative_path":file.relative_to(base).as_posix(),"bytes":file.stat().st_size,"sha256":sha256_file(file)})
    files=sorted(files,key=lambda row:row["relative_path"])
    return {"schema_version":"stage1.sctsr.source_tree_manifest.v1","root":base.as_posix(),"git_head":_git(base,"rev-parse","HEAD"),"git_dirty":bool(_git(base,"status","--porcelain")),"python":sys.version,"platform":platform.platform(),"files":files,"external_references":list(external_references or []),"source_tree_digest":stable_digest({"files":files,"external_references":list(external_references or [])})}


def validate_source_tree_manifest(
    manifest: Mapping[str, Any] | str | Path,
    repository_root: str | Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Verify every source byte named by a release source-tree manifest.

    The stored absolute ``root`` is informational only.  All paths are
    resolved under the caller-provided checkout so a manifest cannot redirect
    verification to a different local tree.
    """

    raw: Any = load_json(manifest) if not isinstance(manifest, Mapping) else manifest
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "stage1.sctsr.source_tree_manifest.v1":
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree manifest schema is invalid")
    rows = raw.get("files")
    if not isinstance(rows, list) or not rows:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree manifest contains no source files")
    root = Path(repository_root).resolve()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree row schema is invalid", failing_field=f"files[{index}]")
        relative = str(row["relative_path"])
        relative_path = Path(relative)
        if relative in seen or relative in {"", "."} or relative_path.is_absolute() or relative_path.drive or ".." in relative_path.parts:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree path is duplicated or escapes the repository", observed=relative)
        seen.add(relative)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree path escapes the repository", observed=relative) from exc
        if not path.is_file():
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Registered source file is missing", artifact_path=str(path))
        observed = {"relative_path": relative_path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected = {"relative_path": relative, "bytes": int(row["bytes"]), "sha256": str(row["sha256"]).upper()}
        if observed != expected:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Registered source file identity changed", observed=observed, expected=expected)
        normalized.append(observed)
    normalized.sort(key=lambda row: row["relative_path"])
    external = raw.get("external_references", [])
    if not isinstance(external, list):
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree external references must be a list")
    digest = stable_digest({"files": normalized, "external_references": external})
    if digest != str(raw.get("source_tree_digest", "")).upper():
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Source-tree aggregate digest mismatch", observed=digest, expected=raw.get("source_tree_digest"))
    if require_clean and raw.get("git_dirty") is not False:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Formal source-tree manifest must have been generated from a clean checkout")
    return {
        "status": "PASS",
        "file_count": len(normalized),
        "source_tree_digest": digest,
        "manifest_sha256": sha256_file(manifest) if not isinstance(manifest, Mapping) else None,
        "git_head": raw.get("git_head"),
        "git_dirty": raw.get("git_dirty"),
    }
