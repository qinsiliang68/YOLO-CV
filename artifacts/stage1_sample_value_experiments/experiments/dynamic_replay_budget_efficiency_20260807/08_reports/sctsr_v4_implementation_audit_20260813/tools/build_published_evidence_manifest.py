from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "stage1.sctsr.published_evidence_manifest.v2"


def _git(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _index_rows(repository: Path, evidence_relative: str) -> list[tuple[str, str, str]]:
    raw = _git(repository, "ls-files", "--stage", "-z", "--", evidence_relative)
    rows: list[tuple[str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise ValueError(f"unmerged Git index entry: {raw_path!r}")
        rows.append((mode, object_id, raw_path.decode("utf-8")))
    return rows


def build_published_manifest(
    *,
    repository_root: str | Path,
    evidence_root: str | Path,
    implementation_source_commit: str,
    output: str | Path,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    evidence = Path(evidence_root).resolve()
    destination = Path(output).resolve()
    evidence_relative = evidence.relative_to(repository).as_posix()
    output_relative = destination.relative_to(repository).as_posix()
    validation_relative = (evidence / "reports/STAGED_EVIDENCE_BYTES_VALIDATION.json").relative_to(repository).as_posix()

    rows: list[dict[str, Any]] = []
    for mode, object_id, repository_relative in _index_rows(repository, evidence_relative):
        if repository_relative in {output_relative, validation_relative}:
            continue
        payload = _git(repository, "show", f":{repository_relative}")
        rows.append(
            {
                "relative_path": Path(repository_relative).relative_to(Path(evidence_relative)).as_posix(),
                "bytes": len(payload),
                "sha256": _sha(payload),
                "git_mode": mode,
                "git_blob_oid": object_id,
            }
        )
    rows.sort(key=lambda row: row["relative_path"])
    core = {
        "schema_version": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_source_commit": implementation_source_commit,
        "publication_scope": "GIT_INDEX_ONLY",
        "evidence_root_repository_relative": evidence_relative,
        "file_count_excluding_manifest": len(rows),
        "total_bytes_excluding_manifest": sum(int(row["bytes"]) for row in rows),
        "self_validation_exclusions": [
            destination.relative_to(evidence).as_posix(),
            (evidence / "reports/STAGED_EVIDENCE_BYTES_VALIDATION.json").relative_to(evidence).as_posix(),
        ],
        "local_only_heavy_artifact_registry": "reports/LOCAL_ONLY_HEAVY_ARTIFACTS.json",
        "files": rows,
    }
    result = {**core, "manifest_digest": _sha(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a byte manifest from the exact Git index publication set")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_published_manifest(
        repository_root=args.repository_root,
        evidence_root=args.evidence_root,
        implementation_source_commit=args.implementation_source_commit,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "file_count": result["file_count_excluding_manifest"],
                "total_bytes": result["total_bytes_excluding_manifest"],
                "manifest_digest": result["manifest_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
