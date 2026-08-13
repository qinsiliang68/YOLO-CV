from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def staged_bytes(repository_root: Path, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f":{relative_path}"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"staged blob missing for {relative_path}: {completed.stderr.decode('utf-8', errors='replace')}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    evidence = args.evidence_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest["files"]:
        absolute = evidence / row["relative_path"]
        repository_relative = absolute.relative_to(repository).as_posix()
        staged = staged_bytes(repository, repository_relative)
        if len(staged) != row["bytes"] or sha256(staged) != row["sha256"]:
            mismatches.append(
                {
                    "relative_path": row["relative_path"],
                    "expected_bytes": row["bytes"],
                    "observed_bytes": len(staged),
                    "expected_sha256": row["sha256"],
                    "observed_sha256": sha256(staged),
                }
            )
    manifest_relative = args.manifest.resolve().relative_to(repository).as_posix()
    manifest_worktree = args.manifest.read_bytes()
    manifest_staged = staged_bytes(repository, manifest_relative)
    if manifest_worktree != manifest_staged:
        mismatches.append(
            {
                "relative_path": args.manifest.resolve().relative_to(evidence).as_posix(),
                "expected_bytes": len(manifest_worktree),
                "observed_bytes": len(manifest_staged),
                "expected_sha256": sha256(manifest_worktree),
                "observed_sha256": sha256(manifest_staged),
            }
        )
    result = {
        "schema_version": "stage1.sctsr.staged_evidence_byte_validation.v1",
        "status": "PASS" if not mismatches else "FAIL",
        "checked_file_count": len(manifest["files"]) + 1,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
