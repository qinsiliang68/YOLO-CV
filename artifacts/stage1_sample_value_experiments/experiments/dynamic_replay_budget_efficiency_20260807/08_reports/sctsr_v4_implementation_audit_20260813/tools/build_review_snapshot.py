from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOTS = (
    "stage1_sctsr_v4",
    "scripts/stage1_sctsr_v4",
    "configs/stage1_sctsr_v4",
    "integrations/ultralytics",
    "tests/stage1_sctsr_v4",
)


def git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    names = git(repository, "ls-tree", "-r", "--name-only", arguments.source_commit, "--", *ROOTS).decode("utf-8").splitlines()
    rows = []
    for index, relative in enumerate(names, 1):
        category = "tests" if relative.startswith("tests/") else "source"
        suffix = Path(relative).suffix or ".bin"
        snapshot = output / category / f"F{index:04d}{suffix}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        payload = git(repository, "cat-file", "blob", f"{arguments.source_commit}:{relative}")
        snapshot.write_bytes(payload)
        blob_oid = git(repository, "rev-parse", f"{arguments.source_commit}:{relative}").decode("ascii").strip()
        rows.append(
            {
                "original_relative_path": relative,
                "snapshot_relative_path": snapshot.relative_to(output.parent).as_posix(),
                "git_blob_oid": blob_oid,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
                "category": category,
            }
        )
    core = {
        "schema_version": "stage1.sctsr.reviewed_file_snapshot.v1",
        "implementation_source_commit": arguments.source_commit,
        "file_count": len(rows),
        "files": rows,
    }
    encoded = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report = {**core, "manifest_digest": hashlib.sha256(encoded).hexdigest().upper()}
    (output.parent / "REVIEWED_FILE_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "file_count": len(rows), "manifest_digest": report["manifest_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
