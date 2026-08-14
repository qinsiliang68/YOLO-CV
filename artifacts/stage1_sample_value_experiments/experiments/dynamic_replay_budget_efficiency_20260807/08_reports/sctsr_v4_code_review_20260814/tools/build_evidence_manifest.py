from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


BASELINE_COMMIT = "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f"
IMPLEMENTATION_COMMIT = "e9b6df61b0eb02e1d32c29175644f1c2af545afc"
DELIVERY_COMMIT = "f285754108c7b8e37afd7f5f0fa58fe8fb23d38a"
TASKBOOK_BLOB = "b201d021712e9c6614e119d35f0e14bdf405c6be"
MANIFEST_NAME = "EVIDENCE_MANIFEST.json"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_manifest() -> dict[str, Any]:
    report_root = Path(__file__).resolve().parents[1]
    repository_root = Path(_git(report_root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    report_relative = report_root.relative_to(repository_root).as_posix()
    raw = _git(repository_root, "ls-files", "--stage", "-z", "--", report_relative)
    files: list[dict[str, Any]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, git_blob, stage = metadata.decode("ascii").split()
        relative = path_bytes.decode("utf-8").replace("\\", "/")
        if stage != "0" or relative.endswith("/" + MANIFEST_NAME):
            continue
        blob = _git(repository_root, "cat-file", "blob", git_blob)
        files.append(
            {
                "relative_path": relative[len(report_relative) + 1 :],
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest().upper(),
                "git_blob": git_blob,
                "git_mode": mode,
            }
        )
    files.sort(key=lambda row: row["relative_path"])
    core: dict[str, Any] = {
        "schema_version": "stage1.sctsr.code_review_evidence_manifest.v1",
        "review_identity": {
            "baseline_commit": BASELINE_COMMIT,
            "implementation_freeze_commit": IMPLEMENTATION_COMMIT,
            "delivery_commit": DELIVERY_COMMIT,
            "taskbook_blob_sha": TASKBOOK_BLOB,
        },
        "coverage": {
            "root": report_relative,
            "source": "GIT_INDEX_STAGE_0_BLOBS",
            "excludes": [
                MANIFEST_NAME,
                "ignored runtime/ canary trees",
                "external 154 MB engineering-canary checkpoint",
            ],
            "exclusion_reason": "The manifest cannot hash itself. Large transient runtime trees are represented by compact SHA-bound receipts and snapshots rather than committed binary artifacts.",
        },
        "file_count": len(files),
        "files": files,
    }
    core["files_digest"] = hashlib.sha256(_canonical_bytes(files)).hexdigest().upper()
    core["manifest_core_digest"] = hashlib.sha256(_canonical_bytes(core)).hexdigest().upper()
    return core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / MANIFEST_NAME)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = build_manifest()
    if args.verify:
        observed = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != current:
            print(json.dumps({"status": "FAIL", "expected": current, "observed": observed}, ensure_ascii=False, sort_keys=True))
            return 1
        print(json.dumps({"status": "PASS", "file_count": current["file_count"], "files_digest": current["files_digest"]}, ensure_ascii=False, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(current, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "file_count": current["file_count"], "files_digest": current["files_digest"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
