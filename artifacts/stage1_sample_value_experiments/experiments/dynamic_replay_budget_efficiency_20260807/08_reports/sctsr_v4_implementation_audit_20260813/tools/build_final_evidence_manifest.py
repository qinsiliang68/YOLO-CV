from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def extended(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    output = args.output.resolve()
    root_extended = extended(root)
    output_relative = output.relative_to(root).as_posix()
    self_validation_relative = "reports/STAGED_EVIDENCE_BYTES_VALIDATION.json"
    rows = []
    skipped_reparse_points = []
    for current, directories, files in os.walk(root_extended, topdown=True, followlinks=False):
        kept = []
        for name in directories:
            candidate = os.path.join(current, name)
            if name == "__pycache__":
                continue
            if os.path.islink(candidate):
                relative = os.path.relpath(candidate, root_extended).replace("\\", "/")
                skipped_reparse_points.append(relative)
            else:
                kept.append(name)
        directories[:] = kept
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            candidate = os.path.join(current, name)
            relative = os.path.relpath(candidate, root_extended).replace("\\", "/")
            if relative in {output_relative, self_validation_relative} or relative.endswith(".tmp"):
                continue
            stat = os.stat(candidate, follow_symlinks=False)
            rows.append(
                {
                    "relative_path": relative,
                    "bytes": stat.st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    rows.sort(key=lambda row: row["relative_path"])
    total_bytes = sum(row["bytes"] for row in rows)
    payload = {
        "schema_version": "stage1.sctsr.final_evidence_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_source_commit": args.implementation_source_commit,
        "evidence_root": root.as_posix(),
        "file_count_excluding_manifest": len(rows),
        "total_bytes_excluding_manifest": total_bytes,
        "self_validation_exclusions": [output_relative, self_validation_relative],
        "skipped_reparse_points": sorted(skipped_reparse_points),
        "files": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_digest"] = hashlib.sha256(canonical).hexdigest().upper()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "file_count": len(rows), "total_bytes": total_bytes, "manifest_digest": payload["manifest_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
