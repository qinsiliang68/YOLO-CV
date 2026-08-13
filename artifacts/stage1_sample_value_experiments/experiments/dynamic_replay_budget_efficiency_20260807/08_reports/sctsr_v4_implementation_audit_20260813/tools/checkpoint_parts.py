from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA = "stage1.sctsr.checkpoint_parts.v1"
MANIFEST_NAME = "CHECKPOINT_PARTS_MANIFEST.json"
MAX_GITHUB_BLOB_BYTES = 100_000_000
DEFAULT_PART_BYTES = 90_000_000
BUFFER_BYTES = 1024 * 1024


def _safe(path: str | Path) -> Path:
    resolved = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    return Path(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _copy_exact(source: BinaryIO, destination: BinaryIO, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    while written < limit:
        chunk = source.read(min(BUFFER_BYTES, limit - written))
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        written += len(chunk)
    destination.flush()
    os.fsync(destination.fileno())
    return written, digest.hexdigest().upper()


def split_file(source: str | Path, output_dir: str | Path, *, part_size: int = DEFAULT_PART_BYTES) -> dict[str, Any]:
    if type(part_size) is not int or part_size <= 0 or part_size >= MAX_GITHUB_BLOB_BYTES:
        raise ValueError("part_size must be a positive integer below GitHub's 100 MB blob limit")
    source_path = _safe(source)
    target = _safe(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)

    original_bytes = source_path.stat().st_size
    parts: list[dict[str, Any]] = []
    with source_path.open("rb") as stream:
        index = 1
        consumed = 0
        while consumed < original_bytes:
            filename = f"{source_path.name}.part{index:04d}"
            part = target / filename
            expected = min(part_size, original_bytes - consumed)
            with part.open("xb") as destination:
                observed, sha = _copy_exact(stream, destination, expected)
            if observed != expected:
                raise IOError(f"short split write for {filename}: {observed} != {expected}")
            parts.append({"index": index, "filename": filename, "bytes": observed, "sha256": sha})
            consumed += observed
            index += 1

    core = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "assembly_algorithm": "RAW_BYTE_CONCATENATION_IN_ASCENDING_INDEX_ORDER",
        "source_filename": source_path.name,
        "original_bytes": original_bytes,
        "original_sha256": _sha256(source_path),
        "part_size_limit_bytes": part_size,
        "github_single_blob_limit_bytes": MAX_GITHUB_BLOB_BYTES,
        "part_count": len(parts),
        "parts": parts,
    }
    core["manifest_digest"] = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    _atomic_json(target / MANIFEST_NAME, core)
    return core


def reassemble_file(manifest_path: str | Path, output: str | Path) -> dict[str, Any]:
    manifest_file = _safe(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "PASS":
        raise ValueError("checkpoint-parts manifest schema or status is invalid")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != manifest.get("part_count"):
        raise ValueError("checkpoint-parts manifest count is invalid")
    if [row.get("index") for row in parts if isinstance(row, dict)] != list(range(1, len(parts) + 1)):
        raise ValueError("checkpoint-parts indices are not contiguous")

    destination = _safe(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.inprogress")
    total = 0
    full_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as combined:
            for row in parts:
                part = manifest_file.parent / str(row["filename"])
                if not part.is_file() or part.stat().st_size != row["bytes"]:
                    raise ValueError(f"part bytes mismatch: {row['filename']}")
                observed_digest = hashlib.sha256()
                with part.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
                        observed_digest.update(chunk)
                        full_digest.update(chunk)
                        combined.write(chunk)
                        total += len(chunk)
                if observed_digest.hexdigest().upper() != row["sha256"]:
                    raise ValueError(f"part SHA-256 mismatch: {row['filename']}")
            combined.flush()
            os.fsync(combined.fileno())
        observed_sha = full_digest.hexdigest().upper()
        if total != manifest.get("original_bytes") or observed_sha != manifest.get("original_sha256"):
            raise ValueError("reassembled checkpoint bytes or SHA-256 mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": "stage1.sctsr.checkpoint_reassembly_receipt.v1",
        "status": "PASS",
        "output": destination.as_posix(),
        "bytes": total,
        "sha256": full_digest.hexdigest().upper(),
        "part_count": len(parts),
        "manifest_digest": manifest["manifest_digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split or reassemble a GitHub-size-bound SCTSR engineering checkpoint")
    subparsers = parser.add_subparsers(dest="action", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--part-size", type=int, default=DEFAULT_PART_BYTES)
    join = subparsers.add_parser("reassemble")
    join.add_argument("--manifest", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "split":
        result = split_file(args.source, args.output_dir, part_size=args.part_size)
    else:
        result = reassemble_file(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
