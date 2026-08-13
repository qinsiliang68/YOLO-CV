from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one command and append an immutable receipt to COMMAND_INDEX.json")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = list(args.command)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        parser.error("a child command is required after --")

    root = args.evidence_root.resolve()
    index_path = root / "COMMAND_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    existing = [row for row in index["commands"] if row["name"] == args.name]
    if existing and not args.replace:
        raise SystemExit(f"command receipt already exists: {args.name}")

    command_dir = root / "commands" / args.name
    command_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(argv, cwd=args.cwd.resolve(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    stdout_path = command_dir / "stdout.log"
    stderr_path = command_dir / "stderr.log"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    receipt = {
        "schema_version": "stage1.sctsr.command_receipt.v1",
        "name": args.name,
        "argv": argv,
        "command_line": subprocess.list2cmdline(argv),
        "cwd": str(args.cwd.resolve()),
        "implementation_source_commit": args.implementation_source_commit,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "exit_code": completed.returncode,
        "stdout_path": stdout_path.relative_to(root).as_posix(),
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_path": stderr_path.relative_to(root).as_posix(),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": sha256_bytes(completed.stderr),
    }
    index["commands"] = [row for row in index["commands"] if row["name"] != args.name] + [receipt]
    atomic_json(index_path, index)
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
