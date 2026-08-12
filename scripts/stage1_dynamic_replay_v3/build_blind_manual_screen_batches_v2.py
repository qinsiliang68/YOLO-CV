#!/usr/bin/env python3
"""Build deterministic, ranking-blind primary-source review batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_manual_screening_v2 import (  # noqa: E402
    blind_order_queue,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-queue", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-seed", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    with args.input_queue.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("input queue is empty")
    ordered = blind_order_queue(rows, frozen_seed=args.frozen_seed)
    _atomic_csv(args.output_queue, ordered)

    args.batch_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[Path] = []
    for offset in range(0, len(ordered), args.batch_size):
        batch_number = offset // args.batch_size + 1
        path = args.batch_dir / f"review_input_{batch_number:03d}.csv"
        _atomic_csv(path, ordered[offset : offset + args.batch_size])
        batch_paths.append(path)
    expected_names = {path.name for path in batch_paths}
    extras = sorted(path.name for path in args.batch_dir.glob("review_input_*.csv") if path.name not in expected_names)
    if extras:
        raise ValueError(f"stale review input batches detected: {extras}")

    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "frozen_seed": args.frozen_seed,
        "input_queue": args.input_queue.as_posix(),
        "input_queue_sha256": _sha256(args.input_queue),
        "output_queue": args.output_queue.as_posix(),
        "output_queue_sha256": _sha256(args.output_queue),
        "review_count": len(ordered),
        "batch_size": args.batch_size,
        "batch_count": len(batch_paths),
        "batches": [
            {
                "path": path.as_posix(),
                "rows": min(args.batch_size, len(ordered) - index * args.batch_size),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(batch_paths)
        ],
        "selection_credit_granted": False,
        "note": "Blind review order and batch generation do not constitute paper screening.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(f"reviews={len(ordered)} batches={len(batch_paths)} batch_size={args.batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
