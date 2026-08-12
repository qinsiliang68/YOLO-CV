#!/usr/bin/env python3
"""Acquire a batch of primary literature sources with immutable receipts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_source_acquisition_v2 import (  # noqa: E402
    SourceAcquisitionError,
    SourceRequest,
    acquire_source,
)


def _read_requests(path: Path) -> list[SourceRequest]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SourceRequest(
            paper_id=row["paper_id"],
            artifact_role=row["artifact_role"],
            url=row["url"],
            destination=row["destination"],
            source_authority=row["source_authority"],
        )
        for row in rows
    ]


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-ledger", required=True, type=Path)
    parser.add_argument("--failure-ledger", required=True, type=Path)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests_to_run = _read_requests(args.requests)
    session = requests.Session()
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for request in requests_to_run:
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                row = acquire_source(
                    request,
                    corpus_root=args.corpus_root,
                    session=session,
                    timeout_seconds=args.timeout_seconds,
                )
            except (SourceAcquisitionError, requests.RequestException) as exc:
                last_error = str(exc)
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
            else:
                successes.append(row)
                break
        else:
            failures.append(
                {
                    "paper_id": request.paper_id,
                    "artifact_role": request.artifact_role,
                    "url": request.url,
                    "destination": request.destination,
                    "attempts": args.retries,
                    "error": last_error,
                }
            )
    success_fields = [
        "paper_id",
        "artifact_role",
        "path",
        "url",
        "retrieved_at",
        "http_status",
        "content_type",
        "bytes",
        "sha256",
        "retrieval_method",
        "source_authority",
        "final_url",
        "receipt_path",
        "reused_existing",
    ]
    failure_fields = ["paper_id", "artifact_role", "url", "destination", "attempts", "error"]
    _write_csv(args.output_ledger, successes, success_fields)
    _write_csv(args.failure_ledger, failures, failure_fields)
    print(f"requested={len(requests_to_run)} acquired={len(successes)} failed={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
