#!/usr/bin/env python3
"""Fetch preregistered OpenAlex discovery snapshots and build a candidate inventory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_discovery_v2 import (  # noqa: E402
    build_candidate_inventory,
    fetch_openalex_queries,
    load_query_plan,
    triage_candidate_inventory,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to publish an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = load_query_plan(args.query_plan)
    query_log, snapshots = fetch_openalex_queries(
        specs,
        corpus_root=args.corpus_root,
        per_page=args.per_page,
        force=args.force,
    )
    inventory = build_candidate_inventory(snapshots)
    triage = triage_candidate_inventory(inventory)
    _write_csv(args.corpus_root / "discovery" / "QUERY_LOG.csv", query_log)
    _write_csv(args.corpus_root / "discovery" / "CANDIDATE_INVENTORY_OPENALEX_v1.csv", inventory)
    _write_csv(args.corpus_root / "discovery" / "CANDIDATE_TRIAGE_OPENALEX_v1.csv", triage)
    manual = sum(row["prefilter_decision"] == "MANUAL_SCREEN_REQUIRED" for row in triage)
    print(
        f"queries={len(query_log)} raw_hits={sum(int(row['result_end']) for row in query_log)} "
        f"unique={len(inventory)} manual_screen={manual}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
