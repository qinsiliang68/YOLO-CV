#!/usr/bin/env python3
"""Merge the legacy 155-paper ledger with the new OpenAlex manual-screen pool."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_discovery_v2 import build_manual_screen_queue  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("manual queue is empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openalex-triage", required=True, type=Path)
    parser.add_argument("--legacy-matrix", required=True, type=Path)
    parser.add_argument("--legacy-full-text-ledger", required=True, type=Path)
    parser.add_argument("--legacy-literature-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    openalex_rows = _read_csv(args.openalex_triage)
    legacy_rows = _read_csv(args.legacy_matrix)
    full_text_rows = _read_csv(args.legacy_full_text_ledger)
    full_titles = {str(row.get("title", "")).strip() for row in full_text_rows}
    note_ids = {
        str(row.get("evidence_id", ""))
        for row in legacy_rows
        if str(row.get("title", "")).strip() in full_titles
    }
    source_ids: set[str] = set()
    for row in legacy_rows:
        evidence_id = str(row.get("evidence_id", ""))
        matching = [item for item in full_text_rows if item.get("title") == row.get("title")]
        if not matching:
            continue
        raw = str(matching[0].get("local_pdf", "")).strip()
        if raw and (args.legacy_literature_root / raw).is_file():
            source_ids.add(evidence_id)
    queue = build_manual_screen_queue(
        openalex_rows,
        legacy_rows,
        legacy_note_ids=note_ids,
        legacy_source_ids=source_ids,
    )
    _write_csv(args.output, queue)
    bands: dict[str, int] = {}
    for row in queue:
        bands[row["queue_band"]] = bands.get(row["queue_band"], 0) + 1
    print(f"queue={len(queue)} bands={bands} legacy_sources_present={len(source_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
