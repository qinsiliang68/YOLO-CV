"""Publish literature and resource-reliability evidence into an active report.

This stage is deliberately independent of the shared analysis state and
manifest.  The caller may update those only after verifying the publication
receipt written here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.literature_evidence import (
    build_literature_evidence_matrix,
)
from stage1_gapvalue240.resource_reliability import (
    analyze_canonical_resources,
)


_OUTPUT_PATHS = {
    "literature": Path("tables/literature_evidence_matrix.csv"),
    "runs": Path("tables/resource_reliability_runs.csv"),
    "machines": Path("tables/resource_reliability_machines.csv"),
    "triads": Path("tables/resource_reliability_triads.csv"),
    "summary": Path("audit/resource_reliability_summary.json"),
    "receipt": Path("audit/literature_resource_publish_receipt.json"),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_literature_resources(
    *,
    inventory_path: str | Path,
    source_ledger_path: str | Path,
    output_dir: str | Path,
    expected_runs: int = 240,
    expected_triads: int = 80,
    max_workers: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the two evidence components and atomically publish their tables."""

    inventory = Path(inventory_path).resolve()
    source_ledger = Path(source_ledger_path).resolve()
    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError(f"output directory must end with .inprogress: {output}")
    for label, path in (("inventory", inventory), ("source ledger", source_ledger)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    targets = {key: output / relative for key, relative in _OUTPUT_PATHS.items()}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing literature/resource outputs: "
            + ", ".join(str(path) for path in existing)
        )

    literature = build_literature_evidence_matrix()
    resources = analyze_canonical_resources(
        inventory,
        source_ledger,
        expected_runs=expected_runs,
        expected_triads=expected_triads,
        max_workers=max_workers,
    )
    payloads = {
        "literature": _csv_bytes(literature),
        "runs": _csv_bytes(resources.runs),
        "machines": _csv_bytes(resources.machines),
        "triads": _csv_bytes(resources.triads),
        "summary": _json_bytes(resources.summary),
    }
    file_records = []
    for key in ("literature", "runs", "machines", "triads", "summary"):
        payload = payloads[key]
        file_records.append(
            {
                "path": _OUTPUT_PATHS[key].as_posix(),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    receipt = {
        "status": "PASS",
        "stage": "LITERATURE_AND_RESOURCE_RELIABILITY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_inventory": str(inventory),
        "canonical_inventory_sha256": _sha256_file(inventory),
        "source_file_ledger": str(source_ledger),
        "source_file_ledger_sha256": _sha256_file(source_ledger),
        "expected_runs": expected_runs,
        "expected_triads": expected_triads,
        "row_counts": {
            "literature_evidence_matrix": len(literature),
            "resource_reliability_runs": len(resources.runs),
            "resource_reliability_machines": len(resources.machines),
            "resource_reliability_triads": len(resources.triads),
        },
        "resource_summary": resources.summary,
        "files": file_records,
    }

    # Every data payload is fully serialized before the first target changes.
    # The receipt is written last and therefore acts as the publication marker.
    output.mkdir(parents=True, exist_ok=True)
    for key in ("literature", "runs", "machines", "triads", "summary"):
        _atomic_write(targets[key], payloads[key])
    _atomic_write(targets["receipt"], _json_bytes(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the primary-source literature matrix and canonical 240-run "
            "resource reliability tables into an active .inprogress report."
        )
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=240)
    parser.add_argument("--expected-triads", type=int, default=80)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = publish_literature_resources(
        inventory_path=args.inventory.resolve(),
        source_ledger_path=args.source_ledger.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_runs=args.expected_runs,
        expected_triads=args.expected_triads,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() tests
    raise SystemExit(main())
