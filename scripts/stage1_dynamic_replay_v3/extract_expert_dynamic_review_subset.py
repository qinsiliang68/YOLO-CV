from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.expert_delivery_audit import (
    extract_manifested_tar_subset,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a manifest-verified source subset for the DynamicReplay review."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--selection-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [
        line.strip()
        for line in args.selection_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    result = extract_manifested_tar_subset(
        args.archive,
        args.output_dir,
        relative_paths=selected,
    )
    output = args.output_dir.resolve()
    manifest = output / "SOURCE_SELECTION_MANIFEST.csv"
    with manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "container_filename",
                "relative_path",
                "size_bytes",
                "sha256",
                "manifest_expected_sha256",
                "manifest_hash_match",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.members)
    receipt = {
        "schema_version": "stage1.expert_dynamic_review_source_subset.v1",
        **result.summary,
        "selection_file": args.selection_file.resolve().relative_to(REPO_ROOT).as_posix(),
        "selection_file_sha256": _sha256(args.selection_file),
        "source_manifest_sha256": _sha256(manifest),
        "executed_expert_code": False,
    }
    (output / "SOURCE_SELECTION_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
