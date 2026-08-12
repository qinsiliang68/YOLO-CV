#!/usr/bin/env python3
"""Rebind trusted SCREENED reviews to a repaired canonical staging freeze."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_review_migration_v4 import (  # noqa: E402
    migrate_screened_reviews,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--source-review-dir", type=Path, required=True)
    parser.add_argument("--screening-queue", type=Path, required=True)
    parser.add_argument("--extraction-ledger", type=Path, required=True)
    parser.add_argument("--output-relative", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    result = migrate_screened_reviews(
        args.corpus_root,
        source_review_dir=args.source_review_dir,
        screening_queue=args.screening_queue,
        extraction_ledger=args.extraction_ledger,
        output_relative=args.output_relative,
        replace_existing=args.replace_existing,
    )
    print(
        f"status=PASS migrated={result.migrated_count} renamed={result.renamed_count} "
        f"output={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
