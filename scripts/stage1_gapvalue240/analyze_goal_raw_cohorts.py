"""Publish all-80-triad raw/calibrated outcome-cohort mechanism evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1_gapvalue240.analyze_goal_raw_frontier import (  # noqa: E402
    publish_tables_atomic,
)
from stage1_gapvalue240.raw_cohort_mechanisms import (  # noqa: E402
    RawCohortMechanismError,
    run_raw_cohort_mechanism_analysis,
)


OUTPUT_TABLES = (
    "raw_cohort_mechanism_membership.csv",
    "raw_cohort_mechanism_feature_dictionary.csv",
    "raw_cohort_mechanism_pair_features.csv",
    "raw_cohort_mechanism_cohort_summaries.csv",
    "raw_cohort_mechanism_contrasts.csv",
    "raw_cohort_mechanism_scoretype_differences.csv",
)
SUMMARY_NAME = "raw_cohort_mechanism_summary.json"
INPUT_TABLES = (
    "triad_outcomes_80.csv",
    "raw_frontier_paired_tail_shift_summary.csv",
    "raw_frontier_paired_dominance.csv",
    "raw_frontier_run_probability_metrics.csv",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this module's seven fixed output files.",
    )
    args = parser.parse_args()
    tables_dir = args.tables_dir.resolve()
    if not any(path.name.endswith(".inprogress") for path in tables_dir.parents):
        raise RawCohortMechanismError(
            "tables-dir must remain under a .inprogress report directory"
        )
    started_at = _utc_now()
    start = time.perf_counter()
    print("[raw-cohort] Loading all 80 canonical triads", flush=True)
    tables, summary = run_raw_cohort_mechanism_analysis(
        tables_dir,
        permutations=args.permutations,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    summary = dict(summary)
    summary.update(
        {
            "analysis_id": "gapvalue240_raw_cohort_mechanism_v1",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "compute_seconds": round(time.perf_counter() - start, 6),
            "input_files": {
                name: {
                    "row_count": int(len(pd.read_csv(tables_dir / name, low_memory=False))),
                    "size_bytes": int((tables_dir / name).stat().st_size),
                    "sha256": _sha256(tables_dir / name),
                }
                for name in INPUT_TABLES
            },
        }
    )
    print("[raw-cohort] Atomically publishing six CSV tables and summary", flush=True)
    published = publish_tables_atomic(
        tables_dir,
        tables,
        summary,
        overwrite=args.overwrite,
        _table_filenames=OUTPUT_TABLES,
        _summary_filename=SUMMARY_NAME,
    )
    final = json.loads(Path(published[SUMMARY_NAME]).read_text(encoding="utf-8"))
    print(
        "[raw-cohort] COMPLETE: "
        f"cohorts={final['cohort_counts']}, "
        f"contrast_tests={final['contrast_test_count']}, "
        f"global_FDR={final['global_fdr_significant_count']}",
        flush=True,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
