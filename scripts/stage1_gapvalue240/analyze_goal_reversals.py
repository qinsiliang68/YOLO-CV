"""Publish same-selection cross-seed reversal and raw-tail mechanism evidence."""

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
from stage1_gapvalue240.reversal_mechanisms import (  # noqa: E402
    ReversalAnalysisError,
    run_same_selection_reversal_analysis,
)


OUTPUT_TABLES = (
    "reversal_digest_triads.csv",
    "reversal_digest_summary.csv",
    "reversal_feature_contrasts.csv",
    "reversal_epoch_cutoffs.csv",
    "reversal_epoch_timeline_contrasts.csv",
    "reversal_raw_tail_details.csv",
    "reversal_safe_frontier_details.csv",
    "reversal_raw_mechanism_contrasts.csv",
    "reversal_confound_crosswalk.csv",
)
SUMMARY_NAME = "reversal_analysis_summary.json"
INPUT_TABLES = (
    "treatment_selection_sets_80.csv",
    "triad_outcomes_80.csv",
    "unified_triad_feature_matrix.csv",
    "FEATURE_ROLE_REGISTRY.csv",
    "paired_epoch_dynamics_32000.csv",
    "raw_frontier_paired_tail_shift_summary.csv",
    "raw_frontier_paired_dominance.csv",
    "resource_reliability_triads.csv",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the ten fixed reversal-analysis outputs.",
    )
    args = parser.parse_args()
    tables_dir = args.tables_dir.resolve()
    if not any(path.name.endswith(".inprogress") for path in tables_dir.parents):
        raise ReversalAnalysisError("tables-dir must remain under a .inprogress report")
    print("[reversal] Loading canonical 80-triad evidence", flush=True)
    started = _utc_now()
    start = time.perf_counter()
    tables, summary = run_same_selection_reversal_analysis(tables_dir)
    summary = dict(summary)
    summary.update(
        {
            "analysis_id": "gapvalue240_same_selection_reversal_v1",
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "source_tables_read_only": True,
            "input_files": {
                name: {
                    "row_count": int(len(pd.read_csv(tables_dir / name, low_memory=False))),
                    "size_bytes": int((tables_dir / name).stat().st_size),
                    "sha256": _sha256(tables_dir / name),
                }
                for name in INPUT_TABLES
            },
            "compute_seconds": round(time.perf_counter() - start, 6),
        }
    )
    print("[reversal] Atomically publishing nine tables and JSON summary", flush=True)
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
        "[reversal] COMPLETE: "
        f"digests={final['reversal_digest_count']}, "
        f"triads={final['reversal_triad_count']}, "
        f"global_predictor_FDR={final['global_fdr_significant_feature_count']}",
        flush=True,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
