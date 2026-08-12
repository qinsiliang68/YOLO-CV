from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.review_claim_ledger import (
    load_expert_findings,
    validate_v3_assessments,
    write_expert_findings_ledger,
    write_merged_review_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the immutable expert review claim ledger."
    )
    parser.add_argument("--findings-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--v3-assessments", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = load_expert_findings(args.findings_json)
    merged_digest = None
    if args.v3_assessments is not None:
        if args.merged_output is None:
            raise SystemExit("--merged-output is required with --v3-assessments")
        with args.v3_assessments.open("r", encoding="utf-8-sig", newline="") as handle:
            assessments = list(csv.DictReader(handle))
        validate_v3_assessments(findings, assessments, repo_root=args.repo_root)
        merged_digest = write_merged_review_ledger(
            findings,
            assessments,
            args.merged_output,
            repo_root=args.repo_root,
        )
    elif args.merged_output is not None:
        raise SystemExit("--merged-output requires --v3-assessments")
    write_expert_findings_ledger(findings, args.output)
    counts = {
        severity: sum(finding.severity == severity for finding in findings)
        for severity in ("P0", "High", "Moderate")
    }
    print(
        "expert_findings_ledger=PASS "
        f"rows={len(findings)} p0={counts['P0']} high={counts['High']} "
        f"moderate={counts['Moderate']} output={args.output}"
    )
    if merged_digest is not None:
        print(f"merged_review_ledger=PASS sha256={merged_digest} output={args.merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
