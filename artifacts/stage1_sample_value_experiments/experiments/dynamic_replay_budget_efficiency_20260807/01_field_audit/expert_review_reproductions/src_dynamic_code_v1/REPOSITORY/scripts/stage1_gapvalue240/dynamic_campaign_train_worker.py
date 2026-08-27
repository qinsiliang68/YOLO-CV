from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.aiops import exit_code_for_exception
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID
from stage1_gapvalue240.campaign_worker import run_campaign_job


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run one preregistered dynamic replay segment.")
    if raw_args.count("--job-id") != 1:
        parser.error("exactly one --job-id is required; duplicate or implicit batching is forbidden")
    forbidden_batch_flags = {
        "--job-list",
        "--job-range",
        "--count",
        "--max-jobs",
        "--next-job",
        "--once",
    }
    observed_forbidden = sorted(forbidden_batch_flags.intersection(raw_args))
    if observed_forbidden:
        parser.error(f"batch scheduling flags are forbidden: {observed_forbidden}")
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-canonical-lock-sha256", required=True)
    parser.add_argument(
        "--campaign-root",
        default=str(
            _BootstrapPath(__file__).resolve().parents[2]
            / "artifacts/stage1_sample_value_experiments/experiments"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument(
        "--allow-dirty-code",
        action="store_true",
        help="Development smoke only; formal queue commands never set this flag.",
    )
    parsed = parser.parse_args(raw_args)
    job_id = str(parsed.job_id)
    if any(token in job_id for token in (",", ";", "..")) or any(char.isspace() for char in job_id):
        parser.error("--job-id must identify one literal physical job")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_campaign_job(
            args.machine_config,
            Path(args.campaign_root),
            args.job_id,
            release_path=Path(args.release).resolve(),
            assignment_path=Path(args.assignment).resolve(),
            expected_release_id=str(args.expected_release_id),
            expected_canonical_lock_sha256=str(args.expected_canonical_lock_sha256),
            allow_dirty_code=bool(args.allow_dirty_code),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "job_id": args.job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "exit_code": exit_code_for_exception(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return exit_code_for_exception(exc)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "job_id": result.job_id,
                "action": result.action,
                "output_dir": str(result.output_dir),
                "completed_epoch": result.completed_epoch,
                "state_path": str(result.state_path),
                "result_path": str(result.result_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
