from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_assignment import load_campaign_assignment
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID, active_run_queue_dir
from stage1_gapvalue240.campaign_lease import activate_assignment


ROOT = _BootstrapPath(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically activate one released assignment on a shared coordination root."
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--coordination-root", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = Path(args.campaign_root).resolve()
    assignment = load_campaign_assignment(
        active_run_queue_dir(campaign),
        Path(args.release).resolve(),
        Path(args.assignment).resolve(),
        expected_campaign_id=CAMPAIGN_ID,
        repo_root=Path(args.repo_root).resolve(),
    )
    manifest = json.loads(assignment.manifest_path.read_text(encoding="utf-8"))
    active = activate_assignment(
        Path(args.coordination_root).resolve(),
        campaign_id=CAMPAIGN_ID,
        release_id=assignment.release_id,
        assignment_id=assignment.assignment_id,
        assignment_sha256=assignment.sha256,
        job_ids=tuple(assignment.rows.job_id.astype(str)),
        expected_previous_assignment_sha256=manifest.get("supersedes_assignment_sha256"),
    )
    print(
        json.dumps(
            {
                "status": "ACTIVE",
                "campaign_id": active.campaign_id,
                "release_id": active.release_id,
                "assignment_id": active.assignment_id,
                "assignment_sha256": active.assignment_sha256,
                "job_count": len(active.job_ids),
                "active_assignment_path": str(active.path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
