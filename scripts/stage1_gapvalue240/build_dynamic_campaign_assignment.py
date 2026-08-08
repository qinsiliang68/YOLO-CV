from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import pandas as pd

from stage1_gapvalue240.campaign_assignment import build_campaign_assignment
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID, active_run_queue_dir


ROOT = _BootstrapPath(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = (
    ROOT
    / "artifacts/stage1_sample_value_experiments/experiments"
    / CAMPAIGN_ID
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a versioned machine assignment and one standalone command per released job."
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--slot-map",
        default=str(ROOT / "configs/stage1_gapvalue240/DYNAMIC_MACHINE_SLOT_MAP_v1.csv"),
    )
    parser.add_argument("--seed-overrides")
    parser.add_argument("--supersedes-assignment")
    parser.add_argument("--reassignment-reason")
    parser.add_argument(
        "--machine-configs-dir",
        default=str(ROOT / "configs/stage1_gapvalue240/machines"),
    )
    return parser.parse_args(argv)


def _mapping(path: Path, *, key: str, value: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {key, value} - set(frame.columns)
    if missing:
        raise ValueError(f"mapping {path} missing columns: {sorted(missing)}")
    if frame.empty or frame[key].astype(str).duplicated().any():
        raise ValueError(f"mapping {path} has empty or duplicate {key}")
    if frame[key].astype(str).str.strip().eq("").any() or frame[value].astype(str).str.strip().eq("").any():
        raise ValueError(f"mapping {path} contains blank identities")
    return dict(zip(frame[key].astype(str), frame[value].astype(str), strict=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = Path(args.campaign_root).resolve()
    queue_dir = active_run_queue_dir(campaign)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else queue_dir / "assignments" / str(args.assignment_id)
    )
    overrides = None
    if args.seed_overrides:
        overrides = _mapping(
            Path(args.seed_overrides).resolve(),
            key="seed_id",
            value="machine_id",
        )
    result = build_campaign_assignment(
        queue_dir,
        Path(args.release).resolve(),
        output,
        campaign_id=CAMPAIGN_ID,
        assignment_id=str(args.assignment_id),
        machine_configs_dir=Path(args.machine_configs_dir).resolve(),
        slot_mapping=_mapping(
            Path(args.slot_map).resolve(),
            key="planning_slot",
            value="machine_id",
        ),
        seed_overrides=overrides,
        supersedes_assignment=(
            Path(args.supersedes_assignment).resolve()
            if args.supersedes_assignment
            else None
        ),
        reassignment_reason=args.reassignment_reason,
        repo_root=ROOT,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "assignment_manifest": str(result.manifest_path),
                "job_assignments": str(result.job_assignments_path),
                "block_assignments": str(result.block_assignments_path),
                "standalone_commands": str(result.standalone_commands_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
