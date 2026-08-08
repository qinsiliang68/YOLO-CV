from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json

from stage1_gapvalue240.campaign_dry_generation import dry_generate_campaign_v2
from stage1_gapvalue240.campaign_engineering_gate import REQUIRED_EVIDENCE_SCHEMAS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dry-generate queue v2, gate v2, release and assignment without activation")
    p.add_argument("--preregistration-dir", required=True)
    p.add_argument("--monitor-source", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--machine-configs-dir", required=True)
    p.add_argument("--slot-mapping-json", required=True)
    p.add_argument("--evidence", action="append", required=True, help="TYPE=PATH")
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--pilot-seed", action="append", required=True)
    p.add_argument("--assignment-id", default="ASSIGNMENT_V2_DRY")
    args = p.parse_args(argv)
    reports = {}
    for item in args.evidence:
        if "=" not in item:
            p.error("--evidence must be TYPE=PATH")
        key, value = item.split("=", 1)
        reports[key] = value
    if set(reports) != set(REQUIRED_EVIDENCE_SCHEMAS):
        p.error("all required evidence types must be provided exactly once")
    with open(args.slot_mapping_json, encoding="utf-8") as stream:
        slots = json.load(stream)
    report = dry_generate_campaign_v2(
        args.preregistration_dir,
        args.monitor_source,
        output_root=args.output_root,
        repo_root=args.repo_root,
        machine_configs_dir=args.machine_configs_dir,
        slot_mapping=slots,
        raw_evidence_reports=reports,
        campaign_id=args.campaign_id,
        pilot_seed_ids=tuple(args.pilot_seed),
        assignment_id=args.assignment_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
