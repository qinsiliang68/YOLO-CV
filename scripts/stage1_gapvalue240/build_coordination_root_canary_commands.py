from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_canary import build_coordination_canary_commands


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate one coordination-root canary command per machine")
    p.add_argument("--machine-configs-dir", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--expected-machine-ids", required=True)
    p.add_argument("--coordination-root-placeholder", default="<SET_SHARED_COORDINATION_ROOT>")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    result = build_coordination_canary_commands(
        args.machine_configs_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        campaign_id=args.campaign_id,
        generation=args.generation,
        expected_machine_ids=_csv(args.expected_machine_ids),
        coordination_root_placeholder=args.coordination_root_placeholder,
    )
    for path in result.values():
        print(Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
