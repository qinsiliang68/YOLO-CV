from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.deployment_plan import (
    build_phase1_logical_jobs,
    build_seeded_random_deployment_plan,
    validate_deployment_plan,
)
from stage1_sctsr_v4.seed_registry import SeedRegistry
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible 12-active-plus-1-buffer SCTSR deployment plan without starting training"
    )
    parser.add_argument("--seed-registry", type=Path, required=True)
    parser.add_argument("--active-machine", action="append", required=True)
    parser.add_argument("--buffer-machine", required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        registry = SeedRegistry.from_mapping(load_json(arguments.seed_registry))
        jobs = build_phase1_logical_jobs(
            discovery_seeds=registry.discovery_seeds,
            confirmation_seeds=registry.confirmation_seeds,
        )
        plan = build_seeded_random_deployment_plan(
            jobs,
            active_machine_ids=arguments.active_machine,
            buffer_machine_id=arguments.buffer_machine,
            assignment_seed=arguments.assignment_seed,
        )
        atomic_write_json(arguments.plan_output, plan)
        checked = validate_deployment_plan(plan)
        return {
            **checked,
            "plan_path": arguments.plan_output.resolve().as_posix(),
            "plan_bytes": arguments.plan_output.stat().st_size,
            "plan_sha256": sha256_file(arguments.plan_output),
            "seed_registry_digest": registry.digest,
            "formal_training_started": False,
            "release_authorization_required": True,
        }

    return run_cli("build_deployment_plan", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
