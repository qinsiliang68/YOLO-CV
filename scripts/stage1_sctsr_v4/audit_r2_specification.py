from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.formal_pool_inputs import load_formal_pool_inputs
from stage1_sctsr_v4.r2_specification_audit import audit_r2_specifications
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only comparison of preregisterable R2 specifications on frozen assets"
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        registry = load_asset_registry(arguments.asset_registry)
        inputs = load_formal_pool_inputs(registry, arguments.repository_root)
        audit = audit_r2_specifications(
            inputs.base_records,
            inputs.t_pool.records,
            selection_seed=arguments.selection_seed,
        )
        atomic_write_json(arguments.audit_output, audit)
        return {
            "status": audit["status"],
            "recommended_option": audit["recommended_option"],
            "activation_status": audit["activation_status"],
            "audit_path": arguments.audit_output.resolve().as_posix(),
            "audit_bytes": arguments.audit_output.stat().st_size,
            "audit_sha256": sha256_file(arguments.audit_output),
            "audit_digest": audit["audit_digest"],
            "formal_training_started": False,
            "identity_pool_generated": False,
        }

    return run_cli("audit_r2_specification", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
