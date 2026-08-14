from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.engineering_canary import run_real_engineering_canary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real-image, real-YOLO SCTSR engineering canary; never starts formal training",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=20260814)
    parser.add_argument("--device", default=None)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        require_receipt_outside_artifact_root(arguments.output, arguments.artifact_root)
        return run_real_engineering_canary(
            arguments.artifact_root,
            repository_root=arguments.repository_root,
            dataset_root=arguments.dataset_root,
            training_seed=arguments.training_seed,
            device=arguments.device,
        )

    return run_cli("run_engineering_canary", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
