from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_resource_validation import run_disk_gpu_preflight


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run canonical disk/GPU preflight")
    p.add_argument("--machine-config", required=True)
    p.add_argument("--canonical-lock", required=True)
    p.add_argument("--required-output-free-bytes", type=int, default=20 * 1024**3)
    p.add_argument("--benchmark-bytes", type=int, default=4 * 1024**2)
    p.add_argument("--allow-no-gpu", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    run_disk_gpu_preflight(
        args.machine_config,
        args.canonical_lock,
        output_path=args.output,
        required_output_free_bytes=args.required_output_free_bytes,
        benchmark_bytes=args.benchmark_bytes,
        require_gpu=not args.allow_no_gpu,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
