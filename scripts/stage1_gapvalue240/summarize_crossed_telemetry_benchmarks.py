from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_benchmark import summarize_crossed_telemetry_benchmarks
from stage1_gapvalue240.util import atomic_write_json, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a crossed local telemetry benchmark pair.")
    parser.add_argument("--baseline-first", required=True)
    parser.add_argument("--telemetry-first", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_path = Path(args.baseline_first).resolve()
    telemetry_path = Path(args.telemetry_first).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = summarize_crossed_telemetry_benchmarks(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        json.loads(telemetry_path.read_text(encoding="utf-8")),
    )
    atomic_write_json(
        output,
        {
            **report,
            "baseline_first_result": str(baseline_path),
            "baseline_first_sha256": sha256_file(baseline_path),
            "telemetry_first_result": str(telemetry_path),
            "telemetry_first_sha256": sha256_file(telemetry_path),
        },
    )
    print(json.dumps(json.loads(output.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
