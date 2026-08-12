#!/usr/bin/env python3
"""Write the Stage1 global research-finalization audit without executing work."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.global_completion_audit_v2 import (
    build_global_completion_audit,
)

EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "08_reports" / "COMPLETION_AUDIT.json"
DEFAULT_MIRROR = (
    Path(r"C:\Users\28898\Desktop") / "YOLO\u7b14\u8bb0"
    / "01_Stage1_\u6709\u9650\u9884\u7b97\u52a8\u6001\u56de\u6d41_\u6587\u732e\u8bc1\u636e_20260810"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--desktop-literature-mirror", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _write_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    report = build_global_completion_audit(
        repo_root=args.repo_root,
        experiment_root=args.experiment_root,
        desktop_literature_mirror=args.desktop_literature_mirror,
    )
    _write_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "INCOMPLETE_SOURCE_MISSING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
