from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    runner = repo_root / "scripts" / "stage1_formal_gate_bucket_pilot.py"
    config = repo_root / "YOLOv11" / "configs" / "runtime" / "stage1_formal_gate_bucket_pilot_machine_b.json"
    cmd = [sys.executable, str(runner), "--config", str(config), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
