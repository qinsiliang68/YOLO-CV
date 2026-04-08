from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    main_py = repo_root / "main.py"
    cmd = [sys.executable, str(main_py), "--task", "stage1_formal_gate_hn_ns_all", *sys.argv[1:]]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
