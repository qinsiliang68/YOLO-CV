from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--task", "stage1_formal_gate_hn_x_crosscheck", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
