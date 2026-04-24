"""
machine_B.py — v3 Stage 1 BINARY gate capacity scan (l, x).

Human workflow on Machine B:
    cd <repo-root>
    uv run python scripts/machines/machine_B.py

Chain:
    1. git pull
    2. build 2-class hardlink view  (data/sewerml_gate_v3_stage1_binary/)
    3. train yolo11l  (~5-6 h)
    4. train yolo11x  (~7-9 h)

Output: research/materials/v3_stage1_binary/yolo11{l,x}/
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sewerml_gate_v3_stage1_binary"
OUT_ROOT = REPO_ROOT / "research" / "materials" / "v3_stage1_binary"
CAPACITIES = ["l", "x"]


def run(cmd):
    print(f"\n>>> {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        sys.exit(f"[Machine B] FAILED at: {cmd}")


def main():
    run(["git", "pull"])
    run(["uv", "run", "python", "scripts/build_v3_stage1_binary_view.py"])
    for cap in CAPACITIES:
        run([
            "uv", "run", "python", "scripts/train_v3_stage1.py",
            "--capacity", cap,
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(OUT_ROOT / f"yolo11{cap}"),
        ])
    print(f"\n[Machine B] ALL DONE — {OUT_ROOT}")


if __name__ == "__main__":
    main()
