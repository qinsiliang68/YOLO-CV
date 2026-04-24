"""
machine_A.py — v3 Stage 1 BINARY gate capacity scan (n, s, m).

Human workflow on Machine A:
    cd <repo-root>
    uv run python scripts/machines/machine_A.py

Chain:
    1. git pull
    2. build 2-class hardlink view  (data/sewerml_gate_v3_stage1_binary/)
    3. train yolo11n  (~2-3 h)
    4. train yolo11s  (~3-4 h)
    5. train yolo11m  (~4-5 h)

Output: research/materials/v3_stage1_binary/yolo11{n,s,m}/
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sewerml_gate_v3_stage1_binary"
OUT_ROOT = REPO_ROOT / "research" / "materials" / "v3_stage1_binary"
CAPACITIES = ["n", "s", "m"]


def run(cmd):
    print(f"\n>>> {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        sys.exit(f"[Machine A] FAILED at: {cmd}")


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
    print(f"\n[Machine A] ALL DONE — {OUT_ROOT}")


if __name__ == "__main__":
    main()
