"""
machine_C.py — Machine C pipeline launcher.

Runs capacity scan for yolo11l (large capacity, ~42M params).

Produces:
    runs/yolo11l/weights/epoch*.pt
    runs/yolo11l/per_epoch_metrics.csv
    runs/yolo11l/best_epoch.json
    runs/yolo11l/final_test_metrics.json

Usage:
    uv run python scripts/machines/machine_C.py \\
        --data-dir /path/to/sewerml_gate_v3_stage1 \\
        --output-dir ./runs

After completion, zip `./runs/yolo11l` and upload to shared storage.
"""
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_v3_stage1.py"

CAPACITIES = ["l"]


def run(capacity, data_dir, output_dir, smoke):
    out = output_dir / f"yolo11{capacity}"
    cmd = [
        "uv", "run", "python", str(TRAIN_SCRIPT),
        "--capacity", capacity,
        "--data-dir", str(data_dir),
        "--output-dir", str(out),
    ]
    if smoke:
        cmd.append("--smoke")
    print(f"\n{'=' * 70}\n[Machine C] capacity={capacity}\n{'=' * 70}")
    print(f"[cmd] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"capacity {capacity} training failed with code {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("./runs"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cap in CAPACITIES:
        run(cap, args.data_dir.resolve(), args.output_dir.resolve(), args.smoke)

    print(f"\n{'=' * 70}\n[Machine C] ALL DONE. Zip and upload:\n{'=' * 70}")
    for cap in CAPACITIES:
        print(f"  {args.output_dir.resolve() / ('yolo11' + cap)}")


if __name__ == "__main__":
    main()
