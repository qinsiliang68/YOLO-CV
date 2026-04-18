"""
machine_A.py — Machine A pipeline launcher.

Runs capacity scan for yolo11n AND yolo11s sequentially (both small capacities).

Produces:
    runs/yolo11n/weights/epoch*.pt, per_epoch_metrics.csv, best_epoch.json, final_test_metrics.json
    runs/yolo11s/weights/epoch*.pt, per_epoch_metrics.csv, best_epoch.json, final_test_metrics.json

Usage (Machine A's local AI only needs to set paths and run this):
    # 1. pull repo
    git pull

    # 2. download dataset package (sewerml_gate_v3_stage1) from shared storage
    #    expected: $DATA_DIR/sewerml_gate_v3_stage1/{train,val_cal,val_op,test}/...

    # 3. run (adjust --data-dir / --output-dir if needed)
    uv run python scripts/machines/machine_A.py \\
        --data-dir /path/to/sewerml_gate_v3_stage1 \\
        --output-dir ./runs

After completion, zip `./runs/yolo11n` and `./runs/yolo11s` and upload to shared
storage. That's the only manual action required.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_v3_stage1.py"

CAPACITIES = ["n", "s"]


def run(capacity: str, data_dir: Path, output_dir: Path, smoke: bool):
    out = output_dir / f"yolo11{capacity}"
    cmd = [
        "uv", "run", "python", str(TRAIN_SCRIPT),
        "--capacity", capacity,
        "--data-dir", str(data_dir),
        "--output-dir", str(out),
    ]
    if smoke:
        cmd.append("--smoke")
    print(f"\n{'=' * 70}\n[Machine A] capacity={capacity}\n{'=' * 70}")
    print(f"[cmd] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"capacity {capacity} training failed with code {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Path to sewerml_gate_v3_stage1 (has train/val_cal/val_op/test inside)")
    ap.add_argument("--output-dir", type=Path, default=Path("./runs"))
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test mode (3 epochs, batch 4)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cap in CAPACITIES:
        run(cap, args.data_dir.resolve(), args.output_dir.resolve(), args.smoke)

    print(f"\n{'=' * 70}\n[Machine A] ALL DONE. Output in {args.output_dir.resolve()}\n{'=' * 70}")
    print("Zip and upload:")
    for cap in CAPACITIES:
        print(f"  {args.output_dir.resolve() / ('yolo11' + cap)}")


if __name__ == "__main__":
    main()
