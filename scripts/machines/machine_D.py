"""
machine_D.py — Machine D pipeline launcher.

Runs capacity scan for yolo11x (extra-large capacity, ~57M params).

Note: yolo11x-cls.pt is the largest variant; requires more VRAM. If batch=24
fails with OOM, Machine D's AI should reduce batch to 16 or 12 via --batch arg
(consistency with other capacities is preferred but 16 is still within shared
protocol tolerance).

Produces:
    runs/yolo11x/weights/epoch*.pt
    runs/yolo11x/per_epoch_metrics.csv
    runs/yolo11x/best_epoch.json
    runs/yolo11x/final_test_metrics.json

Usage:
    uv run python scripts/machines/machine_D.py \\
        --data-dir /path/to/sewerml_gate_v3_stage1 \\
        --output-dir ./runs

After completion, zip `./runs/yolo11x` and upload to shared storage.
"""
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_v3_stage1.py"

CAPACITIES = ["x"]


def run(capacity, data_dir, output_dir, batch, smoke):
    out = output_dir / f"yolo11{capacity}"
    cmd = [
        "uv", "run", "python", str(TRAIN_SCRIPT),
        "--capacity", capacity,
        "--data-dir", str(data_dir),
        "--output-dir", str(out),
        "--batch", str(batch),
    ]
    if smoke:
        cmd.append("--smoke")
    print(f"\n{'=' * 70}\n[Machine D] capacity={capacity} batch={batch}\n{'=' * 70}")
    print(f"[cmd] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"capacity {capacity} training failed with code {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("./runs"))
    ap.add_argument("--batch", type=int, default=24, help="batch size (default 24; lower if OOM)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cap in CAPACITIES:
        run(cap, args.data_dir.resolve(), args.output_dir.resolve(), args.batch, args.smoke)

    print(f"\n{'=' * 70}\n[Machine D] ALL DONE. Zip and upload:\n{'=' * 70}")
    for cap in CAPACITIES:
        print(f"  {args.output_dir.resolve() / ('yolo11' + cap)}")


if __name__ == "__main__":
    main()
