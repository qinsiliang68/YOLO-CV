"""Multi-seed validation: m+hn00 / m+hn02 / m+hn14 × 2 new seeds.

Verifies that the HN sweep conclusions (hn14 > hn00, hn02 ≈ hn14) are
not artifacts of a single random seed.  All 6 runs train from pretrained
yolo11m-cls (no dependency on ASUS checkpoint), reuse existing HN
datasets (teacher scoring is seed-invariant), and run the full gate
evaluation pipeline (temperature calibration + recall-constrained
threshold search).

Design note: the original sweep trains hn02-hn20 from hn00's best
checkpoint (epoch_078.pt).  This script trains all runs from pretrained
yolo11m-cls to eliminate cross-seed dependency and keep the comparison
self-contained.  The experimental question is: "given identical training
budget (200 epochs from pretrained), does HN-augmented data consistently
outperform the base dataset across random seeds?"

Usage:
    uv run python main_multiseed_validation.py          # full run
    uv run python main_multiseed_validation.py --dry-run # plan only
    uv run python main_multiseed_validation.py --device 1  # use GPU 1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS = REPO_ROOT / "scripts"
YOLOV11_ROOT = REPO_ROOT / "YOLOv11"

# ── Seeds: original = 20260330, two new ones ──────────────────────────
SEEDS = [20260401, 20260402]

# ── Three key comparison points ───────────────────────────────────────
RUNS_SPEC = [
    {"ratio": "hn00", "dataset": "YOLOv11/datasets/sewerml_gate2_train7200"},
    {"ratio": "hn02", "dataset": "YOLOv11/datasets/stage1_formal_gate_hn/yolo11m_gate2_formal_hn02"},
    {"ratio": "hn14", "dataset": "YOLOv11/datasets/stage1_formal_gate_hn/yolo11m_gate2_formal_hn14"},
]

# ── Shared training protocol (identical to original sweep) ────────────
TRAIN_PROTOCOL = {
    "task": "classify",
    "model": "yolo11m-cls",
    "epochs": 200,
    "imgsz": 640,
    "batch": 24,
    "workers": 4,
    "pretrained": True,
    "patience": 0,
    "optimizer": "auto",
    "cache": False,
    "save_period": 1,
    "exist_ok": True,
    "resume": False,
}

SPLIT_CSV = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv"
PROJECT_ROOT = YOLOV11_ROOT / "runs" / "stage1_multiseed"
MATERIALS_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_multiseed"
RESULTS_ROOT = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_multiseed"


def seed_tag(seed: int) -> str:
    return f"s{seed % 10000:04d}"


def run_name(ratio: str, seed: int) -> str:
    return f"yolo11m_gate2_ms_{seed_tag(seed)}_{ratio}_200ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="0", help="GPU device (default: 0)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run evaluation on existing checkpoints")
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one(
    ratio: str,
    dataset_rel: str,
    seed: int,
    device: str,
    dry_run: bool,
    skip_train: bool,
) -> dict[str, Any] | None:
    """Train + evaluate one (ratio, seed) combination."""

    tag = seed_tag(seed)
    name = run_name(ratio, seed)
    dataset_root = REPO_ROOT / dataset_rel
    run_dir = PROJECT_ROOT / name
    summary_dir = MATERIALS_ROOT / tag / ratio

    print(f"\n{'='*60}")
    print(f"  [{tag}] {ratio}: seed={seed}")
    print(f"  dataset: {dataset_root}")
    print(f"  run_dir: {run_dir}")
    print(f"  summary: {summary_dir}")
    print(f"{'='*60}\n")

    # ── Preflight ──
    if not dataset_root.exists():
        print(f"  ERROR: dataset not found: {dataset_root}")
        print(f"  This HN dataset must exist on this machine.")
        return None

    if not SPLIT_CSV.exists():
        print(f"  ERROR: split CSV not found: {SPLIT_CSV}")
        return None

    # ── 1. Train ──
    if not skip_train:
        train_cfg = dict(TRAIN_PROTOCOL)
        train_cfg["data"] = str(dataset_root)
        train_cfg["project"] = str(PROJECT_ROOT)
        train_cfg["name"] = name
        train_cfg["seed"] = seed
        train_cfg["device"] = device

        config_dir = REPO_ROOT / "$out" / "generated_configs" / "stage1_multiseed"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{name}.json"
        write_json(config_path, train_cfg)

        if dry_run:
            print(f"  [DRY-RUN] would train: {config_path}")
        else:
            print(f"  [TRAIN] starting {name} ...")
            cmd = [
                sys.executable,
                str(SCRIPTS / "stage1_gate_train.py"),
                "--config", str(config_path),
                "--stdout-log", str(run_dir / "stdout.log"),
                "--stderr-log", str(run_dir / "stderr.log"),
            ]
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if result.returncode != 0:
                print(f"  ERROR: training failed with code {result.returncode}")
                return None
            print(f"  [TRAIN] done: {name}")

    # ── 2. Evaluate all epochs ──
    if dry_run:
        print(f"  [DRY-RUN] would evaluate: {run_dir}")
        return {"ratio": ratio, "seed": seed, "tag": tag, "status": "dry-run"}

    if not run_dir.exists():
        print(f"  ERROR: run_dir not found: {run_dir}")
        return None

    weights_dir = run_dir / "weights"
    if not weights_dir.exists() or not list(weights_dir.glob("epoch*.pt")):
        print(f"  ERROR: no epoch checkpoints in {weights_dir}")
        return None

    print(f"  [EVAL] evaluating all epochs for {name} ...")
    cmd = [
        sys.executable,
        str(SCRIPTS / "stage1_formal_gate_epoch_eval.py"),
        "--run-dir", str(run_dir),
        "--data-root", str(dataset_root),
        "--summary-dir", str(summary_dir),
        "--split-csv", str(SPLIT_CSV),
        "--device", device,
        "--batch", "24",
        "--imgsz", "640",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: evaluation failed with code {result.returncode}")
        return None
    print(f"  [EVAL] done: {name}")

    # ── 3. Read best result ──
    best_path = summary_dir / "best_epoch_manifest.json"
    if not best_path.exists():
        print(f"  WARNING: best_epoch_manifest.json not found")
        return {"ratio": ratio, "seed": seed, "tag": tag, "status": "eval-no-best"}

    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)

    return {
        "ratio": ratio,
        "seed": seed,
        "tag": tag,
        "status": "ok",
        "best_epoch": int(best["epoch"]),
        "spec_at_r995": float(best["spec_at_r995"]),
        "spec_at_r990": float(best["spec_at_r990"]),
        "prec_at_r990": float(best["prec_at_r990"]),
        "ptr_at_r990": float(best["ptr_at_r990"]),
        "tn_at_r995": int(best["tn_at_r995"]),
        "temperature_T": float(best["temperature_T"]),
    }


def load_original_results() -> list[dict[str, Any]]:
    """Load seed-1 results from the original sweep summary."""
    orig_csv = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep" / "hn_sweep_summary.csv"
    results = []
    if not orig_csv.exists():
        return results
    with orig_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("ratio_id", "")
            if rid in ("hn00", "hn02", "hn14"):
                results.append({
                    "ratio": rid,
                    "seed": 20260330,
                    "tag": "s0330",
                    "status": "ok",
                    "best_epoch": int(row["best_epoch"]),
                    "spec_at_r995": float(row["spec_at_r995"]),
                    "spec_at_r990": float(row["spec_at_r990"]),
                    "prec_at_r990": float(row["prec_at_r990"]),
                    "ptr_at_r990": float(row["ptr_at_r990"]),
                    "tn_at_r995": int(row["tn_at_r995"]),
                    "temperature_T": float(row["temperature_T"]),
                })
    return results


def print_comparison(all_results: list[dict[str, Any]]) -> None:
    """Print the multi-seed comparison table."""
    print(f"\n{'='*80}")
    print("MULTI-SEED VALIDATION RESULTS")
    print(f"{'='*80}\n")

    header = (
        f"{'Seed':>8} {'Ratio':>6} {'BestEp':>7} "
        f"{'Spec@R995':>10} {'Spec@R990':>10} {'TN@R995':>8} {'T':>6}"
    )
    print(header)
    print("-" * len(header))

    for ratio in ("hn00", "hn02", "hn14"):
        group = [r for r in all_results if r["ratio"] == ratio and r["status"] == "ok"]
        for r in sorted(group, key=lambda x: x["seed"]):
            marker = " *" if r["seed"] != 20260330 else ""
            print(
                f"{r['seed']:>8} {r['ratio']:>6} {r['best_epoch']:>7} "
                f"{r['spec_at_r995']:>10.4f} {r['spec_at_r990']:>10.4f} "
                f"{r['tn_at_r995']:>8} {r['temperature_T']:>6.2f}{marker}"
            )
        if group:
            specs = [r["spec_at_r995"] for r in group]
            print(
                f"{'':>8} {'':>6} {'mean':>7} "
                f"{sum(specs)/len(specs):>10.4f} {'':>10} "
                f"{'':>8} {'':>6}  [{min(specs):.4f}, {max(specs):.4f}]"
            )
        print()

    # ── Key question: does hn14 beat hn00 in every seed? ──
    print("KEY QUESTION: does hn14 > hn00 in every seed?")
    for seed in [20260330] + SEEDS:
        hn00 = next((r for r in all_results if r["ratio"] == "hn00" and r["seed"] == seed and r["status"] == "ok"), None)
        hn14 = next((r for r in all_results if r["ratio"] == "hn14" and r["seed"] == seed and r["status"] == "ok"), None)
        if hn00 and hn14:
            delta = hn14["spec_at_r995"] - hn00["spec_at_r995"]
            verdict = "YES" if delta > 0 else ("TIE" if delta == 0 else "NO")
            print(f"  seed {seed}: hn14={hn14['spec_at_r995']:.4f} - hn00={hn00['spec_at_r995']:.4f} = {delta:+.4f}  [{verdict}]")
        else:
            print(f"  seed {seed}: incomplete data")


def save_results(all_results: list[dict[str, Any]]) -> None:
    """Save comparison to CSV."""
    out_dir = RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_results = [r for r in all_results if r["status"] == "ok"]
    if not ok_results:
        return

    fieldnames = ["seed", "tag", "ratio", "best_epoch", "spec_at_r995",
                  "spec_at_r990", "prec_at_r990", "ptr_at_r990",
                  "tn_at_r995", "temperature_T"]
    csv_path = out_dir / "multiseed_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(ok_results, key=lambda x: (x["ratio"], x["seed"])):
            writer.writerow({k: r[k] for k in fieldnames})

    write_json(out_dir / "multiseed_comparison.json", ok_results)
    print(f"\nResults saved to: {csv_path}")


def main() -> None:
    args = parse_args()

    print("Multi-seed validation: yolo11m-cls gate HN")
    print(f"  Seeds: {SEEDS}")
    print(f"  Ratios: {[r['ratio'] for r in RUNS_SPEC]}")
    print(f"  Device: {args.device}")
    print(f"  Protocol: {TRAIN_PROTOCOL['epochs']} epochs, batch={TRAIN_PROTOCOL['batch']}, "
          f"imgsz={TRAIN_PROTOCOL['imgsz']}, model={TRAIN_PROTOCOL['model']}")
    print()

    # ── Preflight: check datasets exist ──
    missing = []
    for spec in RUNS_SPEC:
        ds = REPO_ROOT / spec["dataset"]
        if not ds.exists():
            missing.append(str(ds))
    if missing:
        print("ERROR: missing datasets:")
        for m in missing:
            print(f"  {m}")
        print("\nHN datasets (hn02/hn14) must exist on this machine.")
        print("They are seed-invariant — copy from any machine that has them.")
        sys.exit(1)

    # ── Run all 6 jobs ──
    new_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for spec in RUNS_SPEC:
            result = run_one(
                ratio=spec["ratio"],
                dataset_rel=spec["dataset"],
                seed=seed,
                device=args.device,
                dry_run=args.dry_run,
                skip_train=args.skip_train,
            )
            if result:
                new_results.append(result)

    if args.dry_run:
        print("\n[DRY-RUN] 6 runs planned. Remove --dry-run to execute.")
        return

    # ── Comparison ──
    all_results = load_original_results() + new_results
    print_comparison(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
