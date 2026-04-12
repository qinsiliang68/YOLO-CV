"""
Phase 2 Pipeline: 定峰 → 定效 → Robustness
Self-contained. Reads candidate_pool_master.csv, builds dataset views,
trains via ultralytics, evaluates via gate protocol. No complex config system.

Usage:
    python scripts/run_phase2_pipeline.py --phase all --device 0
    python scripts/run_phase2_pipeline.py --phase peak --device 0
    python scripts/run_phase2_pipeline.py --phase validate --device 0
    python scripts/run_phase2_pipeline.py --phase robustness --device 0
    python scripts/run_phase2_pipeline.py --list   # show all experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# ── paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

DATASET_SOURCE = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_gate2_train7200"
SPLIT_CSV = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv"
POOL_MASTER = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "candidate_pool_master.csv"
TEACHER_MANIFEST = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "best_epoch_manifest.json"

PHASE2_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_phase2"
PHASE2_RESULTS = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_phase2"
PHASE2_DATASETS = REPO_ROOT / "YOLOv11" / "datasets" / "stage1_phase2"
PHASE2_RUNS = REPO_ROOT / "YOLOv11" / "runs" / "stage1_phase2"

SHARED_PROTOCOL = dict(epochs=200, imgsz=640, batch=24, workers=4,
                       optimizer="auto", patience=0, save_period=1,
                       pretrained=True, cache=False)


def load_pool() -> list[dict]:
    with open(POOL_MASTER, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_teacher_checkpoint() -> str:
    manifest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))
    cp = manifest["checkpoint_path"]
    p = Path(cp)
    if p.exists():
        return str(p.resolve())
    # try rebase to local repo
    for marker in ("YOLO-CV/", "YOLOv11/"):
        idx = cp.replace("\\", "/").find(marker)
        if idx >= 0:
            local = REPO_ROOT / cp.replace("\\", "/")[idx + len("YOLO-CV/"):]
            if local.exists():
                return str(local.resolve())
    return cp


def sort_pool_by_signal(pool: list[dict], signal: str, descending: bool = True) -> list[dict]:
    return sorted(pool, key=lambda r: float(r[signal]), reverse=descending)


def select_bucket(pool_sorted: list[dict], rank_start: int, rank_end: int) -> list[dict]:
    """Select samples by rank (1-indexed, inclusive)."""
    return pool_sorted[rank_start - 1:rank_end]


def build_dataset_view(
    experiment_id: str,
    selected_ids: list[str],
    replay_count: int,
) -> Path:
    """Build a training dataset with selected samples replayed to replay_count."""
    ds_root = PHASE2_DATASETS / experiment_id
    train_normal = ds_root / "train" / "Normal"
    train_abnormal = ds_root / "train" / "Abnormal"
    val_normal = ds_root / "val" / "Normal"
    val_abnormal = ds_root / "val" / "Abnormal"

    if (ds_root / "_built_marker.json").exists():
        print(f"    dataset view exists, skipping: {ds_root}")
        return ds_root

    for d in [train_normal, train_abnormal, val_normal, val_abnormal]:
        d.mkdir(parents=True, exist_ok=True)

    src_train_abnormal = DATASET_SOURCE / "train" / "Abnormal"
    src_train_normal = DATASET_SOURCE / "train" / "Normal"
    src_val_abnormal = DATASET_SOURCE / "val" / "Abnormal"
    src_val_normal = DATASET_SOURCE / "val" / "Normal"

    # link abnormal training data (unchanged)
    for img in src_train_abnormal.iterdir():
        dst = train_abnormal / img.name
        if not dst.exists():
            try:
                os.link(str(img), str(dst))
            except OSError:
                shutil.copy2(str(img), str(dst))

    # link original normal training data
    for img in src_train_normal.iterdir():
        dst = train_normal / img.name
        if not dst.exists():
            try:
                os.link(str(img), str(dst))
            except OSError:
                shutil.copy2(str(img), str(dst))

    # add replayed copies of selected samples
    n_unique = len(selected_ids)
    if n_unique == 0:
        raise ValueError(f"No samples selected for {experiment_id}")

    # build image_id → actual filename mapping from pool master
    id_to_filename = {}
    pool_rows = load_pool()
    for row in pool_rows:
        img_id = row["image_id"]
        rel_path = row.get("img_rel_path", "")
        if rel_path:
            id_to_filename[img_id] = Path(rel_path).name
        else:
            # fallback: strip "Normal_" prefix
            numeric = img_id.replace("Normal_", "")
            id_to_filename[img_id] = f"{numeric}.png"

    copies_needed = replay_count
    for copy_idx in range(copies_needed):
        src_id = selected_ids[copy_idx % n_unique]
        filename = id_to_filename.get(src_id, f"{src_id}.png")
        src_file = src_train_normal / filename
        if not src_file.exists():
            alt = list(src_train_normal.glob(f"*{src_id.replace('Normal_', '')}*"))
            if alt:
                src_file = alt[0]
            else:
                print(f"    WARNING: {src_id} ({filename}) not found in training normal dir")
                continue
        dst_name = f"_replay_{experiment_id}_{copy_idx:04d}_{src_file.name}"
        dst = train_normal / dst_name
        if not dst.exists():
            try:
                os.link(str(src_file), str(dst))
            except OSError:
                shutil.copy2(str(src_file), str(dst))

    # link val data
    for img in src_val_abnormal.iterdir():
        dst = val_abnormal / img.name
        if not dst.exists():
            try:
                os.link(str(img), str(dst))
            except OSError:
                shutil.copy2(str(img), str(dst))
    for img in src_val_normal.iterdir():
        dst = val_normal / img.name
        if not dst.exists():
            try:
                os.link(str(img), str(dst))
            except OSError:
                shutil.copy2(str(img), str(dst))

    marker = {
        "experiment_id": experiment_id,
        "n_unique": n_unique,
        "replay_count": replay_count,
        "selected_ids": selected_ids[:10],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ds_root / "_built_marker.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"    dataset view built: {ds_root} ({n_unique} unique, {replay_count} replay)")
    return ds_root


def run_training(experiment_id: str, dataset_root: Path, teacher_ckpt: str,
                 device: str, seed: int = 0) -> Path:
    """Train using ultralytics YOLO API."""
    run_dir = PHASE2_RUNS / experiment_id
    if (run_dir / "weights" / "last.pt").exists():
        print(f"    training already complete, skipping: {run_dir}")
        return run_dir

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(teacher_ckpt)
    model.train(
        data=str(dataset_root),
        epochs=SHARED_PROTOCOL["epochs"],
        imgsz=SHARED_PROTOCOL["imgsz"],
        batch=SHARED_PROTOCOL["batch"],
        workers=SHARED_PROTOCOL["workers"],
        optimizer=SHARED_PROTOCOL["optimizer"],
        patience=SHARED_PROTOCOL["patience"],
        save_period=SHARED_PROTOCOL["save_period"],
        pretrained=SHARED_PROTOCOL["pretrained"],
        cache=SHARED_PROTOCOL["cache"],
        device=device,
        project=str(PHASE2_RUNS),
        name=experiment_id,
        exist_ok=True,
        seed=seed,
    )
    print(f"    training complete: {run_dir}")
    return run_dir


def run_gate_eval(experiment_id: str, run_dir: Path) -> dict[str, Any]:
    """Run gate evaluation on all checkpoints."""
    summary_dir = PHASE2_ROOT / experiment_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    if (summary_dir / "best_epoch_manifest.json").exists():
        manifest = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    gate eval already complete: Spec@R99.5={manifest.get('spec_at_r995', '?')}")
        return manifest

    # call existing evaluator
    eval_script = SCRIPTS_DIR / "stage1_formal_gate_epoch_eval.py"
    eval_config = {
        "run_dir": str(run_dir),
        "split_csv": str(SPLIT_CSV),
        "summary_dir": str(summary_dir),
        "produce_per_epoch_gate": True,
        "produce_dashboard": False,
    }
    config_path = summary_dir / "_eval_config.json"
    config_path.write_text(json.dumps(eval_config, indent=2), encoding="utf-8")

    import subprocess
    result = subprocess.run(
        [sys.executable, str(eval_script), "--config", str(config_path)],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"    WARNING: gate eval failed for {experiment_id}")
        return {}

    if (summary_dir / "best_epoch_manifest.json").exists():
        manifest = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    gate eval complete: Spec@R99.5={manifest.get('spec_at_r995', '?')}")
        return manifest
    return {}


# ══════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════

def define_experiments() -> list[dict[str, Any]]:
    experiments = []

    # ── Phase 1: 定峰 (R/C/D 二分 + Training dynamics Q1-Q5) ──

    # R-Q2 二分: Q2 = rank 51-100, split into Q2a(51-75) and Q2b(76-100)
    experiments.append({"id": "p1_R_Q2a", "phase": "peak", "signal": "R",
                        "rank_start": 51, "rank_end": 75, "replay": 151, "seed": 0})
    experiments.append({"id": "p1_R_Q2b", "phase": "peak", "signal": "R",
                        "rank_start": 76, "rank_end": 100, "replay": 151, "seed": 0})

    # D-Q3 二分: Q3 = rank 101-150
    experiments.append({"id": "p1_D_Q3a", "phase": "peak", "signal": "D",
                        "rank_start": 101, "rank_end": 125, "replay": 151, "seed": 0})
    experiments.append({"id": "p1_D_Q3b", "phase": "peak", "signal": "D",
                        "rank_start": 126, "rank_end": 150, "replay": 151, "seed": 0})

    # C-Q4 二分: Q4 = rank 151-200
    experiments.append({"id": "p1_C_Q4a", "phase": "peak", "signal": "C",
                        "rank_start": 151, "rank_end": 175, "replay": 151, "seed": 0})
    experiments.append({"id": "p1_C_Q4b", "phase": "peak", "signal": "C",
                        "rank_start": 176, "rank_end": 200, "replay": 151, "seed": 0})

    # Training dynamics Q1-Q5 (signal="T", requires pre-extraction)
    for qi in range(5):
        experiments.append({
            "id": f"p1_T_Q{qi+1}", "phase": "peak", "signal": "T",
            "rank_start": qi * 50 + 1, "rank_end": (qi + 1) * 50,
            "replay": 151, "seed": 0,
        })

    # ── Phase 2: 定效 ──
    experiments.append({"id": "p2_closedloop", "phase": "validate",
                        "signal": "GOLDILOCKS", "rank_start": 0, "rank_end": 0,
                        "replay": 151, "seed": 0})
    for seed_val in [2, 3, 4]:
        experiments.append({
            "id": f"p2_seed{seed_val}_R_Q2", "phase": "validate",
            "signal": "R", "rank_start": 51, "rank_end": 100,
            "replay": 151, "seed": seed_val,
        })

    # ── Phase 3: Robustness ──
    for qi in range(5):
        experiments.append({
            "id": f"p3_top1_R_Q{qi+1}", "phase": "robustness",
            "signal": "R_top1", "rank_start": qi * 50 + 1, "rank_end": (qi + 1) * 50,
            "replay": 151, "seed": 0,
        })
    experiments.append({"id": "p3_top1_G0", "phase": "robustness",
                        "signal": "G0_top1", "rank_start": 1, "rank_end": 151,
                        "replay": 151, "seed": 0})

    return experiments


def extract_training_dynamics(teacher_ckpt: str, pool: list[dict]) -> dict[str, float]:
    """
    Extract per-sample training dynamics (variability) from hn00 teacher's
    per-epoch predictions. Returns {image_id: variability_score}.

    If pre-computed file exists, load from disk.
    Otherwise, compute from epoch_gate_summary data.
    """
    cache_path = PHASE2_ROOT / "training_dynamics_variability.json"
    if cache_path.exists():
        print("    loading cached training dynamics scores")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("    extracting training dynamics from per-epoch predictions...")
    print("    (this requires per-epoch val_op_predictions_calibrated.csv from hn00 training)")

    hn00_per_epoch = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "per_epoch_gate"
    if not hn00_per_epoch.exists():
        print(f"    WARNING: {hn00_per_epoch} not found.")
        print(f"    Training dynamics signal requires per-epoch predictions from hn00.")
        print(f"    Falling back to R signal as proxy for T signal.")
        # fallback: use R as proxy
        variability = {r["image_id"]: float(r["R"]) for r in pool}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(variability, indent=2), encoding="utf-8")
        return variability

    import numpy as np

    # collect per-sample predictions across epochs
    pool_ids = {r["image_id"] for r in pool}
    sample_scores: dict[str, list[float]] = {sid: [] for sid in pool_ids}

    epoch_dirs = sorted(hn00_per_epoch.iterdir(), key=lambda d: int(d.name.split("_")[-1]) if d.is_dir() else 0)
    for epoch_dir in epoch_dirs:
        if not epoch_dir.is_dir():
            continue
        pred_csv = epoch_dir / "val_op_predictions_calibrated.csv"
        if not pred_csv.exists():
            continue
        with open(pred_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                img_id = row.get("image_id", "")
                if img_id in sample_scores:
                    score = float(row.get("calibrated_score", row.get("score", 0)))
                    sample_scores[img_id].append(score)

    # compute variability (std across epochs)
    variability = {}
    for img_id, scores in sample_scores.items():
        if len(scores) >= 10:
            variability[img_id] = float(np.std(scores))
        else:
            variability[img_id] = 0.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(variability, indent=2), encoding="utf-8")
    print(f"    training dynamics extracted for {len(variability)} samples")
    return variability


def run_experiment(exp: dict[str, Any], pool: list[dict], teacher_ckpt: str,
                   device: str, training_dynamics: dict[str, float] | None = None) -> dict[str, Any]:
    """Run a single experiment end-to-end."""
    exp_id = exp["id"]
    signal = exp["signal"]
    rank_start = exp["rank_start"]
    rank_end = exp["rank_end"]
    replay = exp["replay"]
    seed = exp["seed"]

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_id} (signal={signal}, rank={rank_start}-{rank_end}, seed={seed})")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── select samples ──
    if signal == "GOLDILOCKS":
        # closed-loop: use Goldilocks function (to be defined after Phase 1)
        print("    GOLDILOCKS closedloop: selecting based on Phase 1 peak parameters")
        # For now, use R-Q2 as proxy until Phase 1 results define the function
        pool_sorted = sort_pool_by_signal(pool, "R")
        selected = select_bucket(pool_sorted, 51, 100)
    elif signal == "T":
        if training_dynamics is None:
            training_dynamics = extract_training_dynamics(teacher_ckpt, pool)
        # sort by variability descending (most variable = Q1)
        pool_with_t = []
        for r in pool:
            r_copy = dict(r)
            r_copy["T"] = training_dynamics.get(r["image_id"], 0.0)
            pool_with_t.append(r_copy)
        pool_sorted = sort_pool_by_signal(pool_with_t, "T")
        selected = select_bucket(pool_sorted, rank_start, rank_end)
    elif signal.startswith("R_top1") or signal.startswith("G0_top1"):
        # Robustness: use top1-best teacher instead of gate-best
        # Need to recompute R scores with top1 teacher - for now use existing R
        print("    NOTE: top1 teacher R scores need separate computation.")
        print("    Using gate-teacher R scores as placeholder until top1 checkpoint is available.")
        pool_sorted = sort_pool_by_signal(pool, "R")
        if signal == "G0_top1":
            import random
            random.seed(42)
            indices = random.sample(range(len(pool)), min(replay, len(pool)))
            selected = [pool[i] for i in indices]
        else:
            selected = select_bucket(pool_sorted, rank_start, rank_end)
    else:
        pool_sorted = sort_pool_by_signal(pool, signal)
        selected = select_bucket(pool_sorted, rank_start, rank_end)

    selected_ids = [r["image_id"] for r in selected]
    print(f"    selected {len(selected_ids)} unique samples")

    # ── build dataset ──
    ds_root = build_dataset_view(exp_id, selected_ids, replay)

    # ── train ──
    t0 = time.time()
    run_dir = run_training(exp_id, ds_root, teacher_ckpt, device, seed)
    train_time = time.time() - t0

    # ── evaluate ──
    manifest = run_gate_eval(exp_id, run_dir)

    result = {
        "experiment_id": exp_id,
        "signal": signal,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "n_unique": len(selected_ids),
        "replay": replay,
        "seed": seed,
        "spec_at_r995": manifest.get("spec_at_r995", None),
        "spec_at_r990": manifest.get("spec_at_r990", None),
        "best_epoch": manifest.get("epoch", None),
        "train_hours": round(train_time / 3600, 2),
    }
    # save individual result
    result_path = PHASE2_RESULTS / f"{exp_id}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 experiment pipeline.")
    parser.add_argument("--phase", default="all",
                        help="Phase to run: all, peak, validate, robustness")
    parser.add_argument("--device", default="0", help="GPU device (default: 0)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--experiment", default="", help="Run single experiment by ID.")
    args = parser.parse_args()

    experiments = define_experiments()

    if args.list:
        print(f"\nPhase 2 Pipeline — {len(experiments)} experiments\n")
        for exp in experiments:
            print(f"  [{exp['phase']:12s}] {exp['id']:20s}  signal={exp['signal']:10s}  "
                  f"rank={exp['rank_start']:3d}-{exp['rank_end']:3d}  seed={exp['seed']}")
        return

    if args.experiment:
        experiments = [e for e in experiments if e["id"] == args.experiment]
        if not experiments:
            raise SystemExit(f"Experiment not found: {args.experiment}")

    if args.phase != "all":
        experiments = [e for e in experiments if e["phase"] == args.phase]

    print(f"\nPhase 2 Pipeline")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Device: {args.device}")
    print(f"  Estimated time: ~{len(experiments) * 3.5:.0f} hours")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # preflight checks
    for path, name in [(DATASET_SOURCE, "dataset"), (SPLIT_CSV, "split_csv"),
                        (POOL_MASTER, "pool_master"), (TEACHER_MANIFEST, "teacher_manifest")]:
        if not path.exists():
            raise SystemExit(f"Missing: {name} at {path}")
    print("  Preflight: all paths OK\n")

    pool = load_pool()
    teacher_ckpt = load_teacher_checkpoint()
    print(f"  Pool: {len(pool)} samples")
    print(f"  Teacher: {teacher_ckpt}\n")

    # pre-extract training dynamics if needed
    training_dynamics = None
    if any(e["signal"] == "T" for e in experiments):
        training_dynamics = extract_training_dynamics(teacher_ckpt, pool)

    # run experiments
    t_start = time.time()
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n  [{i+1}/{len(experiments)}]")
        try:
            result = run_experiment(exp, pool, teacher_ckpt, args.device, training_dynamics)
            results.append(result)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({"experiment_id": exp["id"], "error": str(exc)})

    # summary
    total_hours = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Total: {total_hours:.1f} hours")
    print(f"{'='*60}\n")

    summary_path = PHASE2_RESULTS / "phase2_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment_id", "signal", "rank_start", "rank_end",
                  "n_unique", "replay", "seed", "spec_at_r995", "spec_at_r990",
                  "best_epoch", "train_hours"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"  Summary: {summary_path}")
    for r in results:
        spec = r.get("spec_at_r995", "?")
        status = "OK" if "error" not in r else "FAILED"
        print(f"  [{status}] {r['experiment_id']:20s}  Spec@R99.5={spec}")


if __name__ == "__main__":
    main()
