"""
Phase 2 Sliding-Window Pipeline: 精细定峰

对 R/D/T/C 四个信号各做滑动窗口扫描:
  - 窗口宽度: 50 样本
  - 步长: 10 样本
  - 250 个候选样本 → 21 个窗口位置 (rank 1-50, 11-60, ..., 201-250)
  - 每个窗口: 50 unique samples replay 到 151 张, 训练 200 epoch, gate 评估
  - 输出: 信号值 → 门控性能的连续响应曲线

4 信号 × 21 窗口 = 84 run
每 run ~2.6h, 单机 ~218h, 4 机并行 ~55h

Usage:
    python scripts/run_sliding_window_pipeline.py --signal R --device 0     # 单信号
    python scripts/run_sliding_window_pipeline.py --signal all --device 0   # 全部
    python scripts/run_sliding_window_pipeline.py --list                    # 列出实验
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

DATASET_SOURCE = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_gate2_train7200"
SPLIT_CSV = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv"
POOL_MASTER = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "candidate_pool_master.csv"
TEACHER_MANIFEST = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "best_epoch_manifest.json"

SW_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_sliding_window"
SW_RESULTS = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_sliding_window"
SW_DATASETS = REPO_ROOT / "YOLOv11" / "datasets" / "stage1_sliding_window"
SW_RUNS = REPO_ROOT / "YOLOv11" / "runs" / "stage1_sliding_window"

WINDOW_SIZE = 50
STEP_SIZE = 10
REPLAY_COUNT = 151
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
    for marker in ("YOLO-CV/", "YOLOv11/"):
        idx = cp.replace("\\", "/").find(marker)
        if idx >= 0:
            local = REPO_ROOT / cp.replace("\\", "/")[idx + len("YOLO-CV/"):]
            if local.exists():
                return str(local.resolve())
    return cp


def extract_training_dynamics(pool: list[dict]) -> dict[str, float]:
    """Extract per-sample training dynamics from hn00 per-epoch predictions."""
    cache_path = SW_ROOT / "training_dynamics_variability.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    hn00_per_epoch = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "per_epoch_gate"

    pool_ids = {r["image_id"] for r in pool}
    sample_logits: dict[str, list[float]] = {sid: [] for sid in pool_ids}

    if hn00_per_epoch.exists():
        epoch_dirs = sorted(
            [d for d in hn00_per_epoch.iterdir() if d.is_dir()],
            key=lambda d: int(d.name.split("_")[-1]) if d.name.startswith("epoch_") else 0
        )
        for epoch_dir in epoch_dirs:
            pred_csv = epoch_dir / "val_op_predictions_calibrated.csv"
            if not pred_csv.exists():
                continue
            with open(pred_csv, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    img_id = row.get("image_id", "")
                    if img_id in sample_logits:
                        score = float(row.get("calibrated_score", row.get("score", 0)))
                        if score > 0 and score < 1:
                            import math
                            sample_logits[img_id].append(math.log(score / (1 - score)))
                        else:
                            sample_logits[img_id].append(0.0)

    variability = {}
    for img_id, logits in sample_logits.items():
        if len(logits) >= 10:
            arr = np.array(logits)
            variability[img_id] = float(np.std(arr))
        else:
            variability[img_id] = 0.0

    if not variability or all(v == 0 for v in variability.values()):
        # fallback: use R as proxy
        for r in pool:
            variability[r["image_id"]] = float(r["R"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(variability, indent=2), encoding="utf-8")
    return variability


def sort_pool_by_signal(pool: list[dict], signal: str,
                        training_dynamics: dict[str, float] | None = None) -> list[dict]:
    """Sort pool by signal value, descending (rank 1 = highest signal value)."""
    if signal == "T":
        if training_dynamics is None:
            raise ValueError("Training dynamics not provided for signal T")
        pool_copy = []
        for r in pool:
            rc = dict(r)
            rc["T"] = training_dynamics.get(r["image_id"], 0.0)
            pool_copy.append(rc)
        return sorted(pool_copy, key=lambda r: float(r["T"]), reverse=True)
    else:
        return sorted(pool, key=lambda r: float(r[signal]), reverse=True)


def define_windows(signal: str) -> list[dict]:
    """Generate sliding window experiment definitions."""
    experiments = []
    pool_size = 250
    window_idx = 0
    start = 1
    while start + WINDOW_SIZE - 1 <= pool_size:
        end = start + WINDOW_SIZE - 1
        exp_id = f"sw_{signal}_w{window_idx:02d}_r{start:03d}_{end:03d}"
        center_pct = ((start + end) / 2.0 - 0.5) / pool_size * 100
        experiments.append({
            "id": exp_id,
            "signal": signal,
            "window_idx": window_idx,
            "rank_start": start,
            "rank_end": end,
            "center_pct": round(center_pct, 1),
            "n_unique": WINDOW_SIZE,
            "replay": REPLAY_COUNT,
        })
        start += STEP_SIZE
        window_idx += 1
    return experiments


def build_id_to_filename(pool: list[dict]) -> dict[str, str]:
    """Map image_id to actual filename."""
    mapping = {}
    for row in pool:
        img_id = row["image_id"]
        rel_path = row.get("img_rel_path", "")
        if rel_path:
            mapping[img_id] = Path(rel_path).name
        else:
            numeric = img_id.replace("Normal_", "")
            mapping[img_id] = f"{numeric}.png"
    return mapping


def build_dataset_view(exp_id: str, selected_ids: list[str],
                       id_to_filename: dict[str, str]) -> Path:
    """Build training dataset with selected samples replayed."""
    ds_root = SW_DATASETS / exp_id
    marker_path = ds_root / "_built_marker.json"

    if marker_path.exists():
        return ds_root

    train_normal = ds_root / "train" / "Normal"
    train_abnormal = ds_root / "train" / "Abnormal"
    val_normal = ds_root / "val" / "Normal"
    val_abnormal = ds_root / "val" / "Abnormal"

    for d in [train_normal, train_abnormal, val_normal, val_abnormal]:
        d.mkdir(parents=True, exist_ok=True)

    src_dirs = {
        "train_abnormal": DATASET_SOURCE / "train" / "Abnormal",
        "train_normal": DATASET_SOURCE / "train" / "Normal",
        "val_abnormal": DATASET_SOURCE / "val" / "Abnormal",
        "val_normal": DATASET_SOURCE / "val" / "Normal",
    }

    def link_dir(src: Path, dst: Path):
        for img in src.iterdir():
            target = dst / img.name
            if not target.exists():
                try:
                    os.link(str(img), str(target))
                except OSError:
                    shutil.copy2(str(img), str(target))

    link_dir(src_dirs["train_abnormal"], train_abnormal)
    link_dir(src_dirs["train_normal"], train_normal)
    link_dir(src_dirs["val_abnormal"], val_abnormal)
    link_dir(src_dirs["val_normal"], val_normal)

    n_unique = len(selected_ids)
    for copy_idx in range(REPLAY_COUNT):
        src_id = selected_ids[copy_idx % n_unique]
        filename = id_to_filename.get(src_id, f"{src_id}.png")
        src_file = src_dirs["train_normal"] / filename
        if not src_file.exists():
            alt = list(src_dirs["train_normal"].glob(f"*{src_id.replace('Normal_', '')}*"))
            if alt:
                src_file = alt[0]
            else:
                continue
        dst_name = f"_replay_{exp_id}_{copy_idx:04d}_{src_file.name}"
        dst = train_normal / dst_name
        if not dst.exists():
            try:
                os.link(str(src_file), str(dst))
            except OSError:
                shutil.copy2(str(src_file), str(dst))

    marker_path.write_text(json.dumps({
        "experiment_id": exp_id,
        "n_unique": n_unique,
        "replay": REPLAY_COUNT,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return ds_root


def run_training(exp_id: str, dataset_root: Path, teacher_ckpt: str,
                 device: str) -> Path:
    """Train using ultralytics YOLO API."""
    run_dir = SW_RUNS / exp_id
    if (run_dir / "weights" / "last.pt").exists():
        print(f"    training exists, skipping")
        return run_dir

    from ultralytics import YOLO
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
        project=str(SW_RUNS),
        name=exp_id,
        exist_ok=True,
        seed=0,
    )
    return run_dir


def run_gate_eval(exp_id: str, run_dir: Path) -> dict[str, Any]:
    """Run gate evaluation on all checkpoints."""
    summary_dir = SW_ROOT / exp_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    if (summary_dir / "best_epoch_manifest.json").exists():
        manifest = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    eval exists: Spec@R99.5={manifest.get('spec_at_r995', '?')}")
        return manifest

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
        print(f"    WARNING: gate eval failed for {exp_id}")
        return {}

    if (summary_dir / "best_epoch_manifest.json").exists():
        manifest = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    eval complete: Spec@R99.5={manifest.get('spec_at_r995', '?')}")
        return manifest
    return {}


def run_experiment(exp: dict, pool_sorted: list[dict],
                   id_to_filename: dict[str, str],
                   teacher_ckpt: str, device: str) -> dict[str, Any]:
    """Run single sliding window experiment."""
    exp_id = exp["id"]
    rank_start = exp["rank_start"]
    rank_end = exp["rank_end"]

    print(f"\n{'='*60}")
    print(f"  {exp_id}  (rank {rank_start}-{rank_end}, center={exp['center_pct']}%)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    selected = pool_sorted[rank_start - 1:rank_end]
    selected_ids = [r["image_id"] for r in selected]

    # signal value stats for this window
    signal = exp["signal"]
    sig_values = [float(r.get(signal, 0)) for r in selected]
    sig_min = min(sig_values) if sig_values else 0
    sig_max = max(sig_values) if sig_values else 0
    sig_mean = sum(sig_values) / len(sig_values) if sig_values else 0
    print(f"    {signal} range: [{sig_min:.4f}, {sig_max:.4f}], mean={sig_mean:.4f}")

    ds_root = build_dataset_view(exp_id, selected_ids, id_to_filename)
    t0 = time.time()
    run_dir = run_training(exp_id, ds_root, teacher_ckpt, device)
    train_time = time.time() - t0
    manifest = run_gate_eval(exp_id, run_dir)

    result = {
        "experiment_id": exp_id,
        "signal": signal,
        "window_idx": exp["window_idx"],
        "rank_start": rank_start,
        "rank_end": rank_end,
        "center_pct": exp["center_pct"],
        "n_unique": len(selected_ids),
        "signal_min": round(sig_min, 6),
        "signal_max": round(sig_max, 6),
        "signal_mean": round(sig_mean, 6),
        "spec_at_r995": manifest.get("spec_at_r995"),
        "spec_at_r990": manifest.get("spec_at_r990"),
        "prec_at_r990": manifest.get("prec_at_r990"),
        "ptr_at_r990": manifest.get("ptr_at_r990"),
        "best_epoch": manifest.get("epoch"),
        "train_hours": round(train_time / 3600, 2),
    }

    result_path = SW_RESULTS / f"{exp_id}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Sliding window pipeline for Goldilocks peak localization.")
    parser.add_argument("--signal", default="all", help="Signal to scan: R, D, T, C, or all")
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    args = parser.parse_args()

    signals = ["R", "D", "T", "C"] if args.signal == "all" else [args.signal.upper()]

    all_experiments = {}
    for sig in signals:
        all_experiments[sig] = define_windows(sig)

    if args.list:
        total = sum(len(v) for v in all_experiments.values())
        print(f"\nSliding Window Pipeline — {total} experiments\n")
        for sig, exps in all_experiments.items():
            print(f"  Signal {sig}: {len(exps)} windows")
            for e in exps:
                print(f"    {e['id']}  rank {e['rank_start']:3d}-{e['rank_end']:3d}  center={e['center_pct']}%")
        print(f"\n  Total: {total} runs × ~2.6h = ~{total*2.6:.0f}h")
        return

    # preflight
    for path, name in [(DATASET_SOURCE, "dataset"), (SPLIT_CSV, "split_csv"),
                        (POOL_MASTER, "pool_master"), (TEACHER_MANIFEST, "teacher_manifest")]:
        if not path.exists():
            raise SystemExit(f"Missing: {name} at {path}")

    pool = load_pool()
    teacher_ckpt = load_teacher_checkpoint()
    id_to_filename = build_id_to_filename(pool)

    print(f"\nSliding Window Pipeline")
    print(f"  Signals: {signals}")
    print(f"  Window: {WINDOW_SIZE} samples, step {STEP_SIZE}")
    print(f"  Pool: {len(pool)} samples")
    print(f"  Teacher: {teacher_ckpt}\n")

    # pre-extract training dynamics if needed
    training_dynamics = None
    if "T" in signals:
        print("  Extracting training dynamics signal...")
        training_dynamics = extract_training_dynamics(pool)
        print(f"  Training dynamics: {len(training_dynamics)} samples\n")

    # sort pool by each signal once
    sorted_pools = {}
    for sig in signals:
        sorted_pools[sig] = sort_pool_by_signal(pool, sig, training_dynamics)
        print(f"  {sig} sorted: top1={sorted_pools[sig][0]['image_id']} "
              f"({sig}={float(sorted_pools[sig][0].get(sig, 0)):.4f})")

    # run all experiments
    t_start = time.time()
    all_results = []

    total_exps = sum(len(v) for v in all_experiments.values())
    exp_counter = 0

    for sig in signals:
        exps = all_experiments[sig]
        pool_sorted = sorted_pools[sig]

        for exp in exps:
            exp_counter += 1
            print(f"\n  [{exp_counter}/{total_exps}]")
            try:
                result = run_experiment(exp, pool_sorted, id_to_filename, teacher_ckpt, args.device)
                all_results.append(result)
            except Exception as exc:
                print(f"    FAILED: {exc}")
                import traceback
                traceback.print_exc()
                all_results.append({"experiment_id": exp["id"], "signal": sig, "error": str(exc)})

    # write summary
    total_hours = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Total: {total_hours:.1f}h")
    print(f"{'='*60}\n")

    summary_path = SW_RESULTS / "sliding_window_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment_id", "signal", "window_idx", "rank_start", "rank_end",
                  "center_pct", "n_unique", "signal_min", "signal_max", "signal_mean",
                  "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990",
                  "best_epoch", "train_hours"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([r for r in all_results if "error" not in r])

    # per-signal summary
    for sig in signals:
        sig_results = [r for r in all_results if r.get("signal") == sig and "error" not in r]
        if not sig_results:
            continue
        print(f"\n  Signal {sig} response curve:")
        print(f"  {'Rank':>10s}  {'Center%':>8s}  {'Spec@R99.5':>12s}  {'BestEp':>6s}")
        for r in sig_results:
            spec = r.get("spec_at_r995", "?")
            ep = r.get("best_epoch", "?")
            print(f"  {r['rank_start']:3d}-{r['rank_end']:3d}    {r['center_pct']:6.1f}%    {spec:>12}    {ep:>6}")

        # find peak
        valid = [r for r in sig_results if r.get("spec_at_r995") is not None]
        if valid:
            peak = max(valid, key=lambda r: r["spec_at_r995"])
            print(f"\n  Peak: {peak['experiment_id']} (rank {peak['rank_start']}-{peak['rank_end']}, "
                  f"Spec@R99.5={peak['spec_at_r995']:.4f})")

    print(f"\n  Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
