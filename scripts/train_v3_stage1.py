"""
train_v3_stage1.py — Stage 1 training + raw-material dump (no derivatives saved).

**Save discipline** — linearly independent base materials ONLY:
    what is saved = what cannot be re-derived without re-running this script.

Saved per run (all under output-dir):
  weights/epoch0.pt ... epoch{N-1}.pt    200 checkpoints (save_period=1)
  per_epoch_logits/
    val_cal_epoch{i}.npz                 (logits, labels, image_ids)  N epochs
    val_op_epoch{i}.npz                  (logits, labels, image_ids)  N epochs
  best_epoch/
    test_logits.npz                      (logits, labels, image_ids) on test @ best ckpt
    embeddings_train.npz                 (features, labels, image_ids) penultimate
    embeddings_val_op.npz                same
    embeddings_test.npz                  same
  run_meta.json                          git commit / host / GPU / time / duration
  args.yaml, results.csv                 ultralytics internal (kept as-is)

**Not saved** (derivable from the above — recompute via
scripts/aggregate_capacity_results.py):
  per_epoch_metrics.csv   best_epoch.json   final_test_metrics.json
  confusion_matrix_*.csv  per_class_recall_*.csv  tau_spec_curve_*.csv
  T*, tau*, Spec, Recall, Prec, PTR at any recall target, etc.

Pipeline steps performed during run (console output only, not persisted as derivatives):
  1. Train YOLO11{capacity}-cls for N epochs
  2. Per-epoch: infer on val_cal + val_op, save raw logits
  3. In-memory: fit T*, search tau*, lex-rank to pick best epoch (for test inference
     only — the values ARE re-derivable from saved logits)
  4. Run test inference on best ckpt, save raw logits
  5. Run embedding extraction on train/val_op/test via penultimate-layer hook, save raw features

Usage:
    uv run python scripts/train_v3_stage1.py --capacity n --data-dir DATA --output-dir OUT
    uv run python scripts/train_v3_stage1.py --capacity n --data-dir DATA --output-dir OUT --smoke
"""
import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "YOLOv11"))
from ultralytics import YOLO  # noqa: E402

MAIN_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
PRIORITY = ["PF", "DE", "RB", "AF", "OB", "FS"]

DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 64
DEFAULT_IMGSZ = 224


# ---------- utility ----------

def ensure_val_layout(data_dir: Path):
    val = data_dir / "val"
    val_cal = data_dir / "val_cal"
    if val.exists():
        return
    if not val_cal.exists():
        raise FileNotFoundError(f"need val/ or val_cal/ in {data_dir}")
    try:
        val.symlink_to(val_cal, target_is_directory=True)
        print(f"[setup] symlink val -> val_cal")
    except (OSError, NotImplementedError):
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(val), str(val_cal)],
                           check=True, capture_output=True, shell=False)
            print(f"[setup] junction val -> val_cal")
        except Exception:
            import shutil as _sh
            _sh.copytree(val_cal, val)
            print(f"[setup] copied val_cal -> val (fallback)")


def get_class_mapping(data_dir: Path):
    tmp = ImageFolder(str(data_dir / "train"))
    return tmp.classes, tmp.classes.index("Normal")


def git_commit_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------- training ----------

def run_training(capacity, data_dir, output_dir, epochs, batch, imgsz):
    model = YOLO(f"yolo11{capacity}-cls.pt")
    project = str(output_dir.parent)
    name = output_dir.name
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=4,
        close_mosaic=0,
        save_period=1,
        patience=0,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=False,
        val=True,
        optimizer="auto",
        device=0 if torch.cuda.is_available() else "cpu",
    )
    return Path(results.save_dir) if hasattr(results, "save_dir") else output_dir


# ---------- inference ----------

def get_transforms(imgsz):
    return transforms.Compose([
        transforms.Resize(imgsz),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_dataset_image_ids(dataset: ImageFolder):
    """image_ids aligned with dataset sample order (stem of each file)."""
    return [Path(p).stem for p, _ in dataset.samples]


def infer_logits(checkpoint_path: Path, split_dir: Path, imgsz, batch, device):
    """Return logits (N, C), labels (N,), image_ids list, class_names list."""
    model = YOLO(str(checkpoint_path))
    tm = model.model.to(device).eval()
    ds = ImageFolder(str(split_dir), transform=get_transforms(imgsz))
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    ids = _get_dataset_image_ids(ds)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = tm(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            all_logits.append(out.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits).float()
    labels = torch.cat(all_labels).long()
    return logits, labels, ids, ds.classes


def _find_classify_linear(torch_model):
    """Locate the final Linear layer in the Classify head."""
    last_linear = None
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    return last_linear


def infer_embeddings(checkpoint_path: Path, split_dir: Path, imgsz, batch, device):
    """Return penultimate-layer features (N, D), labels, image_ids via forward hook on Linear."""
    model = YOLO(str(checkpoint_path))
    tm = model.model.to(device).eval()
    linear = _find_classify_linear(tm)
    if linear is None:
        raise RuntimeError("Could not locate final Linear layer in classify head")

    feats = []
    def hook(module, inp, out):
        feats.append(inp[0].detach().cpu())

    h = linear.register_forward_hook(hook)
    try:
        ds = ImageFolder(str(split_dir), transform=get_transforms(imgsz))
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
        ids = _get_dataset_image_ids(ds)
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                _ = tm(x)
                all_labels.append(y)
        features = torch.cat(feats).float()
        labels = torch.cat(all_labels).long()
    finally:
        h.remove()
    return features, labels, ids


# ---------- calibration + metrics ----------

def fit_temperature(logits, labels):
    T = torch.tensor([1.0], requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(T.clamp(min=1e-3).item())


def compute_binary_prob(logits, T, normal_idx):
    probs = torch.softmax(logits / T, dim=-1)
    return (1.0 - probs[:, normal_idx]).numpy()


def binary_labels(labels, normal_idx):
    return (labels.numpy() != normal_idx).astype(int)


def search_tau(p_defect, y_binary, target_recall):
    pos = p_defect[y_binary == 1]
    if len(pos) == 0:
        return None
    sorted_desc = np.sort(pos)[::-1]
    k = max(1, min(int(np.ceil(target_recall * len(sorted_desc))), len(sorted_desc)))
    return float(sorted_desc[k - 1])


def binary_metrics(p_defect, y_binary, tau):
    pred = (p_defect >= tau).astype(int)
    tp = int(((pred == 1) & (y_binary == 1)).sum())
    fn = int(((pred == 0) & (y_binary == 1)).sum())
    tn = int(((pred == 0) & (y_binary == 0)).sum())
    fp = int(((pred == 1) & (y_binary == 0)).sum())
    return dict(
        recall=tp / max(tp + fn, 1),
        spec=tn / max(tn + fp, 1),
        prec=tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        ptr=(tp + fp) / max(len(p_defect), 1),
        tp=tp, fn=fn, tn=tn, fp=fp,
    )


def wilson_half_width(n_neg, p=0.5, z=1.96):
    if n_neg <= 0:
        return float("nan")
    return z * math.sqrt(p * (1 - p) / n_neg)


def full_tau_sweep(p_defect, y_binary, n_points=200):
    """Sweep tau from 0 to 1 and return (tau, recall, spec, prec, ptr) per point."""
    taus = np.linspace(0.0, 1.0, n_points + 1)
    rows = []
    for tau in taus:
        m = binary_metrics(p_defect, y_binary, float(tau))
        rows.append({"tau": float(tau), **{k: v for k, v in m.items() if k in
                     ("recall", "spec", "prec", "ptr", "tp", "fn", "tn", "fp")}})
    return pd.DataFrame(rows)


def confusion_matrix_multi(pred_class, true_class, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred_class, true_class):
        m[t, p] += 1
    return m


# ---------- per-epoch eval (save logits) ----------

def evaluate_epoch(ckpt, val_cal_dir, val_op_dir, imgsz, batch, device, normal_idx, save_dir=None):
    cal_logits, cal_labels, cal_ids, _ = infer_logits(ckpt, val_cal_dir, imgsz, batch, device)
    T = fit_temperature(cal_logits, cal_labels)
    op_logits, op_labels, op_ids, _ = infer_logits(ckpt, val_op_dir, imgsz, batch, device)
    op_p = compute_binary_prob(op_logits, T, normal_idx)
    op_y = binary_labels(op_labels, normal_idx)
    tau_995 = search_tau(op_p, op_y, 0.995)
    tau_990 = search_tau(op_p, op_y, 0.990)
    m995 = binary_metrics(op_p, op_y, tau_995) if tau_995 is not None else {}
    m990 = binary_metrics(op_p, op_y, tau_990) if tau_990 is not None else {}

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ep_tag = ckpt.stem  # e.g. "epoch5" or "last"
        # Save ONLY raw materials — T*, tau*, metrics are all derivable from
        # (logits, labels) and are intentionally NOT persisted.
        np.savez_compressed(
            save_dir / f"val_cal_{ep_tag}.npz",
            logits=cal_logits.numpy(), labels=cal_labels.numpy(),
            image_ids=np.array(cal_ids, dtype=object),
        )
        np.savez_compressed(
            save_dir / f"val_op_{ep_tag}.npz",
            logits=op_logits.numpy(), labels=op_labels.numpy(),
            image_ids=np.array(op_ids, dtype=object),
        )

    return {
        "T_star": T,
        "tau_995": tau_995, "tau_990": tau_990,
        "spec@R995": m995.get("spec", 0.0),
        "spec@R990": m990.get("spec", 0.0),
        "prec@R990": m990.get("prec", 0.0),
        "ptr@R990":  m990.get("ptr", 0.0),
        "n_val_op_neg": int((op_y == 0).sum()),
        "n_val_op_pos": int((op_y == 1).sum()),
    }


def lex_rank_best(rows):
    return sorted(rows, key=lambda r: (-r["spec@R995"], -r["spec@R990"],
                                        -r["prec@R990"], r["ptr@R990"]))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", required=True, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t_start = datetime.now(timezone.utc)

    if args.smoke:
        args.epochs = 3
        args.batch = 4
        print("[mode] SMOKE (3 epochs, batch 4)")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_epoch_logits_dir = output_dir / "per_epoch_logits"
    best_dir = output_dir / "best_epoch"

    ensure_val_layout(data_dir)
    class_names, normal_idx = get_class_mapping(data_dir)
    print(f"[setup] data={data_dir}  classes={class_names}  normal_idx={normal_idx}")

    # ---- training ----
    print(f"\n[train] yolo11{args.capacity}  epochs={args.epochs}  batch={args.batch}")
    run_dir = run_training(args.capacity, data_dir, output_dir,
                           args.epochs, args.batch, args.imgsz)
    print(f"[train] DONE.  run_dir={run_dir}")

    weights_dir = run_dir / "weights"
    ckpts = sorted([p for p in weights_dir.iterdir() if p.name.startswith("epoch")],
                   key=lambda p: int(p.stem.replace("epoch", "")))
    if not ckpts:
        ckpts = sorted([p for p in weights_dir.iterdir()
                        if p.suffix == ".pt" and p.name not in ("best.pt", "last.pt")])
        if not ckpts:
            ckpts = [weights_dir / "last.pt"]
    print(f"\n[eval] {len(ckpts)} checkpoints to evaluate + save logits")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_cal_dir = data_dir / "val_cal"
    val_op_dir  = data_dir / "val_op"
    test_dir    = data_dir / "test"
    train_dir   = data_dir / "train"

    rows = []
    for ckpt in ckpts:
        metrics = evaluate_epoch(ckpt, val_cal_dir, val_op_dir, args.imgsz,
                                 args.batch, device, normal_idx,
                                 save_dir=per_epoch_logits_dir)
        metrics["epoch"] = ckpt.stem.replace("epoch", "") if ckpt.stem.startswith("epoch") else ckpt.stem
        metrics["checkpoint"] = str(ckpt.name)
        rows.append(metrics)
        print(f"  [eval] {ckpt.name}  T={metrics['T_star']:.3f}  "
              f"tau995={metrics['tau_995']}  spec@R995={metrics['spec@R995']:.4f}")

    # In-memory lex-rank selection (NOT persisted — derivable from per_epoch_logits/)
    ranked = lex_rank_best(rows)
    best = ranked[0]
    best_ckpt = weights_dir / best["checkpoint"]
    T_best = best["T_star"]
    print(f"\n[best] epoch={best['epoch']}  spec@R995={best['spec@R995']:.4f}  T*={T_best:.3f}")

    # ---- best-epoch raw dump ONLY (derivatives intentionally not persisted) ----
    best_dir.mkdir(parents=True, exist_ok=True)

    # test logits (raw) — labels + image_ids enough to recompute everything
    t_logits, t_labels, t_ids, _ = infer_logits(best_ckpt, test_dir, args.imgsz, args.batch, device)
    np.savez_compressed(
        best_dir / "test_logits.npz",
        logits=t_logits.numpy(), labels=t_labels.numpy(),
        image_ids=np.array(t_ids, dtype=object),
    )
    print(f"[best_raw] test_logits.npz (raw logits only)")

    # embeddings on train + val_op + test (raw features, only way to get them
    # without re-running inference with the hook)
    for split_name, split_dir in [("train", train_dir), ("val_op", val_op_dir), ("test", test_dir)]:
        print(f"[best_raw] extracting embeddings on {split_name} ...")
        feats, labs, ids = infer_embeddings(best_ckpt, split_dir, args.imgsz, args.batch, device)
        np.savez_compressed(
            best_dir / f"embeddings_{split_name}.npz",
            features=feats.numpy(), labels=labs.numpy(),
            image_ids=np.array(ids, dtype=object),
        )
        print(f"           embeddings_{split_name}.npz  shape={tuple(feats.shape)}")

    # console-only final-test report (NOT persisted — recomputable from test_logits.npz
    # via scripts/aggregate_capacity_results.py)
    tau995_best = best["tau_995"]
    tau990_best = best["tau_990"]
    t_p = compute_binary_prob(t_logits, T_best, normal_idx)
    t_y = binary_labels(t_labels, normal_idx)
    m_t995 = binary_metrics(t_p, t_y, tau995_best) if tau995_best is not None else {}
    m_t990 = binary_metrics(t_p, t_y, tau990_best) if tau990_best is not None else {}
    n_neg = int((t_y == 0).sum())
    print(f"\n[TEST console] {args.capacity}/epoch={best['epoch']}:")
    print(f"    spec@R99.5 = {m_t995.get('spec', 0.0):.4f}")
    print(f"    spec@R99.0 = {m_t990.get('spec', 0.0):.4f}")
    print(f"    Wilson hw  = {wilson_half_width(n_neg) * 100:.2f} pp  (n_neg = {n_neg})")
    print(f"    T* = {T_best:.3f}  tau995 = {tau995_best}  tau990 = {tau990_best}")
    print(f"(these numbers are NOT saved — compute from best_epoch/test_logits.npz + T* from "
          f"per_epoch_logits/val_cal_{best_ckpt.stem}.npz if needed)")

    # ---- run metadata ----
    t_end = datetime.now(timezone.utc)
    meta = {
        "capacity": args.capacity,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "smoke": args.smoke,
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "time_start_utc": t_start.isoformat(),
        "time_end_utc": t_end.isoformat(),
        "duration_seconds": (t_end - t_start).total_seconds(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "n_checkpoints_evaluated": len(ckpts),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] all outputs in {output_dir}")
    print(f"       wall-clock: {meta['duration_seconds']:.1f}s  host={meta['hostname']}")


if __name__ == "__main__":
    main()
