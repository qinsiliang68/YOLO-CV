"""
train_v3_stage1.py — End-to-end Stage 1 training + external evaluation for one capacity.

Pipeline (runs fully automatic, no downstream human intervention needed):

  1. Prepare data: ensure data_dir has train/, val/ (val alias to val_cal)
  2. Train YOLO11{capacity}-cls for N epochs, save_period=1 (all checkpoints)
  3. Per-epoch external evaluation:
     - Inference on val_cal with torch.no_grad, get logits
     - Fit temperature T* via multiclass NLL minimization (L-BFGS)
     - Inference on val_op with calibrated probs
     - Binary P(defect|x) = 1 - softmax(z/T*)[Normal_idx]
     - Search threshold tau* such that Recall(tau*) >= 0.995 (and 0.990)
     - Compute Spec@R99.5, Spec@R99.0, Prec@R99.0, PTR@R99.0
  4. Lex-rank all epochs, pick best
  5. Apply (theta*, T*, tau*) to test set -> final Spec@R99.5, Spec@R99.0 + Wilson CI
  6. Save all outputs to output_dir:
     - per_epoch_metrics.csv   200 rows, all epoch metrics
     - best_epoch.json         best epoch selection log
     - final_test_metrics.json final frozen-model test metrics
     - weights/epoch*.pt       200 checkpoints

Usage:
    uv run python scripts/train_v3_stage1.py --capacity n --data-dir DATA --output-dir OUT
    uv run python scripts/train_v3_stage1.py --capacity n --data-dir DATA --output-dir OUT --smoke
"""
import argparse
import json
import math
import os
import subprocess
import sys
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
ALL_FOLDERS = ["Normal"] + PRIORITY  # matches export layout

DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 24
DEFAULT_IMGSZ = 224


def ensure_val_layout(data_dir: Path):
    """Ultralytics cls expects train/+val/. Alias val -> val_cal if missing."""
    val = data_dir / "val"
    val_cal = data_dir / "val_cal"
    if val.exists():
        return
    if not val_cal.exists():
        raise FileNotFoundError(f"data_dir must contain val/ or val_cal/; got neither at {data_dir}")
    # try symlink, fall back to Windows junction, finally copy
    try:
        val.symlink_to(val_cal, target_is_directory=True)
        print(f"[setup] symlinked {val.name} -> val_cal")
    except (OSError, NotImplementedError):
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(val), str(val_cal)],
                check=True, capture_output=True, shell=False,
            )
            print(f"[setup] junction {val.name} -> val_cal")
        except Exception as e:
            import shutil as _sh
            _sh.copytree(val_cal, val)
            print(f"[setup] copied val_cal -> val (fallback, reason: {e})")


def get_class_mapping(data_dir: Path):
    """Return (class_names sorted as ultralytics sees them, normal_class_idx)."""
    tmp = ImageFolder(str(data_dir / "train"))
    names = tmp.classes  # alphabetical
    return names, names.index("Normal")


def run_training(capacity: str, data_dir: Path, output_dir: Path,
                 epochs: int, batch: int, imgsz: int):
    """Launch ultralytics training. Returns path to run dir (contains weights/)."""
    model = YOLO(f"yolo11{capacity}-cls.pt")
    project = str(output_dir.parent)
    name = output_dir.name
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        save_period=1,
        patience=0,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=False,
        val=True,        # ultralytics in-training val (we ignore its metrics)
        optimizer="auto",
        device=0 if torch.cuda.is_available() else "cpu",
    )
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else output_dir
    return run_dir


def get_inference_transforms(imgsz: int):
    return transforms.Compose([
        transforms.Resize(imgsz),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def infer_logits(checkpoint_path: Path, split_dir: Path, imgsz: int, batch: int, device):
    """Run forward pass on all images in split_dir, return logits (N, C) + labels (N,)."""
    model = YOLO(str(checkpoint_path))
    torch_model = model.model.to(device).eval()
    tfm = get_inference_transforms(imgsz)
    dataset = ImageFolder(str(split_dir), transform=tfm)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    class_names = dataset.classes
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = torch_model(x)
            # ultralytics cls forward returns logits tensor or tuple/list; unwrap
            if isinstance(out, (list, tuple)):
                out = out[0]
            all_logits.append(out.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits, dim=0).float()
    labels = torch.cat(all_labels, dim=0).long()
    return logits, labels, class_names


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass temperature scaling via L-BFGS on NLL."""
    T = torch.tensor([1.0], requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.clamp(min=1e-3).item())


def compute_binary_prob(logits: torch.Tensor, T: float, normal_idx: int) -> torch.Tensor:
    """P(defect | x) = 1 - softmax(z/T)[normal_idx]."""
    probs = torch.softmax(logits / T, dim=-1)
    p_normal = probs[:, normal_idx]
    return 1.0 - p_normal


def search_tau_for_recall(p_defect: np.ndarray, y_binary: np.ndarray, target_recall: float):
    """Find min tau such that Recall(tau) >= target_recall on positives only."""
    pos_scores = p_defect[y_binary == 1]
    if len(pos_scores) == 0:
        return None
    sorted_desc = np.sort(pos_scores)[::-1]
    # need top-K scores to hit recall; K = ceil(target_recall * n_pos)
    k = int(np.ceil(target_recall * len(sorted_desc)))
    k = max(1, min(k, len(sorted_desc)))
    tau = sorted_desc[k - 1]
    return float(tau)


def binary_metrics(p_defect: np.ndarray, y_binary: np.ndarray, tau: float):
    """Given tau, compute recall, spec, prec, ptr."""
    pred = (p_defect >= tau).astype(int)
    tp = int(((pred == 1) & (y_binary == 1)).sum())
    fn = int(((pred == 0) & (y_binary == 1)).sum())
    tn = int(((pred == 0) & (y_binary == 0)).sum())
    fp = int(((pred == 1) & (y_binary == 0)).sum())
    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    ptr = (tp + fp) / max(len(p_defect), 1)
    return dict(recall=recall, spec=spec, prec=prec, ptr=ptr, tp=tp, fn=fn, tn=tn, fp=fp)


def wilson_half_width(n_neg: int, p: float = 0.5, alpha_z: float = 1.96) -> float:
    if n_neg <= 0:
        return float("nan")
    return alpha_z * math.sqrt(p * (1 - p) / n_neg)


def binary_labels_from_folder(labels: torch.Tensor, normal_idx: int) -> np.ndarray:
    return (labels.numpy() != normal_idx).astype(int)


def evaluate_epoch(ckpt_path, val_cal_dir, val_op_dir, imgsz, batch, device, normal_idx):
    # val_cal -> fit T
    cal_logits, cal_labels, _ = infer_logits(ckpt_path, val_cal_dir, imgsz, batch, device)
    T_star = fit_temperature(cal_logits, cal_labels)

    # val_op -> calibrated binary probs
    op_logits, op_labels, _ = infer_logits(ckpt_path, val_op_dir, imgsz, batch, device)
    op_p = compute_binary_prob(op_logits, T_star, normal_idx).numpy()
    op_y = binary_labels_from_folder(op_labels, normal_idx)

    tau_995 = search_tau_for_recall(op_p, op_y, 0.995)
    tau_990 = search_tau_for_recall(op_p, op_y, 0.990)

    m_995 = binary_metrics(op_p, op_y, tau_995) if tau_995 is not None else {}
    m_990 = binary_metrics(op_p, op_y, tau_990) if tau_990 is not None else {}

    return {
        "T_star": T_star,
        "tau_995": tau_995,
        "tau_990": tau_990,
        "spec@R995": m_995.get("spec", 0.0),
        "spec@R990": m_990.get("spec", 0.0),
        "prec@R990": m_990.get("prec", 0.0),
        "ptr@R990":  m_990.get("ptr", 0.0),
        "n_val_op_neg": int((op_y == 0).sum()),
        "n_val_op_pos": int((op_y == 1).sum()),
    }


def lex_rank_best(rows):
    """Sort rows by (spec@R995 desc, spec@R990 desc, prec@R990 desc, ptr@R990 asc). First is best."""
    sortkey = lambda r: (-r["spec@R995"], -r["spec@R990"], -r["prec@R990"], r["ptr@R990"])
    return sorted(rows, key=sortkey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", required=True, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--smoke", action="store_true", help="3 epochs, batch 4 (sanity test only)")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 3
        args.batch = 4
        print("[mode] SMOKE TEST: 3 epochs, batch 4")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data layout for ultralytics
    ensure_val_layout(data_dir)
    class_names, normal_idx = get_class_mapping(data_dir)
    print(f"[setup] data_dir={data_dir}")
    print(f"[setup] classes={class_names}  normal_idx={normal_idx}")

    # 2. Train
    print(f"\n[train] capacity=yolo11{args.capacity}  epochs={args.epochs}  batch={args.batch}")
    run_dir = run_training(args.capacity, data_dir, output_dir,
                           args.epochs, args.batch, args.imgsz)
    print(f"[train] DONE. run_dir={run_dir}")

    # 3. Per-epoch external evaluation
    weights_dir = run_dir / "weights"
    ckpts = sorted([p for p in weights_dir.iterdir() if p.name.startswith("epoch")],
                   key=lambda p: int(p.stem.replace("epoch", "")))
    if not ckpts:
        # fall back: only last.pt saved (save_period ignored or early termination)
        ckpts = sorted([p for p in weights_dir.iterdir() if p.suffix == ".pt"
                        and p.name not in ("best.pt", "last.pt")])
        if not ckpts:
            ckpts = [weights_dir / "last.pt"]
    print(f"\n[eval] {len(ckpts)} checkpoints to evaluate")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_cal_dir = data_dir / "val_cal"
    val_op_dir = data_dir / "val_op"
    test_dir = data_dir / "test"

    rows = []
    for i, ckpt in enumerate(ckpts):
        epoch_id = ckpt.stem.replace("epoch", "") if ckpt.stem.startswith("epoch") else ckpt.stem
        metrics = evaluate_epoch(ckpt, val_cal_dir, val_op_dir, args.imgsz, args.batch, device, normal_idx)
        metrics["epoch"] = epoch_id
        metrics["checkpoint"] = str(ckpt.name)
        rows.append(metrics)
        print(f"  [eval] {ckpt.name}: T={metrics['T_star']:.3f} tau995={metrics['tau_995']} "
              f"spec@R995={metrics['spec@R995']:.4f}")

    # save per-epoch
    per_epoch_csv = output_dir / "per_epoch_metrics.csv"
    pd.DataFrame(rows).to_csv(per_epoch_csv, index=False)
    print(f"\n[save] per_epoch_metrics.csv ({len(rows)} rows)")

    # 4. Lex-rank best
    ranked = lex_rank_best(rows)
    best = ranked[0]
    best_info = {
        "best_epoch": best["epoch"],
        "best_checkpoint": best["checkpoint"],
        "lex_rank": [
            ("spec@R995", best["spec@R995"]),
            ("spec@R990", best["spec@R990"]),
            ("prec@R990", best["prec@R990"]),
            ("-ptr@R990", -best["ptr@R990"]),
        ],
        "T_star": best["T_star"],
        "tau_995": best["tau_995"],
        "tau_990": best["tau_990"],
    }
    (output_dir / "best_epoch.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    print(f"[best] epoch={best['epoch']}  spec@R995={best['spec@R995']:.4f}")

    # 5. Final test evaluation with frozen best checkpoint + (T*, tau*)
    best_ckpt = weights_dir / best["checkpoint"]
    t_logits, t_labels, _ = infer_logits(best_ckpt, test_dir, args.imgsz, args.batch, device)
    t_p = compute_binary_prob(t_logits, best["T_star"], normal_idx).numpy()
    t_y = binary_labels_from_folder(t_labels, normal_idx)

    m_t995 = binary_metrics(t_p, t_y, best["tau_995"]) if best["tau_995"] is not None else {}
    m_t990 = binary_metrics(t_p, t_y, best["tau_990"]) if best["tau_990"] is not None else {}
    n_neg = int((t_y == 0).sum())

    final = {
        "capacity": args.capacity,
        "best_epoch": best["epoch"],
        "T_star": best["T_star"],
        "tau_995": best["tau_995"],
        "tau_990": best["tau_990"],
        "test_n": len(t_y),
        "test_n_negative": n_neg,
        "test_n_positive": int((t_y == 1).sum()),
        "spec@R995_test": m_t995.get("spec", 0.0),
        "spec@R990_test": m_t990.get("spec", 0.0),
        "prec@R990_test": m_t990.get("prec", 0.0),
        "ptr@R990_test":  m_t990.get("ptr", 0.0),
        "recall@R995_test": m_t995.get("recall", 0.0),
        "recall@R990_test": m_t990.get("recall", 0.0),
        "wilson_half_width_pp@p=0.5": wilson_half_width(n_neg) * 100,
    }
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(f"\n[TEST] capacity={args.capacity}  epoch={best['epoch']}")
    print(f"       spec@R99.5 = {final['spec@R995_test']:.4f}")
    print(f"       spec@R99.0 = {final['spec@R990_test']:.4f}")
    print(f"       Wilson hw(p=0.5) = {final['wilson_half_width_pp@p=0.5']:.2f} pp")
    print(f"\n[DONE] all outputs in {output_dir}")


if __name__ == "__main__":
    main()
