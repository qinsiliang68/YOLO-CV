"""
aggregate_capacity_results.py — recompute all derivatives from raw artifacts and
produce comparison summary.

Strict "linearly-independent raw materials" discipline: the per-capacity run
directories contain ONLY raw logits + embeddings + checkpoints. This script
performs the full downstream derivation from scratch:

  for each capacity:
    for each epoch:
      load per_epoch_logits/val_cal_epoch{i}.npz -> fit T*
      load per_epoch_logits/val_op_epoch{i}.npz  -> apply T*, search tau*_995/990,
                                                      compute Spec/Prec/PTR
    lex-rank (Spec@R995, Spec@R990, Prec@R990, -PTR@R990) -> best epoch
    load best_epoch/test_logits.npz -> apply T*(best) & tau* -> final Spec/Recall

Outputs (all in output-dir):
  per_capacity/yolo11{cap}/
    per_epoch_metrics.csv   recomputed
    best_epoch.json         recomputed
    final_test.json         recomputed
    confusion_matrix_test.csv
    per_class_recall_test.csv
    tau_spec_curve_val_op.csv
  capacity_comparison.csv
  capacity_comparison.md
  summary.json

Run:
    uv run python scripts/aggregate_capacity_results.py \\
        --runs-dir /path/to/runs --output-dir /path/to/summary
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CAPACITIES = ["n", "s", "m", "l", "x"]
# 7-class names expected in the ImageFolder sorted order used at train time:
#   alphabetical -> AF, DE, FS, Normal, OB, PF, RB
# (Normal is at index 3 in that canonical order)
CLASS_NAMES_ALPHA = sorted(["Normal", "PF", "DE", "RB", "AF", "OB", "FS"])
NORMAL_IDX_DEFAULT = CLASS_NAMES_ALPHA.index("Normal")


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(min=1e-3).item())


def binary_prob(logits: torch.Tensor, T: float, normal_idx: int) -> np.ndarray:
    return (1.0 - torch.softmax(logits / T, dim=-1)[:, normal_idx]).numpy()


def binary_labels(labels: torch.Tensor, normal_idx: int) -> np.ndarray:
    return (labels.numpy() != normal_idx).astype(int)


def search_tau(p_defect, y_bin, target_recall):
    pos = p_defect[y_bin == 1]
    if len(pos) == 0:
        return None
    sdesc = np.sort(pos)[::-1]
    k = max(1, min(int(np.ceil(target_recall * len(sdesc))), len(sdesc)))
    return float(sdesc[k - 1])


def metrics(p_defect, y_bin, tau):
    pred = (p_defect >= tau).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    return dict(
        recall=tp / max(tp + fn, 1),
        spec=tn / max(tn + fp, 1),
        prec=tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        ptr=(tp + fp) / max(len(p_defect), 1),
        tp=tp, fn=fn, tn=tn, fp=fp,
    )


def wilson_hw(n_neg, p=0.5, z=1.96):
    return z * math.sqrt(p * (1 - p) / n_neg) if n_neg > 0 else float("nan")


def full_tau_sweep(p, y, n_points=200):
    taus = np.linspace(0.0, 1.0, n_points + 1)
    return pd.DataFrame([{"tau": float(t), **{k: v for k, v in metrics(p, y, float(t)).items()
                                                if k in ("recall","spec","prec","ptr","tp","fn","tn","fp")}}
                          for t in taus])


def confusion_matrix_multi(pred_class, true_class, n):
    m = np.zeros((n, n), dtype=int)
    for p, t in zip(pred_class, true_class):
        m[t, p] += 1
    return m


def normal_idx_from_meta(run_dir: Path, default_idx: int) -> int:
    """Pick Normal class index from run_meta's class names if present, else default."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return default_idx
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # we didn't persist class_names; fall through to default
        return default_idx
    except Exception:
        return default_idx


def process_capacity(run_dir: Path, out_dir: Path, normal_idx: int) -> dict:
    """Recompute all metrics for one capacity from raw artifacts."""
    per_ep_dir = run_dir / "per_epoch_logits"
    best_dir = run_dir / "best_epoch"

    val_cal_files = sorted(per_ep_dir.glob("val_cal_epoch*.npz"),
                            key=lambda p: int(p.stem.split("epoch")[1]))
    val_op_files = sorted(per_ep_dir.glob("val_op_epoch*.npz"),
                           key=lambda p: int(p.stem.split("epoch")[1]))

    if len(val_cal_files) == 0 or len(val_op_files) == 0:
        # maybe no per-epoch saves; try single "last"
        val_cal_files = sorted(per_ep_dir.glob("val_cal_*.npz"))
        val_op_files = sorted(per_ep_dir.glob("val_op_*.npz"))
    if len(val_cal_files) == 0:
        raise RuntimeError(f"no per_epoch_logits files in {per_ep_dir}")

    rows = []
    for cal_f, op_f in zip(val_cal_files, val_op_files):
        cal = np.load(cal_f, allow_pickle=True)
        op = np.load(op_f, allow_pickle=True)
        cal_logits = torch.from_numpy(cal["logits"]).float()
        cal_labels = torch.from_numpy(cal["labels"]).long()
        op_logits = torch.from_numpy(op["logits"]).float()
        op_labels = torch.from_numpy(op["labels"]).long()
        T = fit_temperature(cal_logits, cal_labels)
        op_p = binary_prob(op_logits, T, normal_idx)
        op_y = binary_labels(op_labels, normal_idx)
        tau_995 = search_tau(op_p, op_y, 0.995)
        tau_990 = search_tau(op_p, op_y, 0.990)
        m995 = metrics(op_p, op_y, tau_995) if tau_995 is not None else {}
        m990 = metrics(op_p, op_y, tau_990) if tau_990 is not None else {}
        ep = cal_f.stem.split("epoch")[-1] if "epoch" in cal_f.stem else cal_f.stem
        rows.append({
            "epoch": ep,
            "T_star": T,
            "tau_995": tau_995,
            "tau_990": tau_990,
            "spec@R995": m995.get("spec", 0.0),
            "spec@R990": m990.get("spec", 0.0),
            "prec@R990": m990.get("prec", 0.0),
            "ptr@R990":  m990.get("ptr", 0.0),
        })
    per_epoch_df = pd.DataFrame(rows)
    per_epoch_path = out_dir / "per_epoch_metrics.csv"
    per_epoch_df.to_csv(per_epoch_path, index=False)

    # lex-rank
    ranked = per_epoch_df.sort_values(
        by=["spec@R995", "spec@R990", "prec@R990", "ptr@R990"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    best_info = {
        "best_epoch": str(best["epoch"]),
        "T_star": best["T_star"],
        "tau_995": best["tau_995"],
        "tau_990": best["tau_990"],
        "spec@R995_val_op": best["spec@R995"],
        "spec@R990_val_op": best["spec@R990"],
    }
    (out_dir / "best_epoch.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")

    # test metrics
    test_npz = best_dir / "test_logits.npz"
    if not test_npz.exists():
        print(f"  WARNING: {test_npz} missing; skipping test metrics")
        return {"summary": best_info}
    td = np.load(test_npz, allow_pickle=True)
    t_logits = torch.from_numpy(td["logits"]).float()
    t_labels = torch.from_numpy(td["labels"]).long()
    T_best = best["T_star"]
    t_p = binary_prob(t_logits, T_best, normal_idx)
    t_y = binary_labels(t_labels, normal_idx)
    m_t995 = metrics(t_p, t_y, best["tau_995"]) if best["tau_995"] is not None else {}
    m_t990 = metrics(t_p, t_y, best["tau_990"]) if best["tau_990"] is not None else {}
    n_neg = int((t_y == 0).sum())
    final = {
        "best_epoch": str(best["epoch"]),
        "T_star": T_best,
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
        "wilson_half_width_pp@p=0.5": wilson_hw(n_neg) * 100,
    }
    (out_dir / "final_test.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    # confusion matrix + per-class recall on test (default class order)
    pred_class = torch.softmax(t_logits / T_best, dim=-1).argmax(dim=-1).numpy()
    true_class = t_labels.numpy()
    n_cls = int(max(true_class.max(), pred_class.max())) + 1
    conf = confusion_matrix_multi(pred_class, true_class, n_cls)
    # class names guess: if 7 classes, use ALPHA ordering; else numeric
    if n_cls == len(CLASS_NAMES_ALPHA):
        names = CLASS_NAMES_ALPHA
    else:
        names = [f"class{i}" for i in range(n_cls)]
    pd.DataFrame(
        conf, index=[f"true_{c}" for c in names], columns=[f"pred_{c}" for c in names],
    ).to_csv(out_dir / "confusion_matrix_test.csv")
    # per-class recall (binary gate at tau_995)
    rows_cls = []
    pred_binary_995 = (t_p >= best["tau_995"]).astype(int) if best["tau_995"] is not None else None
    if pred_binary_995 is not None:
        for ci, cn in enumerate(names):
            if cn == "Normal":
                continue
            mask = (true_class == ci)
            if mask.sum() == 0:
                continue
            rows_cls.append({"class": cn, "n_test": int(mask.sum()),
                             "recall_at_R99.5": float(pred_binary_995[mask].mean())})
    pd.DataFrame(rows_cls).to_csv(out_dir / "per_class_recall_test.csv", index=False)

    # full tau sweep on val_op at best epoch
    # find the matching val_op file for the best epoch
    best_ep = str(best["epoch"])
    op_file = per_ep_dir / f"val_op_epoch{best_ep}.npz"
    if op_file.exists():
        op = np.load(op_file, allow_pickle=True)
        op_logits_b = torch.from_numpy(op["logits"]).float()
        op_labels_b = torch.from_numpy(op["labels"]).long()
        op_p_b = binary_prob(op_logits_b, T_best, normal_idx)
        op_y_b = binary_labels(op_labels_b, normal_idx)
        tau_sweep = full_tau_sweep(op_p_b, op_y_b, n_points=200)
        tau_sweep.to_csv(out_dir / "tau_spec_curve_val_op.csv", index=False)

    return {"summary": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--normal-idx", type=int, default=NORMAL_IDX_DEFAULT,
                    help=f"Normal class index in logits (default: {NORMAL_IDX_DEFAULT} "
                         f"for ImageFolder alphabetical ordering {CLASS_NAMES_ALPHA})")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_cap_root = args.output_dir / "per_capacity"
    per_cap_root.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    full = {}
    for cap in CAPACITIES:
        run_dir = args.runs_dir / f"yolo11{cap}"
        if not run_dir.exists():
            print(f"[SKIP] {cap}: {run_dir} missing")
            continue
        print(f"\n[process] capacity={cap}")
        out_dir = per_cap_root / f"yolo11{cap}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = process_capacity(run_dir, out_dir, args.normal_idx)
        summary = result["summary"]
        summary["capacity"] = cap
        comparison_rows.append({
            "capacity": cap,
            "best_epoch": summary.get("best_epoch"),
            "T_star": round(summary.get("T_star", 0.0), 4),
            "tau_995": round(summary.get("tau_995", 0.0) or 0.0, 6),
            "tau_990": round(summary.get("tau_990", 0.0) or 0.0, 6),
            "spec@R995_test": round(summary.get("spec@R995_test", 0.0), 4),
            "spec@R990_test": round(summary.get("spec@R990_test", 0.0), 4),
            "prec@R990_test": round(summary.get("prec@R990_test", 0.0), 4),
            "ptr@R990_test":  round(summary.get("ptr@R990_test", 0.0), 4),
            "recall@R995_test": round(summary.get("recall@R995_test", 0.0), 4),
            "test_n_negative": summary.get("test_n_negative"),
            "wilson_hw_pp": round(summary.get("wilson_half_width_pp@p=0.5", 0.0), 3),
        })
        full[cap] = summary

    if not comparison_rows:
        print("[ERROR] no runs processed"); return

    df = pd.DataFrame(comparison_rows)
    df.to_csv(args.output_dir / "capacity_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"capacities": full, "rows": comparison_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = ["# Capacity Scan Results — v3 Stage 1 (Binary Gate)\n"]
    md.append(f"Evaluated {len(comparison_rows)} capacity tier(s).\n")
    md.append("## Test-set metrics (frozen best epoch)\n")
    md.append("| capacity | best_epoch | spec@R99.5 | spec@R99.0 | prec@R99.0 | ptr@R99.0 | Wilson hw |")
    md.append("|---|---|---|---|---|---|---|")
    for r in comparison_rows:
        md.append(f"| {r['capacity']} | {r['best_epoch']} | "
                   f"{r['spec@R995_test']:.4f} | {r['spec@R990_test']:.4f} | "
                   f"{r['prec@R990_test']:.4f} | {r['ptr@R990_test']:.4f} | "
                   f"±{r['wilson_hw_pp']:.2f}pp |")
    md.append("\n## Calibration & operating point\n")
    md.append("| capacity | T* | τ99.5 | τ99.0 |")
    md.append("|---|---|---|---|")
    for r in comparison_rows:
        md.append(f"| {r['capacity']} | {r['T_star']:.3f} | {r['tau_995']:.4f} | {r['tau_990']:.4f} |")
    md.append(f"\nTest set size: {comparison_rows[0].get('test_n_negative')} negatives (natural distribution).")
    (args.output_dir / "capacity_comparison.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[DONE] summary in {args.output_dir}")


if __name__ == "__main__":
    main()
