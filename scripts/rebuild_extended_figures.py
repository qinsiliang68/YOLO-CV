"""Rebuild all HN sweep figures from extended_eval_results.csv."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "research" / "results" / "stage1_formal" / "extended_eval_results.csv"
OUT = REPO / "research" / "results" / "stage1_formal" / "gate_hn_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PAL = {"yolo11n": "#377EB8", "yolo11s": "#4DAF4A", "yolo11m": "#E41A1C",
       "yolo11l": "#FF7F00", "yolo11x": "#984EA3"}
LABELS = {"yolo11n": "n (5.4M)", "yolo11s": "s (9M)", "yolo11m": "m (20M)",
          "yolo11l": "l (24M)", "yolo11x": "x (57M)"}
RATIOS = [f"hn{r:02d}" for r in range(0, 22, 2)]
MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


def load_data():
    with open(DATA, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_specs(rows, model, metric="spec_at_r995"):
    mrows = {r["ratio"]: float(r[metric]) for r in rows if r["model"] == model}
    return [mrows.get(ratio, np.nan) for ratio in RATIOS]


# ═══════════════════════════════════════════
# FIG 1: Heatmap — 5 models × 11 ratios
# ═══════════════════════════════════════════
def build_heatmap(rows):
    delta = np.full((5, 11), np.nan)
    for i, model in enumerate(MODELS):
        specs = get_specs(rows, model)
        baseline = specs[0]
        for j in range(11):
            if not np.isnan(specs[j]):
                delta[i, j] = specs[j] - baseline

    vmax = max(0.05, np.nanmax(np.abs(delta)))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.88, bottom=0.15)
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(5):
        for j in range(11):
            v = delta[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
            else:
                sign = "+" if v > 0 else ""
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(11))
    ax.set_xticklabels(RATIOS, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels([LABELS[m] for m in MODELS])
    ax.set_xlabel("HN ratio")
    ax.set_title("ΔSpec@R99.5 relative to hn00 baseline (extended val, n=707)")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cb.set_label("ΔSpec@R99.5")

    # Mark winners
    for i, model in enumerate(MODELS):
        specs = get_specs(rows, model)
        best_j = int(np.nanargmax(specs))
        if best_j != 0 and (specs[best_j] - specs[0]) > 0.01:
            ax.plot(best_j, i, marker="*", color="gold", markersize=14, markeredgecolor="black", markeredgewidth=0.8)

    path = OUT / "fig_hn_sweep_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap: {path}")


# ═══════════════════════════════════════════
# FIG 2: Capacity curves — Spec@R99.5 vs ratio
# ═══════════════════════════════════════════
def build_capacity_curves(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(11)
    for model in MODELS:
        specs = get_specs(rows, model)
        ax1.plot(x, specs, marker="o", color=PAL[model], label=LABELS[model], linewidth=1.8, markersize=5)
        best_j = int(np.nanargmax(specs))
        ax1.plot(best_j, specs[best_j], marker="*", color=PAL[model], markersize=14, markeredgecolor="black", markeredgewidth=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(RATIOS, rotation=45, ha="right")
    ax1.set_ylabel("Spec@R99.5")
    ax1.set_xlabel("HN ratio")
    ax1.set_title("(a) Spec@R99.5 vs HN ratio")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.15)

    # Best epoch vs ratio
    for model in MODELS:
        mrows = {r["ratio"]: r for r in rows if r["model"] == model}
        epochs = []
        for ratio in RATIOS:
            r = mrows.get(ratio)
            if r:
                name = r["name"]
                ep_part = name.split("epoch_")[-1] if "epoch_" in name else name.split("epoch")[-1]
                ep = int(ep_part.replace(".pt", "").lstrip("0") or "0")
                epochs.append(ep)
            else:
                epochs.append(np.nan)
        ax2.plot(x, epochs, marker="o", color=PAL[model], label=LABELS[model], linewidth=1.4, markersize=4)
        for j, ep in enumerate(epochs):
            if ep == 1:
                ax2.plot(j, ep, marker="x", color=PAL[model], markersize=10, markeredgewidth=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(RATIOS, rotation=45, ha="right")
    ax2.set_ylabel("Gate-best epoch")
    ax2.set_xlabel("HN ratio")
    ax2.set_title("(b) Best epoch vs HN ratio")
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    path = OUT / "fig_hn_capacity_curves_panel.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Capacity curves: {path}")


# ═══════════════════════════════════════════
# FIG 3: Per-model 4-metric panels (n and m)
# ═══════════════════════════════════════════
def build_per_model_panel(rows, model, model_label):
    metrics = [
        ("spec_at_r995", "Spec@R99.5", "#355C7D"),
        ("spec_at_r990", "Spec@R99.0", "#6C5B7B"),
        ("prec_at_r990", "Prec@R99.0", "#F67280"),
        ("ptr_at_r990", "PTR@R99.0", "#C06C84"),
    ]
    x = np.arange(11)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (field, title, color) in zip(axes.flatten(), metrics):
        vals = get_specs(rows, model, field)
        ax.plot(x, vals, color=color, linewidth=1.6, marker="o", markersize=4)
        best_j = int(np.nanargmax(vals))
        ax.plot(best_j, vals[best_j], marker="*", color="gold", markersize=12, markeredgecolor="black")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.set_ylabel(title)

    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(RATIOS, rotation=45, ha="right")
        ax.set_xlabel("HN ratio")

    fig.suptitle(f"{model_label} HN sweep (extended val, n=707)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / f"fig_hn_{model.replace('yolo11','')}_ratio_metric_curves_panel.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {model_label} panel: {path}")


# ═══════════════════════════════════════════
# FIG 4: Per-model best epoch vs ratio
# ═══════════════════════════════════════════
def build_best_epoch_chart(rows, model, model_label):
    mrows = {r["ratio"]: r for r in rows if r["model"] == model}
    x = np.arange(11)
    epochs = []
    for ratio in RATIOS:
        r = mrows.get(ratio)
        if r:
            name = r["name"]
            ep_part = name.split("epoch_")[-1] if "epoch_" in name else name.split("epoch")[-1]
            ep = int(ep_part.replace(".pt", "").lstrip("0") or "0")
            epochs.append(ep)
        else:
            epochs.append(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, epochs, color=PAL[model], alpha=0.7, edgecolor="black", linewidth=0.5)
    for j, ep in enumerate(epochs):
        if ep == 1:
            ax.text(j, ep + 2, "ep=1", ha="center", fontsize=8, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(RATIOS, rotation=45, ha="right")
    ax.set_ylabel("Gate-best epoch")
    ax.set_xlabel("HN ratio")
    ax.set_title(f"{model_label}: gate-best epoch vs HN ratio (extended val)")
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    path = OUT / f"fig_hn_{model.replace('yolo11','')}_best_epoch_vs_ratio.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {model_label} best epoch: {path}")


# ═══════════════════════════════════════════
# FIG 5: Capacity scan baseline bar chart
# ═══════════════════════════════════════════
def build_capacity_bar(rows):
    baselines = []
    for model in MODELS:
        mrows = [r for r in rows if r["model"] == model and r["ratio"] == "hn00"]
        if mrows:
            baselines.append(float(mrows[0]["spec_at_r995"]))
        else:
            baselines.append(0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(5)
    bars = ax.bar(x, baselines, color=[PAL[m] for m in MODELS], edgecolor="black", linewidth=0.6, width=0.6)
    for i, (bar, val) in enumerate(zip(bars, baselines)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel("Spec@R99.5 (baseline hn00)")
    ax.set_title("Gate capacity scan baselines (extended val, n=707)")
    ax.set_ylim(0, max(baselines) * 1.15)
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    path = REPO / "research" / "results" / "stage1_formal" / "capacity_scan" / "paper_main" / "figures" / "fig_stage1_gate_capacity_bar.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Capacity bar: {path}")


# ═══════════════════════════════════════════
# FIG 6: Gain bar chart — delta per model
# ═══════════════════════════════════════════
def build_gain_bar(rows):
    deltas = []
    winners = []
    for model in MODELS:
        specs = get_specs(rows, model)
        baseline = specs[0]
        best_val = max(specs)
        best_j = int(np.nanargmax(specs))
        deltas.append(best_val - baseline)
        winners.append(RATIOS[best_j])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(5)
    bars = ax.bar(x, deltas, color=[PAL[m] for m in MODELS], edgecolor="black", linewidth=0.6, width=0.6)
    for i, (bar, d, w) in enumerate(zip(bars, deltas, winners)):
        label = f"+{d:.3f}\n({w})" if d > 0.005 else f"{d:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, max(d, 0) + 0.003, label, ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel("ΔSpec@R99.5 (best HN − baseline)")
    ax.set_title("HN gain by model capacity (extended val, n=707)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    path = OUT / "fig_hn_gain_decay_bar.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Gain bar: {path}")


def main():
    rows = load_data()
    print(f"Loaded {len(rows)} rows from {DATA}\n")

    build_heatmap(rows)
    build_capacity_curves(rows)
    for model in MODELS:
        build_per_model_panel(rows, model, LABELS[model])
        build_best_epoch_chart(rows, model, LABELS[model])
    build_capacity_bar(rows)
    build_gain_bar(rows)

    print("\nAll figures rebuilt.")


if __name__ == "__main__":
    main()
