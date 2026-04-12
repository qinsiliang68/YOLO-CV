"""
1. Complete yolo11l-cls HN sweep summary (add hn00 from capacity scan + hn06 from materials)
2. Rebuild HN heatmap and curves panel with complete l data (9 points: hn00,hn06-hn20)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, json
from pathlib import Path

def venue_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "xtick.direction": "out", "ytick.direction": "out",
    })

venue_rc()

PAL = {"n": "#377EB8", "s": "#4DAF4A", "m": "#E41A1C", "l": "#FF7F00", "x": "#984EA3"}
OUT = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal")

# ══════════════════════════════════════════════════════════
# STEP 1: Complete l sweep summary
# ══════════════════════════════════════════════════════════

def complete_l_summary():
    summary_path = OUT / "gate_hn_l_sweep/hn_sweep_summary.csv"
    rows = list(csv.DictReader(open(summary_path)))
    existing_ratios = {r["ratio_id"] for r in rows}
    header = list(rows[0].keys())

    # hn00 from capacity scan
    if "hn00" not in existing_ratios:
        cap = json.load(open(r"C:\GitHub\YOLO-CV\research\materials\stage1_formal\gate_capacity\yolo11l_gate2_formal\best_epoch_manifest.json"))
        hn00 = {
            "ratio_id": "hn00", "ratio_percent": "0", "backflow_count": "0",
            "model": "yolo11l-cls", "run_name": "yolo11l_gate2_formal_200ep",
            "best_epoch": str(cap["epoch"]),
            "best_checkpoint_path": cap["checkpoint_path"],
            "spec_at_r995": str(cap["spec_at_r995"]), "spec_at_r990": str(cap["spec_at_r990"]),
            "prec_at_r990": str(cap["prec_at_r990"]), "ptr_at_r990": str(cap["ptr_at_r990"]),
            "tau_r995": str(cap["tau_r995"]), "tau_r990": str(cap["tau_r990"]),
            "temperature_T": str(cap["temperature_T"]),
            "tn_at_r995": str(cap["tn_at_r995"]), "fn_at_r995": str(cap["fn_at_r995"]),
            "tn_at_r990": str(cap["tn_at_r990"]), "fn_at_r990": str(cap["fn_at_r990"]),
            "summary_dir": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_formal\gate_capacity\yolo11l_gate2_formal",
            "dataset_root": "", "reuse_existing": "1",
        }
        rows.insert(0, hn00)
        print("  added hn00 from capacity scan")

    # hn06 from materials
    if "hn06" not in existing_ratios:
        m06 = json.load(open(r"C:\GitHub\YOLO-CV\research\materials\stage1_formal\gate_hn_l_sweep\hn06\best_epoch_manifest.json"))
        hn06 = {
            "ratio_id": "hn06", "ratio_percent": "6", "backflow_count": "65",
            "model": "yolo11l-cls", "run_name": "yolo11l_gate2_formal_hn06_200ep",
            "best_epoch": str(m06["epoch"]),
            "best_checkpoint_path": m06["checkpoint_path"],
            "spec_at_r995": str(m06["spec_at_r995"]), "spec_at_r990": str(m06["spec_at_r990"]),
            "prec_at_r990": str(m06["prec_at_r990"]), "ptr_at_r990": str(m06["ptr_at_r990"]),
            "tau_r995": str(m06["tau_r995"]), "tau_r990": str(m06["tau_r990"]),
            "temperature_T": str(m06["temperature_T"]),
            "tn_at_r995": str(m06["tn_at_r995"]), "fn_at_r995": str(m06["fn_at_r995"]),
            "tn_at_r990": str(m06["tn_at_r990"]), "fn_at_r990": str(m06["fn_at_r990"]),
            "summary_dir": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_formal\gate_hn_l_sweep\hn06",
            "dataset_root": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_formal_gate_hn\yolo11l_gate2_formal_hn06",
            "reuse_existing": "0",
        }
        # insert after hn00
        idx = next((i for i, r in enumerate(rows) if int(r.get("ratio_percent", "0")) > 6), len(rows))
        rows.insert(idx, hn06)
        print("  added hn06 from materials")

    # sort by ratio_percent
    rows.sort(key=lambda r: int(r.get("ratio_percent", "0")))

    # write back
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"  l sweep summary updated: {len(rows)} rows → {summary_path}")
    return rows


# ══════════════════════════════════════════════════════════
# STEP 2: Build complete HN heatmap (n/s/m/l)
# ══════════════════════════════════════════════════════════

def build_hn_heatmap():
    all_ratios = list(range(0, 22, 2))  # 0,2,...,20
    ratio_labels = [f"hn{r:02d}" for r in all_ratios]
    models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]

    baselines = {"yolo11n": 0.5119, "yolo11s": 0.5238, "yolo11m": 0.5952, "yolo11l": 0.5238}

    # load from summaries
    def load_sweep(model_key):
        path = OUT / f"gate_hn_{model_key}_sweep/hn_sweep_summary.csv"
        rows = list(csv.DictReader(open(path)))
        d = {}
        for r in rows:
            ratio_pct = int(r["ratio_percent"])
            d[ratio_pct] = float(r["spec_at_r995"])
        return d

    sweeps = {
        "yolo11n": load_sweep("n"),
        "yolo11s": load_sweep("s"),
        "yolo11m": load_sweep("m"),
        "yolo11l": load_sweep("l"),
    }

    delta = np.full((4, 11), np.nan)
    for i, m in enumerate(models):
        bl = baselines[m]
        for j, r in enumerate(all_ratios):
            if r in sweeps[m]:
                delta[i, j] = sweeps[m][r] - bl

    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=0.15)

    vmax = 0.08
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(4):
        for j in range(11):
            v = delta[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                sign = "+" if v > 0 else ""
                color = "white" if abs(v) > 0.04 else "black"
                row_max = np.nanmax(delta[i, :])
                weight = "bold" if v == row_max and v > 0 else "normal"
                ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(11))
    ax.set_xticklabels(ratio_labels, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("HN ratio")
    ax.set_title("$\\Delta$Spec@R99.5 relative to hn00 baseline: "
                  "four capacity tiers under shared protocol", fontweight="bold", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$\\Delta$Spec@R99.5", fontsize=10)

    out = OUT / "gate_hn_paper/figures/fig_hn_sweep_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[heatmap] → {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# STEP 3: Build complete HN curves panel (n/s/m/l)
# ══════════════════════════════════════════════════════════

def build_hn_curves():
    all_ratios = list(range(0, 22, 2))
    models_keys = ["n", "s", "m", "l"]

    def load_sweep(key):
        path = OUT / f"gate_hn_{key}_sweep/hn_sweep_summary.csv"
        rows = list(csv.DictReader(open(path)))
        d = {}
        for r in rows:
            pct = int(r["ratio_percent"])
            d[pct] = {"spec": float(r["spec_at_r995"]), "ep": int(r["best_epoch"])}
        return d

    sweeps = {k: load_sweep(k) for k in models_keys}

    # find winners
    winners = {}
    for k in models_keys:
        best_r = max(sweeps[k], key=lambda r: sweeps[k][r]["spec"])
        winners[k] = best_r

    lw = {"n": 2.0, "s": 2.0, "m": 2.6, "l": 2.0}
    markers = {"n": "o", "s": "s", "m": "D", "l": "^"}
    global_best = 0.6429
    m_baseline = 0.5952

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.85, 1]})
    fig.subplots_adjust(wspace=0.22, left=0.07, right=0.97, top=0.88, bottom=0.13)

    for k in models_keys:
        sw = sweeps[k]
        xs = sorted(sw.keys())
        ys = [sw[r]["spec"] for r in xs]
        eps = [sw[r]["ep"] for r in xs]

        ax1.plot(xs, ys, marker=markers[k], markersize=5, color=PAL[k],
                 linewidth=lw[k], label=f"yolo11{k}-cls", zorder=2)

        # winner annotation
        wr = winners[k]
        ax1.annotate(f"hn{wr:02d}", xy=(wr, sw[wr]["spec"]),
                     xytext=(wr + 1, sw[wr]["spec"] + 0.012),
                     fontsize=8, color=PAL[k], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=PAL[k], lw=1.0))

        # best epoch panel
        ax2.plot(xs, eps, marker=markers[k], markersize=5, color=PAL[k],
                 linewidth=lw[k], label=f"yolo11{k}-cls", zorder=2)
        ep1_xs = [r for r in xs if sw[r]["ep"] == 1]
        ep1_ys = [1] * len(ep1_xs)
        if ep1_xs:
            ax2.scatter(ep1_xs, ep1_ys, marker="X", s=100, color=PAL[k],
                        edgecolors="black", linewidths=0.8, zorder=3)

    ax1.axhline(global_best, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(20.5, global_best, f"global best\nm+hn14\n({global_best})", fontsize=7, va="center", color="gray")
    ax1.axhline(m_baseline, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.text(20.5, m_baseline, f"m+hn00\n({m_baseline})", fontsize=7, va="center", color="gray")

    ax1.set_xlabel("HN ratio (%)")
    ax1.set_ylabel("Spec@R99.5")
    ax1.set_title("(a) Spec@R99.5 vs HN ratio  (higher is better)", fontweight="bold")
    ax1.set_xticks(all_ratios)
    ax1.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9, edgecolor="#CCCCCC")
    ax1.set_ylim(0.44, 0.68)

    ax2.set_xlabel("HN ratio (%)")
    ax2.set_ylabel("Best epoch")
    ax2.set_title("(b) Best epoch vs HN ratio", fontweight="bold")
    ax2.set_xticks(all_ratios)
    ax2.set_yscale("log")
    ax2.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax2.text(20.5, 1.3, "best_ep=1\n(X marks)", fontsize=7, color="gray")

    out = OUT / "gate_hn_paper/figures/fig_hn_capacity_curves_panel.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[curves] → {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# STEP 4: HN gain decay bar
# ══════════════════════════════════════════════════════════

def build_gain_decay():
    models = ["n (5.4M)", "s (10M)", "m (25M)", "l (24M)"]
    delta = [0.0714, 0.0714, 0.0476, 0.0]
    colors = [PAL["n"], PAL["s"], PAL["m"], PAL["l"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(4), delta, width=0.55, color=colors, edgecolor="black", linewidth=0.8, zorder=2)
    for i, d in enumerate(delta):
        label = f"+{d:.4f}" if d > 0 else "0"
        ax.text(i, d + 0.003, label, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("$\\Delta$Spec@R99.5 (best ratio vs hn00)")
    ax.set_title("HN gain diminishes to zero with increasing capacity\n"
                  "(under shared training protocol)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_ylim(-0.01, 0.09)

    out = OUT / "gate_hn_paper/figures/fig_hn_gain_decay_bar.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[decay] → {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Step 1: Complete l sweep summary ===")
    complete_l_summary()
    print("\n=== Step 2: HN heatmap ===")
    build_hn_heatmap()
    print("\n=== Step 3: HN curves panel ===")
    build_hn_curves()
    print("\n=== Step 4: HN gain decay ===")
    build_gain_decay()
    print("\nAll done.")
