"""
aggregate_capacity_results.py — combine per-capacity final_test_metrics.json into
a single comparison table. Run on the aggregator machine after all 4 machines
upload their results.

Usage:
    uv run python scripts/aggregate_capacity_results.py \\
        --runs-dir /path/to/runs \\
        --output-dir /path/to/summary

Expects:
    runs-dir/yolo11{n,s,m,l,x}/final_test_metrics.json
    runs-dir/yolo11{n,s,m,l,x}/per_epoch_metrics.csv
    runs-dir/yolo11{n,s,m,l,x}/best_epoch.json

Outputs:
    output-dir/capacity_comparison.csv   5 rows, columns: capacity, best_epoch,
                                                T_star, tau_995, tau_990,
                                                spec@R995_test, spec@R990_test,
                                                wilson_half_width_pp
    output-dir/capacity_comparison.md    human-readable summary
    output-dir/summary.json              full JSON combining all 5 final_test_metrics
"""
import argparse
import json
from pathlib import Path

import pandas as pd

CAPACITIES = ["n", "s", "m", "l", "x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_results = {}
    for cap in CAPACITIES:
        cap_dir = runs_dir / f"yolo11{cap}"
        meta_file = cap_dir / "final_test_metrics.json"
        if not meta_file.exists():
            print(f"[SKIP] {cap}: {meta_file} not found")
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        rows.append({
            "capacity": cap,
            "best_epoch": meta.get("best_epoch"),
            "T_star": round(meta.get("T_star", 0.0), 4),
            "tau_995": round(meta.get("tau_995", 0.0) or 0.0, 6),
            "tau_990": round(meta.get("tau_990", 0.0) or 0.0, 6),
            "spec@R995_test": round(meta.get("spec@R995_test", 0.0), 4),
            "spec@R990_test": round(meta.get("spec@R990_test", 0.0), 4),
            "prec@R990_test": round(meta.get("prec@R990_test", 0.0), 4),
            "ptr@R990_test":  round(meta.get("ptr@R990_test", 0.0), 4),
            "recall@R995_test": round(meta.get("recall@R995_test", 0.0), 4),
            "recall@R990_test": round(meta.get("recall@R990_test", 0.0), 4),
            "test_n": meta.get("test_n"),
            "test_n_negative": meta.get("test_n_negative"),
            "wilson_hw_pp": round(meta.get("wilson_half_width_pp@p=0.5", 0.0), 3),
        })
        full_results[cap] = meta

    if not rows:
        print("[ERROR] no results found")
        return

    df = pd.DataFrame(rows)
    csv_path = out_dir / "capacity_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")

    # Markdown summary
    md = ["# Capacity Scan Results — v3 Stage 1 (Binary Gate)\n"]
    md.append(f"Evaluated {len(rows)} capacity tier{'s' if len(rows) != 1 else ''}.\n")
    md.append("## Test-set metrics (frozen best epoch)\n")
    md.append("| capacity | best_epoch | spec@R99.5 | spec@R99.0 | prec@R99.0 | ptr@R99.0 | Wilson hw |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['capacity']} | {r['best_epoch']} | "
            f"{r['spec@R995_test']:.4f} | {r['spec@R990_test']:.4f} | "
            f"{r['prec@R990_test']:.4f} | {r['ptr@R990_test']:.4f} | "
            f"±{r['wilson_hw_pp']:.2f}pp |"
        )
    md.append("")
    md.append("## Calibration & operating point\n")
    md.append("| capacity | T* | τ99.5 | τ99.0 |")
    md.append("|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['capacity']} | {r['T_star']:.3f} | {r['tau_995']:.4f} | {r['tau_990']:.4f} |")
    md.append("")
    md.append("Test set size: {} frames ({} negative).".format(
        rows[0].get("test_n"), rows[0].get("test_n_negative")))
    md_path = out_dir / "capacity_comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[write] {md_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"capacities": full_results, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[write] {summary_path}")

    print(f"\n[DONE] summary in {out_dir}")


if __name__ == "__main__":
    main()
