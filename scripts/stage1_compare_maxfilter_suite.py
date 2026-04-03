from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stage-1 max-filter experiments under the shared PTSG protocol.")
    parser.add_argument("--baseline-dir", required=True, help="Baseline PTSG directory.")
    parser.add_argument("--baseline-label", default="H0 current best hn02 + P2", help="Baseline row label.")
    parser.add_argument("--output-dir", required=True, help="Output directory for suite summary.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment formatted as label::material_dir",
    )
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def choose_best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    def key_fn(row: dict[str, str]) -> tuple[float, float, float, float]:
        return (
            as_float(row.get("spec_at_r995")),
            as_float(row.get("spec_at_r990")),
            as_float(row.get("prec_at_r990")),
            -as_float(row.get("ptr_at_r990")),
        )

    return max(rows, key=key_fn)


def select_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def load_counts(material_dir: Path, variant: str) -> dict[str, Any]:
    payload = load_json(material_dir / variant.lower() / "threshold_summary.json")
    ops = payload["operating_points"]
    return {"r995": ops["recall_ge_99_5"], "r990": ops["recall_ge_99_0"]}


def build_row(label: str, material_dir: Path) -> dict[str, Any]:
    summary_rows = load_csv_rows(material_dir / "ptsg_summary.csv")
    best_row = choose_best_row(summary_rows)
    best_counts = load_counts(material_dir, str(best_row["variant"]))
    p0_row = select_variant(summary_rows, "P0")
    p2_row = select_variant(summary_rows, "P2")
    return {
        "label": label,
        "material_dir": str(material_dir),
        "best_variant": best_row["variant"],
        "best_spec_at_r995": best_row["spec_at_r995"],
        "best_spec_at_r990": best_row["spec_at_r990"],
        "best_prec_at_r990": best_row["prec_at_r990"],
        "best_ptr_at_r990": best_row["ptr_at_r990"],
        "p0_spec_at_r995": p0_row.get("spec_at_r995", ""),
        "p0_spec_at_r990": p0_row.get("spec_at_r990", ""),
        "p2_spec_at_r995": p2_row.get("spec_at_r995", ""),
        "p2_spec_at_r990": p2_row.get("spec_at_r990", ""),
        "tn_at_r995": best_counts["r995"]["tn"],
        "fn_at_r995": best_counts["r995"]["fn"],
        "tn_at_r990": best_counts["r990"]["tn"],
        "fn_at_r990": best_counts["r990"]["fn"],
    }


def choose_best_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            as_float(row["best_spec_at_r995"]),
            as_float(row["best_spec_at_r990"]),
            as_float(row["best_prec_at_r990"]),
            -as_float(row["best_ptr_at_r990"]),
        )

    return max(rows, key=key_fn)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_row(args.baseline_label, Path(args.baseline_dir).resolve())]
    for item in args.experiment:
        label, _, path_text = item.partition("::")
        if not label or not path_text:
            raise SystemExit(f"Invalid --experiment value: {item}")
        rows.append(build_row(label, Path(path_text).resolve()))

    best = choose_best_experiment(rows)
    verdict = "new_experiment_wins" if best["label"] != args.baseline_label else "baseline_still_best"

    write_csv(output_dir / "stage1_maxfilter_suite_summary.csv", rows)
    (output_dir / "stage1_maxfilter_suite_summary.json").write_text(
        json.dumps(
            {
                "best_label": best["label"],
                "verdict": verdict,
                "ranking_rule": [
                    "Spec@R99.5 descending",
                    "Spec@R99.0 descending",
                    "Prec@R99.0 descending",
                    "PTR@R99.0 ascending",
                ],
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage-1 Max-Filter Suite Summary",
        "",
        "| Label | Best Variant | Best Spec@R99.5 | Best Spec@R99.0 | Best Prec@R99.0 | Best PTR@R99.0 | P0 Spec@R99.5 | P2 Spec@R99.5 | TN@R99.5 | FN@R99.5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {best_variant} | {best_spec_at_r995} | {best_spec_at_r990} | {best_prec_at_r990} | {best_ptr_at_r990} | {p0_spec_at_r995} | {p2_spec_at_r995} | {tn_at_r995} | {fn_at_r995} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Best row: `{best['label']}`",
            f"- Verdict: `{verdict}`",
        ]
    )
    (output_dir / "stage1_maxfilter_suite_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
