from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare H0/H1/H2 for stage-1 strong-embedding trust gate.")
    parser.add_argument("--baseline-dir", required=True, help="Existing baseline PTSG directory containing ptsg_summary.csv.")
    parser.add_argument("--candidate-dir", required=True, help="New contrastive PTSG directory containing ptsg_summary.csv.")
    parser.add_argument("--output-dir", required=True, help="Output directory for H0/H1/H2 comparison.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def select_variant(summary_rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    for row in summary_rows:
        if row.get("variant") == variant:
            return row
    raise SystemExit(f"Variant '{variant}' not found in summary.")


def load_operating_counts(root: Path, variant: str) -> dict[str, Any]:
    payload = load_json(root / variant.lower() / "threshold_summary.json")
    ops = payload["operating_points"]
    return {"r995": ops["recall_ge_99_5"], "r990": ops["recall_ge_99_0"]}


def build_row(group: str, description: str, summary_row: dict[str, str], counts: dict[str, Any]) -> dict[str, Any]:
    r995 = counts["r995"]
    r990 = counts["r990"]
    return {
        "group": group,
        "description": description,
        "spec_at_r995": summary_row["spec_at_r995"],
        "spec_at_r990": summary_row["spec_at_r990"],
        "prec_at_r990": summary_row["prec_at_r990"],
        "ptr_at_r990": summary_row["ptr_at_r990"],
        "threshold_at_r995": summary_row["threshold_at_r995"],
        "threshold_at_r990": summary_row["threshold_at_r990"],
        "tn_at_r995": r995["tn"],
        "fp_at_r995": r995["fp"],
        "fn_at_r995": r995["fn"],
        "tp_at_r995": r995["tp"],
        "tn_at_r990": r990["tn"],
        "fp_at_r990": r990["fp"],
        "fn_at_r990": r990["fn"],
        "tp_at_r990": r990["tp"],
    }


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            as_float(row["spec_at_r995"]),
            as_float(row["spec_at_r990"]),
            as_float(row["prec_at_r990"]),
            -as_float(row["ptr_at_r990"]),
        )

    return max(rows, key=score_key)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_csv_rows(baseline_dir / "ptsg_summary.csv")
    candidate_summary = load_csv_rows(candidate_dir / "ptsg_summary.csv")

    rows = [
        build_row(
            "H0",
            "current best yolo11l-cls + hn02 + P2",
            select_variant(baseline_summary, "P2"),
            load_operating_counts(baseline_dir, "P2"),
        ),
        build_row(
            "H1",
            "contrastive backbone + calibration + plain score",
            select_variant(candidate_summary, "P0"),
            load_operating_counts(candidate_dir, "P0"),
        ),
        build_row(
            "H2",
            "contrastive backbone + calibration + trust gate",
            select_variant(candidate_summary, "P2"),
            load_operating_counts(candidate_dir, "P2"),
        ),
    ]

    best_row = choose_best(rows)
    verdict = "worth_continue" if best_row["group"] == "H2" else "no_clear_gain"
    verdict_detail = (
        "H2 clearly beats H0 under the stage-1 ranking rule."
        if best_row["group"] == "H2"
        else "H2 does not clearly beat H0; stage-1 strong-embedding route is not yet worth extending."
    )

    write_csv(output_dir / "embedding_gate_summary.csv", rows)
    (output_dir / "embedding_gate_summary.json").write_text(
        json.dumps(
            {
                "best_group": best_row["group"],
                "verdict": verdict,
                "verdict_detail": verdict_detail,
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
        "# Stage-1 Strong-Embedding Gate Summary",
        "",
        "| Group | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {description} | {spec_at_r995} | {spec_at_r990} | {prec_at_r990} | {ptr_at_r990} | {tn_at_r995} | {fn_at_r995} | {tn_at_r990} | {fn_at_r990} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Best group: `{best_row['group']}`",
            f"- Verdict: `{verdict}`",
            f"- Detail: {verdict_detail}",
        ]
    )
    (output_dir / "embedding_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
