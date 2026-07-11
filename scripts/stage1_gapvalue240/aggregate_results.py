from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from stage1_gapvalue240.aggregate import aggregate_and_write
from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.machine_assets import validate_machine_asset_report
from stage1_gapvalue240.reporting import generate_html_report, generate_markdown_report
from stage1_gapvalue240.runtime_contract import load_runtime_contract, validate_runtime_links, verify_release_identity
from stage1_gapvalue240.util import stable_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate only v1.2 VALIDATED attempts across machine output roots.")
    parser.add_argument("--machine-config", action="append", required=True)
    parser.add_argument("--output-root", action="append", default=[])
    parser.add_argument("--aggregate-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    machines = [load_machine_config(path) for path in args.machine_config]
    first = machines[0]
    repo = first.path_value("repo_root")
    runtime = load_runtime_contract(repo / "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml")
    links = validate_runtime_links(runtime, repo)
    verify_release_identity(runtime, repo)
    science = load_contract(repo / runtime.data["science_contract"]["path"])

    content_snapshots = set()
    output_roots = []
    for machine in machines:
        if machine.path_value("repo_root") != repo:
            raise RuntimeError("All aggregation machine configs must point to the same release repository")
        asset = validate_machine_asset_report(
            runtime,
            machine.path_value("machine_asset_report"),
            expected_machine_id=str(machine.data["machine_id"]),
            minimum_image_verification="existence",
        )
        content_snapshots.add(asset["content_snapshot_id"])
        output_roots.append(machine.path_value("output_root"))
    if len(content_snapshots) != 1:
        raise RuntimeError(f"Machine content snapshots differ: {sorted(content_snapshots)}")
    output_roots.extend(Path(path).resolve() for path in args.output_root)

    matrix_path = Path(links["queue"]["frozen_matrix"]["path"])
    matrix = pd.read_csv(matrix_path)
    selection_index = pd.read_csv(links["queue"]["selection_index"]["path"], dtype="string")
    selection_hashes = dict(zip(selection_index.run_slot.astype(str), selection_index.sha256.astype(str)))
    expected_per_run = {}
    for row in matrix.to_dict("records"):
        slot = str(row["run_slot"])
        expected_per_run[slot] = {
            "selection_sha256": selection_hashes[slot],
            "scientific_config_hash": stable_hash({
                "run_row": row,
                "training": science.data["training"],
                "replay": science.data["replay"],
                "calibration": science.data["calibration"],
                "evaluation_adapter": science.data["evaluation_adapter"],
            }),
        }
    expected_identity = {
        "release_ref": str(runtime.data["release"]["git_tag"]),
        "runtime_contract_sha256": runtime.sha256,
        "science_contract_file_sha256": links["science_contract"]["file_sha256"],
        "science_contract_sha256": science.sha256,
        "matrix_sha256": links["queue"]["frozen_matrix"]["sha256"],
        "selection_index_sha256": links["queue"]["selection_index"]["sha256"],
        "input_snapshot_id": next(iter(content_snapshots)),
        "dry_run": False,
    }
    aggregate_dir = Path(args.aggregate_dir).resolve() if args.aggregate_dir else output_roots[0] / "aggregate"
    report = aggregate_and_write(
        output_roots,
        matrix_path,
        aggregate_dir,
        expected_identity=expected_identity,
        expected_per_run=expected_per_run,
    )
    generate_markdown_report(aggregate_dir, aggregate_dir / "FINAL_REPORT.md")
    generate_html_report(aggregate_dir, aggregate_dir / "FINAL_REPORT.html")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
