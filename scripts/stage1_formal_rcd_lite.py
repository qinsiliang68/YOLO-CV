from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from stage1_formal_capacity_suite import (
    archive_existing_path,
    maybe_prepare_run,
    maybe_run_evaluator,
    normalize_run_dir,
    print_step,
    resolve_path,
    resolve_str,
    run_python,
    write_json,
)
from pipeline_common import REPO_ROOT, YOLOV11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal stage-1 RCD-Lite pipeline from a single anchor checkpoint.")
    parser.add_argument("--config", required=True, help="RCD-Lite config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing it.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing datasets / runs / summaries before rerunning.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["spec_at_r995"]),
            float(row["spec_at_r990"]),
            float(row["prec_at_r990"]),
            -float(row["ptr_at_r990"]),
        )

    return max(rows, key=key_fn)


def sanitize_run_dir_name(text: str) -> str:
    return text.replace("-cls", "").replace(".", "_")


def resolve_budget_count(summary_csv: Path, ratio_id: str, configured: int) -> int:
    if configured > 0:
        return configured
    rows = load_csv_rows(summary_csv)
    match = next((row for row in rows if str(row.get("ratio_id")) == ratio_id), None)
    if match is None:
        raise SystemExit(f"Missing ratio `{ratio_id}` in {summary_csv}")
    return int(match["backflow_count"])


def resolve_nonnegative_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    if value is None:
        value = default
    return max(int(value), 0)


def verify_anchor(
    *,
    hn_summary_csv: Path,
    anchor_summary_dir: Path,
    anchor_ratio_id: str,
    dry_run: bool,
    output_dir: Path,
) -> dict[str, Any]:
    rows = load_csv_rows(hn_summary_csv)
    summary_row = next((row for row in rows if str(row.get("ratio_id")) == anchor_ratio_id), None)
    best_manifest = load_json(anchor_summary_dir / "best_epoch_manifest.json")
    run_manifest = load_json(anchor_summary_dir / "run_manifest.json")
    checks = {
        "anchor_ratio_id": anchor_ratio_id,
        "hn_summary_csv": str(hn_summary_csv),
        "anchor_summary_dir": str(anchor_summary_dir),
        "summary_row_present": summary_row is not None,
        "best_epoch_match": True if summary_row is None else int(summary_row["best_epoch"]) == int(best_manifest["epoch"]),
        "spec_r995_match": True if summary_row is None else math.isclose(float(summary_row["spec_at_r995"]), float(best_manifest["spec_at_r995"]), rel_tol=0.0, abs_tol=1e-9),
        "spec_r990_match": True if summary_row is None else math.isclose(float(summary_row["spec_at_r990"]), float(best_manifest["spec_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "prec_r990_match": True if summary_row is None else math.isclose(float(summary_row["prec_at_r990"]), float(best_manifest["prec_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "ptr_r990_match": True if summary_row is None else math.isclose(float(summary_row["ptr_at_r990"]), float(best_manifest["ptr_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "tau_r995_match": True if summary_row is None else math.isclose(float(summary_row["tau_r995"]), float(best_manifest["tau_r995"]), rel_tol=0.0, abs_tol=1e-9),
        "tau_r990_match": True if summary_row is None else math.isclose(float(summary_row["tau_r990"]), float(best_manifest["tau_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "temperature_match": True if summary_row is None else math.isclose(float(summary_row["temperature_T"]), float(best_manifest["temperature_T"]), rel_tol=0.0, abs_tol=1e-9),
        "checkpoint_path": best_manifest["checkpoint_path"],
        "checkpoint_exists": Path(best_manifest["checkpoint_path"]).exists(),
        "commit_hash": run_manifest.get("commit_hash", ""),
    }
    checks["passed"] = all(
        bool(checks[key])
        for key in (
            "best_epoch_match",
            "spec_r995_match",
            "spec_r990_match",
            "prec_r990_match",
            "ptr_r990_match",
            "tau_r995_match",
            "tau_r990_match",
            "temperature_match",
        )
    ) and (checks["checkpoint_exists"] or dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "anchor_hn_sanity.json", checks)
    md_lines = [
        f"# Anchor HN Sanity Check: {anchor_ratio_id}",
        "",
        f"- best epoch match: `{checks['best_epoch_match']}`",
        f"- summary row present: `{checks['summary_row_present']}`",
        f"- Spec@R99.5 match: `{checks['spec_r995_match']}`",
        f"- Spec@R99.0 match: `{checks['spec_r990_match']}`",
        f"- Prec@R99.0 match: `{checks['prec_r990_match']}`",
        f"- PTR@R99.0 match: `{checks['ptr_r990_match']}`",
        f"- tau_R99.5 match: `{checks['tau_r995_match']}`",
        f"- tau_R99.0 match: `{checks['tau_r990_match']}`",
        f"- temperature match: `{checks['temperature_match']}`",
        f"- checkpoint exists: `{checks['checkpoint_exists']}`",
        f"- overall passed: `{checks['passed']}`",
    ]
    (output_dir / "anchor_hn_sanity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise SystemExit(f"Anchor HN sanity check failed for {anchor_ratio_id}")
    return {"best_manifest": best_manifest, "run_manifest": run_manifest, "summary_row": summary_row}


def build_train_config(
    *,
    base_checkpoint_path: str,
    dataset_root: Path,
    project_root: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    optimizer: str,
    cache: bool,
    save_period: int,
    seed: int,
    resume_value: str | bool,
) -> dict[str, Any]:
    return {
        "task": "classify",
        "model": base_checkpoint_path,
        "data": str(dataset_root),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": str(project_root),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "patience": 0,
        "optimizer": optimizer,
        "cache": cache,
        "resume": resume_value,
        "save_period": save_period,
        "seed": seed,
    }


def summarize_rcd_results(
    *,
    results_dir: Path,
    summary_dir: Path,
    anchor_best: dict[str, Any],
    anchor_ratio_id: str,
    selection_summary_path: Path,
) -> None:
    best_path = summary_dir / "best_epoch_manifest.json"
    if not best_path.exists():
        return
    rcd_best = load_json(best_path)
    selection_summary = load_json(selection_summary_path) if selection_summary_path.exists() else {}
    rows = [
        {
            "setting": f"anchor_{anchor_ratio_id}",
            "best_epoch": int(anchor_best["epoch"]),
            "spec_at_r995": float(anchor_best["spec_at_r995"]),
            "spec_at_r990": float(anchor_best["spec_at_r990"]),
            "prec_at_r990": float(anchor_best["prec_at_r990"]),
            "ptr_at_r990": float(anchor_best["ptr_at_r990"]),
            "tau_r995": float(anchor_best["tau_r995"]),
            "tau_r990": float(anchor_best["tau_r990"]),
            "temperature_T": float(anchor_best["temperature_T"]),
            "checkpoint_path": str(anchor_best["checkpoint_path"]),
        },
        {
            "setting": "rcd_lite",
            "best_epoch": int(rcd_best["epoch"]),
            "spec_at_r995": float(rcd_best["spec_at_r995"]),
            "spec_at_r990": float(rcd_best["spec_at_r990"]),
            "prec_at_r990": float(rcd_best["prec_at_r990"]),
            "ptr_at_r990": float(rcd_best["ptr_at_r990"]),
            "tau_r995": float(rcd_best["tau_r995"]),
            "tau_r990": float(rcd_best["tau_r990"]),
            "temperature_T": float(rcd_best["temperature_T"]),
            "checkpoint_path": str(rcd_best["checkpoint_path"]),
        },
    ]
    anchor_row = rows[0]
    rcd_row = rows[1]
    compare_row = {
        "anchor_setting": anchor_row["setting"],
        "candidate_setting": rcd_row["setting"],
        "anchor_best_epoch": anchor_row["best_epoch"],
        "candidate_best_epoch": rcd_row["best_epoch"],
        "delta_spec_at_r995": round(rcd_row["spec_at_r995"] - anchor_row["spec_at_r995"], 6),
        "delta_spec_at_r990": round(rcd_row["spec_at_r990"] - anchor_row["spec_at_r990"], 6),
        "delta_prec_at_r990": round(rcd_row["prec_at_r990"] - anchor_row["prec_at_r990"], 6),
        "delta_ptr_at_r990": round(rcd_row["ptr_at_r990"] - anchor_row["ptr_at_r990"], 6),
    }
    write_csv(
        results_dir / "rcd_lite_compare.csv",
        [
            "setting",
            "best_epoch",
            "spec_at_r995",
            "spec_at_r990",
            "prec_at_r990",
            "ptr_at_r990",
            "tau_r995",
            "tau_r990",
            "temperature_T",
            "checkpoint_path",
        ],
        rows,
    )
    write_json(
        results_dir / "rcd_lite_compare.json",
        {
            "anchor_ratio_id": anchor_ratio_id,
            "rows": rows,
            "compare_row": compare_row,
            "selection_summary": selection_summary,
        },
    )
    lines = [
        "# Formal Stage-1 RCD-Lite Compare",
        "",
        "| Setting | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {best_epoch} | {spec_at_r995:.6f} | {spec_at_r990:.6f} | {prec_at_r990:.6f} | {ptr_at_r990:.6f} | {tau_r995:.4f} | {tau_r990:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Fixed budget count: `{selection_summary.get('fixed_budget_count', '')}`",
            f"- Candidate pool size: `{selection_summary.get('candidate_top_k', '')}`",
            f"- Selected unique normals: `{selection_summary.get('selected_unique_count', '')}`",
            f"- Delta Spec@R99.5 vs anchor: `{compare_row['delta_spec_at_r995']:+.6f}`",
            f"- Delta Spec@R99.0 vs anchor: `{compare_row['delta_spec_at_r990']:+.6f}`",
            f"- Delta Prec@R99.0 vs anchor: `{compare_row['delta_prec_at_r990']:+.6f}`",
            f"- Delta PTR@R99.0 vs anchor: `{compare_row['delta_ptr_at_r990']:+.6f}`",
        ]
    )
    (results_dir / "rcd_lite_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_rows = [
        {
            "setting": "rcd_lite",
            "best_epoch": int(rcd_best["epoch"]),
            "best_checkpoint_path": str(rcd_best["checkpoint_path"]),
            "summary_dir": str(summary_dir),
            "spec_at_r995": float(rcd_best["spec_at_r995"]),
            "spec_at_r990": float(rcd_best["spec_at_r990"]),
            "prec_at_r990": float(rcd_best["prec_at_r990"]),
            "ptr_at_r990": float(rcd_best["ptr_at_r990"]),
        }
    ]
    write_csv(
        results_dir / "rcd_lite_best_checkpoint_registry.csv",
        [
            "setting",
            "best_epoch",
            "best_checkpoint_path",
            "summary_dir",
            "spec_at_r995",
            "spec_at_r990",
            "prec_at_r990",
            "ptr_at_r990",
        ],
        registry_rows,
    )
    write_json(results_dir / "rcd_lite_best_checkpoint_registry.json", {"rows": registry_rows})
    (results_dir / "rcd_lite_best_checkpoint_registry.md").write_text(
        "# Formal Stage-1 RCD-Lite Best Checkpoint Registry\n\n"
        f"- best epoch: `{registry_rows[0]['best_epoch']}`\n"
        f"- checkpoint: `{registry_rows[0]['best_checkpoint_path']}`\n",
        encoding="utf-8",
    )
    (results_dir / "rcd_lite_report.md").write_text(
        "\n".join(
            [
                "# Formal Stage-1 RCD-Lite Report",
                "",
                f"- Anchor ratio: `{anchor_ratio_id}`",
                f"- Anchor checkpoint: `{anchor_best['checkpoint_path']}`",
                f"- Candidate pool size: `{selection_summary.get('candidate_top_k', '')}`",
                f"- Fixed extra-normal budget: `{selection_summary.get('fixed_budget_count', '')}`",
                f"- Selected unique normals: `{selection_summary.get('selected_unique_count', '')}`",
                f"- Selected duplication total: `{selection_summary.get('duplicated_images', '')}`",
                "- Formal comparison: `anchor HN uniform budget -> RCD-Lite redistributed budget`",
                f"- Delta Spec@R99.5: `{compare_row['delta_spec_at_r995']:+.6f}`",
                f"- Delta Spec@R99.0: `{compare_row['delta_spec_at_r990']:+.6f}`",
                f"- Delta Prec@R99.0: `{compare_row['delta_prec_at_r990']:+.6f}`",
                f"- Delta PTR@R99.0: `{compare_row['delta_ptr_at_r990']:+.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_suite(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    cfg = load_json(config_path)
    print_step("task", resolve_str(cfg.get("label"), "formal gate RCD-Lite"))

    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200")
    anchor_summary_dir = resolve_path(cfg.get("anchor_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn14")
    hn_summary_csv = resolve_path(cfg.get("anchor_hn_summary_csv"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep" / "hn_sweep_summary.csv")
    project_root = resolve_path(cfg.get("project_root"), base=YOLOV11_ROOT / "runs" / "stage1_formal_gate_rcd")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "stage1_formal_gate_rcd")
    temp_config_dir = resolve_path(cfg.get("temp_config_dir"), base=REPO_ROOT / "$out" / "generated_configs" / "stage1_formal" / "gate_rcd_lite")
    summary_dir = resolve_path(cfg.get("summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_rcd_lite" / "yolo11m_gate2_formal_hn14_rcd_lite")
    results_dir = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_rcd_lite")
    score_output_dir = resolve_path(cfg.get("score_output_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_rcd_assets" / "yolo11m_hn14_rcd_scores")
    feature_output_dir = resolve_path(cfg.get("feature_output_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_rcd_assets" / "yolo11m_hn14_features")
    dataset_view_root = resolve_path(cfg.get("dataset_view_root"), base=YOLOV11_ROOT / "datasets" / "stage1_formal_gate_rcd")
    split_csv = resolve_path(cfg.get("split_csv"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv")

    anchor_ratio_id = resolve_str(cfg.get("anchor_ratio_id"), "hn14")
    base_model = resolve_str(cfg.get("base_model"), "yolo11m-cls")
    sanity_dir = results_dir / "sanity"
    anchor = verify_anchor(
        hn_summary_csv=hn_summary_csv,
        anchor_summary_dir=anchor_summary_dir,
        anchor_ratio_id=anchor_ratio_id,
        dry_run=args.dry_run,
        output_dir=sanity_dir,
    )
    anchor_best = anchor["best_manifest"]
    anchor_checkpoint_path = str(anchor_best["checkpoint_path"])
    budget_count = resolve_budget_count(hn_summary_csv, resolve_str(cfg.get("budget_source_ratio"), anchor_ratio_id), int(cfg.get("fixed_budget_count", 0) or 0))
    gallery_top_n = resolve_nonnegative_int(cfg, "gallery_top_n", 0)

    run_name_prefix = sanitize_run_dir_name(resolve_str(cfg.get("run_name_prefix"), base_model))
    dataset_name_prefix = sanitize_run_dir_name(resolve_str(cfg.get("dataset_name_prefix"), base_model))
    run_name = f"{run_name_prefix}_{anchor_ratio_id}_rcd_lite_200ep"
    dataset_root = dataset_view_root / f"{dataset_name_prefix}_{anchor_ratio_id}_rcd_lite"
    run_dir = project_root / run_name
    temp_config_path = temp_config_dir / f"{run_name}.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    if args.rerun:
        archive_existing_path(run_dir, recycle_root / "runs", dry_run=args.dry_run)
        archive_existing_path(summary_dir, recycle_root / "materials", dry_run=args.dry_run)
        archive_existing_path(results_dir, recycle_root / "results", dry_run=args.dry_run)
        archive_existing_path(dataset_root, recycle_root / "datasets", dry_run=args.dry_run)
        archive_existing_path(score_output_dir, recycle_root / "assets", dry_run=args.dry_run)
        archive_existing_path(feature_output_dir, recycle_root / "assets", dry_run=args.dry_run)

    print_step("anchor", f"{base_model} {anchor_ratio_id} -> epoch {anchor_best['epoch']} ({anchor_checkpoint_path})")
    print_step("budget", f"fixed extra-normal budget={budget_count}")

    run_python(
        "scripts/stage1_score_train_normals_rcd_lite.py",
        [
            "--weights",
            anchor_checkpoint_path,
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(score_output_dir),
            "--device",
            resolve_str(cfg.get("score_device"), resolve_str(cfg.get("device"), "0")),
            "--imgsz",
            str(int(cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(cfg.get("score_batch", 1) or 1)),
            "--chunk-size",
            str(int(cfg.get("score_chunk_size", 16) or 16)),
            "--top-k",
            str(int(cfg.get("candidate_top_k", 250) or 250)),
            "--temperature",
            str(float(anchor_best["temperature_T"])),
            "--normal-class",
            resolve_str(cfg.get("normal_class"), "Normal"),
            "--gallery-top-n",
            str(gallery_top_n),
        ],
        dry_run=args.dry_run,
    )

    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            anchor_checkpoint_path,
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(feature_output_dir),
            "--device",
            resolve_str(cfg.get("device"), "0"),
            "--imgsz",
            str(int(cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(cfg.get("feature_batch", 4) or 4)),
            "--chunk-size",
            str(int(cfg.get("feature_chunk_size", 16) or 16)),
            "--normal-class",
            resolve_str(cfg.get("normal_class"), "Normal"),
            "--splits",
            "train",
        ],
        dry_run=args.dry_run,
    )

    run_python(
        "scripts/stage1_build_rcd_lite_dataset.py",
        [
            "--source-dataset",
            str(source_dataset),
            "--scores-csv",
            str(score_output_dir / "train_normal_rcd_scores.csv"),
            "--train-features-csv",
            str(feature_output_dir / "train_features.csv"),
            "--train-embeddings-npy",
            str(feature_output_dir / "train_embeddings.npy"),
            "--anchor-best-manifest",
            str(anchor_summary_dir / "best_epoch_manifest.json"),
            "--output-dataset",
            str(dataset_root),
            "--candidate-top-k",
            str(int(cfg.get("candidate_top_k", 250) or 250)),
            "--fixed-budget-count",
            str(budget_count),
            "--r-sigma",
            str(float(cfg.get("r_sigma", 0.03) or 0.03)),
            "--r-beta",
            str(float(cfg.get("r_beta", 0.75) or 0.75)),
            "--c-sigma",
            str(float(cfg.get("c_sigma", 0.85) or 0.85)),
            "--rknn-k",
            str(int(cfg.get("rknn_k", 10) or 10)),
            "--kappa",
            str(float(cfg.get("distribution_kappa", 2.0) or 2.0)),
            "--gallery-top-n",
            str(gallery_top_n),
            "--link-mode",
            resolve_str(cfg.get("link_mode"), "hardlink"),
            "--seed",
            str(int(cfg.get("seed", 20260330) or 20260330)),
        ],
        dry_run=args.dry_run,
    )

    normalize_run_dir(run_dir, dry_run=args.dry_run)
    if not args.dry_run:
        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_rcd_lite"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            batch=int(cfg.get("batch", 24) or 24),
            epochs=int(cfg.get("epochs", 200) or 200),
            status="planned",
            dry_run=False,
        )

        maybe_run_evaluator(
            task_kind="gate",
            run_dir=run_dir,
            data_root=dataset_root,
            summary_dir=summary_dir,
            split_csv=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            device=resolve_str(cfg.get("device"), "0"),
            eval_batch=int(cfg.get("eval_batch", cfg.get("batch", 24)) or 24),
            imgsz=int(cfg.get("imgsz", 640) or 640),
            dry_run=False,
        )

    resume_value: str | bool = False
    if not args.dry_run:
        last_checkpoint = run_dir / "weights" / "last.pt"
        if bool(cfg.get("resume", True)) and last_checkpoint.exists():
            resume_value = str(last_checkpoint)
            print_step("resume", f"{run_name}: resume from {last_checkpoint}")

        train_cfg = build_train_config(
            base_checkpoint_path=anchor_checkpoint_path,
            dataset_root=dataset_root,
            project_root=project_root,
            run_name=run_name,
            epochs=int(cfg.get("epochs", 200) or 200),
            imgsz=int(cfg.get("imgsz", 640) or 640),
            batch=int(cfg.get("batch", 24) or 24),
            workers=int(cfg.get("workers", 4) or 4),
            device=resolve_str(cfg.get("device"), "0"),
            optimizer=resolve_str(cfg.get("optimizer"), "auto"),
            cache=bool(cfg.get("cache", False)),
            save_period=int(cfg.get("save_period", 1) or 1),
            seed=int(cfg.get("seed", 20260330) or 20260330),
            resume_value=resume_value,
        )
        write_json(temp_config_path, train_cfg)

        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_rcd_lite"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            batch=int(cfg.get("batch", 24) or 24),
            epochs=int(cfg.get("epochs", 200) or 200),
            status="training_started",
            dry_run=False,
        )

        run_python(
            "scripts/stage1_gate_train.py",
            [
                "--config",
                str(temp_config_path),
                "--stdout-log",
                str(stdout_log),
                "--stderr-log",
                str(stderr_log),
            ],
            dry_run=False,
        )

        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_rcd_lite"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            batch=int(cfg.get("batch", 24) or 24),
            epochs=int(cfg.get("epochs", 200) or 200),
            status="training_completed",
            dry_run=False,
        )

    maybe_run_evaluator(
        task_kind="gate",
        run_dir=run_dir,
        data_root=dataset_root,
        summary_dir=summary_dir,
        split_csv=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        device=resolve_str(cfg.get("device"), "0"),
        eval_batch=int(cfg.get("eval_batch", cfg.get("batch", 24)) or 24),
        imgsz=int(cfg.get("imgsz", 640) or 640),
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_rcd_lite"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            batch=int(cfg.get("batch", 24) or 24),
            epochs=int(cfg.get("epochs", 200) or 200),
            status="evaluation_completed",
            dry_run=False,
        )

        summarize_rcd_results(
            results_dir=results_dir,
            summary_dir=summary_dir,
            anchor_best=anchor_best,
            anchor_ratio_id=anchor_ratio_id,
            selection_summary_path=dataset_root / "rcd_sampling_summary.json",
        )


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
