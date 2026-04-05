from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
    parser = argparse.ArgumentParser(description="Run a formal stage-1 HN sweep under the calibrated gate protocol.")
    parser.add_argument("--config", required=True, help="HN sweep config path.")
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


def model_stem(model_name: str) -> str:
    return model_name.replace("-cls", "")


def sanitize_run_dir_name(text: str) -> str:
    return text.replace("-cls", "").replace(".", "_")


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["spec_at_r995"]),
            float(row["spec_at_r990"]),
            float(row["prec_at_r990"]),
            -float(row["ptr_at_r990"]),
        )

    return max(rows, key=key_fn)


def sync_reference_dir(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def count_scored_normals(scores_csv: Path) -> int:
    return len(load_csv_rows(scores_csv))


def ratio_id(percent: int) -> str:
    return f"hn{percent:02d}"


def backflow_count(total_normals: int, ratio_percent: int) -> int:
    if ratio_percent <= 0:
        return 0
    return int(round(total_normals * ratio_percent / 100.0))


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


def build_reference_row(
    *,
    model_name: str,
    ratio_percent: int,
    count: int,
    base_summary_dir: Path,
    base_best: dict[str, Any],
    base_run_manifest: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    return {
        "ratio_id": ratio_id(ratio_percent),
        "ratio_percent": ratio_percent,
        "backflow_count": count,
        "model": model_name,
        "run_name": Path(base_run_manifest.get("run_dir", model_name)).name,
        "best_epoch": int(base_best["epoch"]),
        "best_checkpoint_path": base_best["checkpoint_path"],
        "spec_at_r995": float(base_best["spec_at_r995"]),
        "spec_at_r990": float(base_best["spec_at_r990"]),
        "prec_at_r990": float(base_best["prec_at_r990"]),
        "ptr_at_r990": float(base_best["ptr_at_r990"]),
        "tau_r995": float(base_best["tau_r995"]),
        "tau_r990": float(base_best["tau_r990"]),
        "temperature_T": float(base_best["temperature_T"]),
        "tn_at_r995": int(base_best["tn_at_r995"]),
        "fn_at_r995": int(base_best["fn_at_r995"]),
        "tn_at_r990": int(base_best["tn_at_r990"]),
        "fn_at_r990": int(base_best["fn_at_r990"]),
        "summary_dir": str(base_summary_dir),
        "dataset_root": str(dataset_root),
        "reuse_existing": 1,
    }


def summarize_hn_sweep(
    *,
    materials_root: Path,
    results_dir: Path,
    model_name: str,
    label: str,
) -> None:
    if not materials_root.exists():
        return
    rows: list[dict[str, Any]] = []
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        ratio_manifest_path = summary_dir / "hn_ratio_manifest.json"
        best_path = summary_dir / "best_epoch_manifest.json"
        if not ratio_manifest_path.exists() or not best_path.exists():
            continue
        ratio_meta = load_json(ratio_manifest_path)
        best = load_json(best_path)
        rows.append(
            {
                "ratio_id": ratio_meta["ratio_id"],
                "ratio_percent": int(ratio_meta["ratio_percent"]),
                "backflow_count": int(ratio_meta["backflow_count"]),
                "model": model_name,
                "run_name": ratio_meta["run_name"],
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_path": best["checkpoint_path"],
                "spec_at_r995": float(best["spec_at_r995"]),
                "spec_at_r990": float(best["spec_at_r990"]),
                "prec_at_r990": float(best["prec_at_r990"]),
                "ptr_at_r990": float(best["ptr_at_r990"]),
                "tau_r995": float(best["tau_r995"]),
                "tau_r990": float(best["tau_r990"]),
                "temperature_T": float(best["temperature_T"]),
                "tn_at_r995": int(best["tn_at_r995"]),
                "fn_at_r995": int(best["fn_at_r995"]),
                "tn_at_r990": int(best["tn_at_r990"]),
                "fn_at_r990": int(best["fn_at_r990"]),
                "summary_dir": str(summary_dir),
                "dataset_root": ratio_meta["dataset_root"],
                "reuse_existing": int(ratio_meta.get("reuse_existing", 0)),
            }
        )

    if not rows:
        return

    rows = sorted(rows, key=lambda item: (int(item["ratio_percent"]), item["ratio_id"]))
    best_row = choose_best(rows)
    ranking_rule = [
        "Spec@R99.5 descending",
        "Spec@R99.0 descending",
        "Prec@R99.0 descending",
        "PTR@R99.0 ascending",
    ]

    fieldnames = [
        "ratio_id",
        "ratio_percent",
        "backflow_count",
        "model",
        "run_name",
        "best_epoch",
        "best_checkpoint_path",
        "spec_at_r995",
        "spec_at_r990",
        "prec_at_r990",
        "ptr_at_r990",
        "tau_r995",
        "tau_r990",
        "temperature_T",
        "tn_at_r995",
        "fn_at_r995",
        "tn_at_r990",
        "fn_at_r990",
        "summary_dir",
        "dataset_root",
        "reuse_existing",
    ]
    write_csv(results_dir / "hn_sweep_summary.csv", fieldnames, rows)
    write_json(
        results_dir / "hn_sweep_summary.json",
        {
            "label": label,
            "model": model_name,
            "ranking_rule": ranking_rule,
            "best_row": best_row,
            "rows": rows,
        },
    )

    md_lines = [
        f"# {label}",
        "",
        "| Ratio | Backflow Count | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {ratio_id} | {backflow_count} | {best_epoch} | {spec_at_r995:.6f} | {spec_at_r990:.6f} | {prec_at_r990:.6f} | {ptr_at_r990:.6f} | {tau_r995:.4f} | {tau_r990:.4f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            f"- Best ratio: `{best_row['ratio_id']}`",
            f"- Best epoch: `{best_row['best_epoch']}`",
            f"- Ranking rule: `{ ' -> '.join(ranking_rule) }`",
        ]
    )
    (results_dir / "hn_sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def verify_base_capacity(
    *,
    summary_csv: Path,
    base_summary_dir: Path,
    model_name: str,
    dry_run: bool,
    output_dir: Path,
) -> dict[str, Any]:
    rows = load_csv_rows(summary_csv)
    model_row = next((row for row in rows if str(row.get("model")) == model_name), None)
    if model_row is None:
        raise SystemExit(f"Missing model row for {model_name} in {summary_csv}")

    best_manifest = load_json(base_summary_dir / "best_epoch_manifest.json")
    run_manifest = load_json(base_summary_dir / "run_manifest.json")

    checks = {
        "model": model_name,
        "summary_csv": str(summary_csv),
        "base_summary_dir": str(base_summary_dir),
        "best_epoch_match": int(model_row["best_epoch"]) == int(best_manifest["epoch"]),
        "spec_r995_match": math.isclose(float(model_row["spec_at_r995"]), float(best_manifest["spec_at_r995"]), rel_tol=0.0, abs_tol=1e-9),
        "spec_r990_match": math.isclose(float(model_row["spec_at_r990"]), float(best_manifest["spec_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "prec_r990_match": math.isclose(float(model_row["prec_at_r990"]), float(best_manifest["prec_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "ptr_r990_match": math.isclose(float(model_row["ptr_at_r990"]), float(best_manifest["ptr_at_r990"]), rel_tol=0.0, abs_tol=1e-9),
        "tau_r995_match": math.isclose(float(model_row["tau_r995"]), float(best_manifest["tau_r995"]), rel_tol=0.0, abs_tol=1e-9),
        "tau_r990_match": math.isclose(float(model_row["tau_r990"]), float(best_manifest["tau_r990"]), rel_tol=0.0, abs_tol=1e-9),
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
        )
    ) and (checks["checkpoint_exists"] or dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "base_capacity_sanity.json", checks)
    md_lines = [
        f"# Base Capacity Sanity Check: {model_name}",
        "",
        f"- best epoch match: `{checks['best_epoch_match']}`",
        f"- Spec@R99.5 match: `{checks['spec_r995_match']}`",
        f"- Spec@R99.0 match: `{checks['spec_r990_match']}`",
        f"- Prec@R99.0 match: `{checks['prec_r990_match']}`",
        f"- PTR@R99.0 match: `{checks['ptr_r990_match']}`",
        f"- tau_R99.5 match: `{checks['tau_r995_match']}`",
        f"- tau_R99.0 match: `{checks['tau_r990_match']}`",
        f"- checkpoint exists: `{checks['checkpoint_exists']}`",
        f"- overall passed: `{checks['passed']}`",
    ]
    (output_dir / "base_capacity_sanity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise SystemExit(f"Base capacity sanity check failed for {model_name}")
    return {"best_manifest": best_manifest, "run_manifest": run_manifest}


def maybe_prepare_existing_reference(
    *,
    base_summary_dir: Path,
    target_dir: Path,
    ratio_percent: int,
    count: int,
    model_name: str,
    source_dataset: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print_step("reference", f"{base_summary_dir} -> {target_dir} (hn00)")
        return
    sync_reference_dir(base_summary_dir, target_dir)
    write_json(
        target_dir / "hn_ratio_manifest.json",
        {
            "ratio_id": ratio_id(ratio_percent),
            "ratio_percent": ratio_percent,
            "backflow_count": count,
            "model": model_name,
            "run_name": Path(load_json(target_dir / "run_manifest.json").get("run_dir", target_dir.name)).name,
            "dataset_root": str(source_dataset),
            "reuse_existing": True,
        },
    )


def run_ratio(
    *,
    config: dict[str, Any],
    ratio_percent: int,
    count: int,
    base_checkpoint_path: str,
    source_dataset: Path,
    materials_root: Path,
    project_root: Path,
    recycle_root: Path,
    temp_config_dir: Path,
    dataset_view_root: Path,
    split_csv: Path,
    train_scores_csv: Path,
    dry_run: bool,
    rerun: bool,
) -> None:
    model_name = str(config["base_model"])
    rid = ratio_id(ratio_percent)
    run_prefix = sanitize_run_dir_name(resolve_str(config.get("run_name_prefix"), model_stem(model_name)))
    dataset_prefix = sanitize_run_dir_name(resolve_str(config.get("dataset_name_prefix"), model_stem(model_name)))
    run_name = f"{run_prefix}_{rid}_200ep"
    summary_dir = materials_root / rid
    run_dir = project_root / run_name
    dataset_root = source_dataset if ratio_percent == 0 else dataset_view_root / f"{dataset_prefix}_{rid}"
    temp_config_path = temp_config_dir / f"{run_name}.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    if rerun:
        archive_existing_path(run_dir, recycle_root / "runs", dry_run=dry_run)
        archive_existing_path(summary_dir, recycle_root / "materials", dry_run=dry_run)
        if ratio_percent > 0:
            archive_existing_path(dataset_root, recycle_root / "datasets", dry_run=dry_run)

    if ratio_percent > 0:
        run_python(
            "scripts/stage1_build_hn_dataset.py",
            [
                "--source-dataset",
                str(source_dataset),
                "--scores-csv",
                str(train_scores_csv),
                "--output-dataset",
                str(dataset_root),
                "--top-k",
                str(count),
                "--repeat",
                str(int(config.get("repeat", 1) or 1)),
                "--link-mode",
                resolve_str(config.get("link_mode"), "hardlink"),
            ],
            dry_run=dry_run,
        )

    if not dry_run and not dataset_root.exists():
        raise SystemExit(f"Dataset view was not created: {dataset_root}")

    if not dry_run:
        write_json(
            summary_dir / "hn_ratio_manifest.json",
            {
                "ratio_id": rid,
                "ratio_percent": ratio_percent,
                "backflow_count": count,
                "model": model_name,
                "run_name": run_name,
                "dataset_root": str(dataset_root),
                "reuse_existing": False,
            },
        )

    normalize_run_dir(run_dir, dry_run=dry_run)

    if not dry_run:
        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(config.get("task_name"), "stage1_formal_gate_hn_sweep"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(config.get("normal_class"), "Normal"),
            batch=int(config.get("batch", 24) or 24),
            epochs=int(config.get("epochs", 200) or 200),
            status="planned",
            dry_run=False,
        )

        maybe_run_evaluator(
            task_kind="gate",
            run_dir=run_dir,
            data_root=dataset_root,
            summary_dir=summary_dir,
            split_csv=split_csv,
            normal_class=resolve_str(config.get("normal_class"), "Normal"),
            device=resolve_str(config.get("device"), "0"),
            eval_batch=int(config.get("eval_batch", config.get("batch", 24)) or 24),
            imgsz=int(config.get("imgsz", 640) or 640),
            dry_run=False,
        )

    max_epoch = 0
    if not dry_run:
        last_checkpoint = run_dir / "weights" / "last.pt"
        resume_value: str | bool = False
        if bool(config.get("resume", True)) and last_checkpoint.exists():
            resume_value = str(last_checkpoint)
            print_step("resume", f"{run_name}: resume from {last_checkpoint}")

        train_cfg = build_train_config(
            base_checkpoint_path=base_checkpoint_path,
            dataset_root=dataset_root,
            project_root=project_root,
            run_name=run_name,
            epochs=int(config.get("epochs", 200) or 200),
            imgsz=int(config.get("imgsz", 640) or 640),
            batch=int(config.get("batch", 24) or 24),
            workers=int(config.get("workers", 4) or 4),
            device=resolve_str(config.get("device"), "0"),
            optimizer=resolve_str(config.get("optimizer"), "auto"),
            cache=bool(config.get("cache", False)),
            save_period=int(config.get("save_period", 1) or 1),
            seed=int(config.get("seed", 20260330) or 20260330),
            resume_value=resume_value,
        )
        write_json(temp_config_path, train_cfg)

        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(config.get("task_name"), "stage1_formal_gate_hn_sweep"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(config.get("normal_class"), "Normal"),
            batch=int(config.get("batch", 24) or 24),
            epochs=int(config.get("epochs", 200) or 200),
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
            task_name=resolve_str(config.get("task_name"), "stage1_formal_gate_hn_sweep"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(config.get("normal_class"), "Normal"),
            batch=int(config.get("batch", 24) or 24),
            epochs=int(config.get("epochs", 200) or 200),
            status="training_completed",
            dry_run=False,
        )

    maybe_run_evaluator(
        task_kind="gate",
        run_dir=run_dir,
        data_root=dataset_root,
        summary_dir=summary_dir,
        split_csv=split_csv,
        normal_class=resolve_str(config.get("normal_class"), "Normal"),
        device=resolve_str(config.get("device"), "0"),
        eval_batch=int(config.get("eval_batch", config.get("batch", 24)) or 24),
        imgsz=int(config.get("imgsz", 640) or 640),
        dry_run=dry_run,
    )

    if not dry_run:
        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(config.get("task_name"), "stage1_formal_gate_hn_sweep"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(config.get("normal_class"), "Normal"),
            batch=int(config.get("batch", 24) or 24),
            epochs=int(config.get("epochs", 200) or 200),
            status="evaluation_completed",
            dry_run=False,
        )


def run_suite(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    cfg = load_json(config_path)
    print_step("task", resolve_str(cfg.get("label"), "formal gate HN sweep"))

    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200")
    base_summary_dir = resolve_path(cfg.get("base_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_capacity")
    base_summary_csv = resolve_path(cfg.get("base_summary_csv"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_capacity" / "binary_gate_capacity_summary.csv")
    materials_root = resolve_path(cfg.get("materials_root"), base=REPO_ROOT / "research" / "materials" / "stage1_formal")
    results_dir = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal")
    project_root = resolve_path(cfg.get("project_root"), base=YOLOV11_ROOT / "runs" / "stage1_formal_gate_hn")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "stage1_formal_gate_hn")
    temp_config_dir = resolve_path(cfg.get("temp_config_dir"), base=REPO_ROOT / "$out" / "generated_configs" / "stage1_formal" / "gate_hn")
    split_csv = resolve_path(cfg.get("split_csv"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv")
    scoring_output_dir = resolve_path(cfg.get("scoring_output_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_assets")
    dataset_view_root = resolve_path(cfg.get("dataset_view_root"), base=YOLOV11_ROOT / "datasets" / "stage1_formal_gate_hn")

    base_model = resolve_str(cfg.get("base_model"), "")
    if not base_model:
        raise SystemExit("HN suite config must define 'base_model'.")

    sanity_dir = results_dir / "sanity"
    sanity = verify_base_capacity(
        summary_csv=base_summary_csv,
        base_summary_dir=base_summary_dir,
        model_name=base_model,
        dry_run=args.dry_run,
        output_dir=sanity_dir,
    )
    base_best = sanity["best_manifest"]
    base_run_manifest = sanity["run_manifest"]
    base_checkpoint_path = str(base_best["checkpoint_path"])

    ratios = cfg.get("ratios") or []
    if not isinstance(ratios, list) or not ratios:
        raise SystemExit("HN suite config must define a non-empty 'ratios' list.")

    max_ratio = max(int(value) for value in ratios)
    max_count_placeholder = int(cfg.get("max_backflow_count", 0) or 0)
    print_step("base", f"{base_model} -> epoch {base_best['epoch']} ({base_checkpoint_path})")

    run_python(
        "scripts/stage1_score_train_normals.py",
        [
            "--weights",
            base_checkpoint_path,
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(scoring_output_dir),
            "--device",
            resolve_str(cfg.get("score_device"), resolve_str(cfg.get("device"), "0")),
            "--imgsz",
            str(int(cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(cfg.get("score_batch", 1) or 1)),
            "--chunk-size",
            str(int(cfg.get("score_chunk_size", 32) or 32)),
            "--top-k",
            str(max_count_placeholder if max_count_placeholder > 0 else 250),
        ],
        dry_run=args.dry_run,
    )

    train_scores_csv = scoring_output_dir / "train_normal_scores.csv"
    if args.dry_run:
        total_normals = max(1, int(cfg.get("estimated_train_normals", 1080) or 1080))
    else:
        total_normals = count_scored_normals(train_scores_csv)
    print_step("data", f"train normals scored: {total_normals}")

    rows_for_sweep_manifest: list[dict[str, Any]] = []
    for percent in [int(value) for value in ratios]:
        count = backflow_count(total_normals, percent)
        rid = ratio_id(percent)
        summary_dir = materials_root / rid
        rows_for_sweep_manifest.append(
            {
                "ratio_id": rid,
                "ratio_percent": percent,
                "backflow_count": count,
                "summary_dir": str(summary_dir),
                "reuse_existing": percent == 0,
            }
        )
        if percent == 0:
            if args.rerun:
                archive_existing_path(summary_dir, recycle_root / "materials", dry_run=args.dry_run)
            maybe_prepare_existing_reference(
                base_summary_dir=base_summary_dir,
                target_dir=summary_dir,
                ratio_percent=percent,
                count=count,
                model_name=base_model,
                source_dataset=source_dataset,
                dry_run=args.dry_run,
            )
            continue

        run_ratio(
            config=cfg,
            ratio_percent=percent,
            count=count,
            base_checkpoint_path=base_checkpoint_path,
            source_dataset=source_dataset,
            materials_root=materials_root,
            project_root=project_root,
            recycle_root=recycle_root,
            temp_config_dir=temp_config_dir,
            dataset_view_root=dataset_view_root,
            split_csv=split_csv,
            train_scores_csv=train_scores_csv,
            dry_run=args.dry_run,
            rerun=args.rerun,
        )

        summarize_hn_sweep(
            materials_root=materials_root,
            results_dir=results_dir,
            model_name=base_model,
            label=resolve_str(cfg.get("label"), "formal gate HN sweep"),
        )

    if not args.dry_run:
        write_json(
            results_dir / "sweep_manifest.json",
            {
                "label": resolve_str(cfg.get("label"), "formal gate HN sweep"),
                "base_model": base_model,
                "base_checkpoint_path": base_checkpoint_path,
                "source_dataset": str(source_dataset),
                "materials_root": str(materials_root),
                "results_dir": str(results_dir),
                "project_root": str(project_root),
                "scoring_output_dir": str(scoring_output_dir),
                "dataset_view_root": str(dataset_view_root),
                "ratios": rows_for_sweep_manifest,
            },
        )

    summarize_hn_sweep(
        materials_root=materials_root,
        results_dir=results_dir,
        model_name=base_model,
        label=resolve_str(cfg.get("label"), "formal gate HN sweep"),
    )


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
