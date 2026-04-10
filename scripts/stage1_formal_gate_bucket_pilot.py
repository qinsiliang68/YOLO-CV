from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from pipeline_common import REPO_ROOT, YOLOV11_ROOT
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer-1 single-signal bucket pilot for stage-1 gate replay.")
    parser.add_argument("--config", required=True, help="Bucket pilot runtime config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing them.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing outputs before rerunning.")
    parser.add_argument("--preflight-only", action="store_true", help="Prepare scores and dataset views without starting training.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path_token(text: str | Path) -> str:
    return str(text).replace("\\", "/").rstrip("/").lower()


def try_rebase_repo_path(text: str | Path) -> Path | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path.resolve()
    normalized = raw.replace("\\", "/")
    lower = normalized.lower()
    markers = ("yolo-cv/", "yolov11/", "research/", "$out/", "_recycle_bin/")
    for marker in markers:
        index = lower.rfind(marker)
        if index < 0:
            continue
        suffix = normalized[index + len(marker) :] if marker == "yolo-cv/" else normalized[index:]
        return (REPO_ROOT / Path(suffix)).resolve()
    return None


def resolve_repo_token(text: str | Path) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    rebased = try_rebase_repo_path(raw)
    if rebased is not None:
        return str(rebased)
    return raw


def resolve_manifest_checkpoint(summary_dir: Path, best_manifest: dict[str, Any]) -> str:
    checkpoint_path = str(best_manifest.get("checkpoint_path", "")).strip()
    checkpoint_file = str(best_manifest.get("checkpoint_file", "")).strip()
    run_manifest = load_json_if_exists(summary_dir / "run_manifest.json")
    run_dir_text = str(run_manifest.get("run_dir", "")).strip()
    candidates: list[str] = []
    if checkpoint_path:
        candidates.append(resolve_repo_token(checkpoint_path))
    if run_dir_text and checkpoint_file:
        run_dir_candidate = resolve_repo_token(run_dir_text)
        if run_dir_candidate:
            candidates.append(str((Path(run_dir_candidate) / "weights" / checkpoint_file).resolve()))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        key = normalize_path_token(item)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    for item in ordered:
        if Path(item).exists():
            return str(Path(item).resolve())
    return ordered[0] if ordered else checkpoint_path


def sanitize_model_stem(model_name: str) -> str:
    return Path(model_name).stem.replace("-cls", "")


def resolve_ratio_row(summary_csv: Path, ratio_id: str) -> dict[str, str]:
    rows = load_csv_rows(summary_csv)
    match = next((row for row in rows if str(row.get("ratio_id")) == ratio_id), None)
    if match is None:
        raise SystemExit(f"Missing ratio `{ratio_id}` in {summary_csv}")
    return match


def is_experiment_complete(summary_dir: Path) -> bool:
    best_manifest = summary_dir / "best_epoch_manifest.json"
    epoch_summary = summary_dir / "epoch_gate_summary.csv"
    run_manifest = summary_dir / "run_manifest.json"
    if not (best_manifest.exists() and epoch_summary.exists() and run_manifest.exists()):
        return False
    try:
        payload = load_json(run_manifest)
    except Exception:
        return False
    return str(payload.get("status", "")).strip() == "evaluation_completed"


def build_train_config(
    *,
    teacher_checkpoint_path: str,
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
        "model": teacher_checkpoint_path,
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


def expected_experiments(cfg: dict[str, Any], score_output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_count = int(cfg.get("bucket_count", 5) or 5)
    for signal_name in ("R", "C", "D"):
        for bucket_index in range(bucket_count):
            bucket_name = f"Q{bucket_index + 1}"
            experiment_id = f"{signal_name}-{bucket_name}"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "signal": signal_name,
                    "bucket": bucket_name,
                    "setting_name": f"{signal_name.lower()}_{bucket_name.lower()}_bucket_uniform151",
                    "candidate_scores_csv": str(score_output_dir / "experiments" / f"{experiment_id}_candidate_scores.csv"),
                    "metadata_json": str(score_output_dir / "metadata" / f"{experiment_id}_bucket_metadata.json"),
                    "n_unique": int(cfg.get("candidate_top_k", 250) or 250) // bucket_count,
                    "n_replay": int(cfg.get("fixed_budget_count", 151) or 151),
                }
            )
    rows.append(
        {
            "experiment_id": "G0",
            "signal": "baseline",
            "bucket": "G0",
            "setting_name": "g0_uniform_random_pool250",
            "candidate_scores_csv": str(score_output_dir / "experiments" / "G0_candidate_scores.csv"),
            "metadata_json": str(score_output_dir / "metadata" / "G0_bucket_metadata.json"),
            "n_unique": int(cfg.get("fixed_budget_count", 151) or 151),
            "n_replay": int(cfg.get("fixed_budget_count", 151) or 151),
        }
    )
    return rows


def build_preflight(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    source_dataset: Path,
    split_csv: Path,
    teacher_summary_dir: Path,
    teacher_best: dict[str, Any],
    teacher_checkpoint_path: str,
    hn_summary_csv: Path,
    pool_top_csv: Path,
    feature_dir: Path,
    score_output_dir: Path,
    dataset_view_root: Path,
    results_dir: Path,
    registry_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "source_dataset": source_dataset.exists(),
        "split_csv": split_csv.exists(),
        "teacher_summary_dir": teacher_summary_dir.exists(),
        "teacher_best_manifest": (teacher_summary_dir / "best_epoch_manifest.json").exists(),
        "teacher_checkpoint_path": Path(teacher_checkpoint_path).exists() if teacher_checkpoint_path else False,
        "hn_summary_csv": hn_summary_csv.exists(),
        "pool_top_csv": pool_top_csv.exists(),
    }
    preflight = {
        "task_name": resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
        "config_path": str(config_path),
        "teacher_ratio_id": resolve_str(cfg.get("teacher_ratio_id"), "hn00"),
        "teacher_checkpoint_path": teacher_checkpoint_path,
        "teacher_best_epoch": int(teacher_best.get("epoch", -1)) if teacher_best else None,
        "teacher_tau_r995": float(teacher_best.get("tau_r995", 0.0)) if teacher_best else None,
        "teacher_tau_r990": float(teacher_best.get("tau_r990", 0.0)) if teacher_best else None,
        "fixed_budget_count": int(cfg.get("fixed_budget_count", 151) or 151),
        "candidate_top_k": int(cfg.get("candidate_top_k", 250) or 250),
        "experiments_planned": [row["experiment_id"] for row in registry_rows],
        "feature_dir": str(feature_dir),
        "score_output_dir": str(score_output_dir),
        "dataset_view_root": str(dataset_view_root),
        "results_dir": str(results_dir),
        "required_inputs": required,
        "ready_for_full_train": all(required.values()),
    }
    write_json(results_dir / "PREFLIGHT_gate_bucket_pilot.json", preflight)
    (results_dir / "PREFLIGHT_gate_bucket_pilot.md").write_text(
        "\n".join(
            [
                "# Gate Bucket Pilot Preflight",
                "",
                f"- task_name: `{preflight['task_name']}`",
                f"- teacher_ratio_id: `{preflight['teacher_ratio_id']}`",
                f"- teacher_checkpoint_path: `{preflight['teacher_checkpoint_path']}`",
                f"- teacher_best_epoch: `{preflight['teacher_best_epoch']}`",
                f"- fixed_budget_count: `{preflight['fixed_budget_count']}`",
                f"- candidate_top_k: `{preflight['candidate_top_k']}`",
                f"- experiments_planned: `{', '.join(preflight['experiments_planned'])}`",
                f"- ready_for_full_train: `{preflight['ready_for_full_train']}`",
                "",
                "Required inputs:",
                *[f"- {key}: `{value}`" for key, value in required.items()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return preflight


def ensure_feature_exports(
    *,
    teacher_checkpoint_path: str,
    source_dataset: Path,
    feature_dir: Path,
    cfg: dict[str, Any],
    dry_run: bool,
    rerun: bool,
) -> None:
    expected_outputs = [
        feature_dir / "train_features.csv",
        feature_dir / "train_embeddings.npy",
        feature_dir / "train_features_meta.json",
    ]
    if not rerun and all(path.exists() for path in expected_outputs):
        print_step("skip", f"reuse feature exports: {feature_dir}")
        return
    if rerun and feature_dir.exists() and not dry_run:
        shutil.rmtree(feature_dir)
    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            teacher_checkpoint_path,
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(feature_dir),
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
        dry_run=dry_run,
    )


def ensure_bucket_scores(
    *,
    teacher_checkpoint_path: str,
    teacher_summary_dir: Path,
    source_dataset: Path,
    pool_top_csv: Path,
    feature_dir: Path,
    score_output_dir: Path,
    fixed_budget_count: int,
    cfg: dict[str, Any],
    dry_run: bool,
    rerun: bool,
) -> None:
    registry_csv = score_output_dir / "bucket_experiment_registry.csv"
    if not rerun and registry_csv.exists():
        print_step("skip", f"reuse bucket score registry: {registry_csv}")
        return
    if rerun and score_output_dir.exists() and not dry_run:
        shutil.rmtree(score_output_dir)
    run_python(
        "scripts/stage1_prepare_bucket_signal_scores.py",
        [
            "--teacher-weights",
            teacher_checkpoint_path,
            "--teacher-best-manifest",
            str(teacher_summary_dir / "best_epoch_manifest.json"),
            "--source-dataset",
            str(source_dataset),
            "--pool-top-csv",
            str(pool_top_csv),
            "--train-features-csv",
            str(feature_dir / "train_features.csv"),
            "--train-embeddings-npy",
            str(feature_dir / "train_embeddings.npy"),
            "--output-dir",
            str(score_output_dir),
            "--device",
            resolve_str(cfg.get("score_device", cfg.get("device")), "0"),
            "--imgsz",
            str(int(cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(cfg.get("score_batch", 1) or 1)),
            "--candidate-top-k",
            str(int(cfg.get("candidate_top_k", 250) or 250)),
            "--fixed-budget-count",
            str(int(fixed_budget_count)),
            "--bucket-count",
            str(int(cfg.get("bucket_count", 5) or 5)),
            "--density-k",
            str(int(cfg.get("density_k", 15) or 15)),
            "--seed",
            str(int(cfg.get("seed", 20260330) or 20260330)),
            "--normal-class",
            resolve_str(cfg.get("normal_class"), "Normal"),
        ],
        dry_run=dry_run,
    )


def ensure_dataset_view(
    *,
    source_dataset: Path,
    dataset_root: Path,
    candidate_scores_csv: Path,
    setting_id: str,
    setting_name: str,
    fixed_budget_count: int,
    link_mode: str,
    dry_run: bool,
    rerun: bool,
) -> None:
    if not rerun and (dataset_root / "weighted_replay_summary.json").exists():
        print_step("skip", f"reuse dataset view: {dataset_root}")
        return
    run_python(
        "scripts/stage1_build_info_sampling_lite_dataset.py",
        [
            "--source-dataset",
            str(source_dataset),
            "--candidate-scores-csv",
            str(candidate_scores_csv),
            "--output-dataset",
            str(dataset_root),
            "--setting-name",
            setting_name,
            "--setting-id",
            setting_id,
            "--expected-duplication-total",
            str(int(fixed_budget_count)),
            "--link-mode",
            link_mode,
        ],
        dry_run=dry_run,
    )


def run_experiment(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    experiment: dict[str, Any],
    teacher_checkpoint_path: str,
    dataset_root: Path,
    project_root: Path,
    summary_dir: Path,
    temp_config_dir: Path,
    split_csv: Path,
    dry_run: bool,
    rerun: bool,
    recycle_root: Path,
) -> dict[str, Any]:
    experiment_id = str(experiment["experiment_id"])
    run_name = f"{sanitize_model_stem(resolve_str(cfg.get('base_model'), 'yolo11m-cls'))}_gate2_formal_bucket_{experiment_id.lower().replace('-', '_')}_200ep"
    run_dir = project_root / run_name
    temp_config_path = temp_config_dir / f"{run_name}.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    metadata_src = Path(str(experiment["metadata_json"]))

    if rerun:
        archive_existing_path(run_dir, recycle_root / "runs", dry_run=dry_run)
        archive_existing_path(summary_dir, recycle_root / "materials", dry_run=dry_run)

    normalize_run_dir(run_dir, dry_run=dry_run)
    maybe_prepare_run(
        task_kind="gate",
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=int(cfg.get("epochs", 200) or 200),
        status="planned",
        dry_run=dry_run,
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
        dry_run=dry_run,
    )
    if not rerun and is_experiment_complete(summary_dir):
        print_step("skip", f"{experiment_id}: evaluation already completed")
        return {
            "experiment_id": experiment_id,
            "signal": experiment["signal"],
            "bucket": experiment["bucket"],
            "setting_name": experiment["setting_name"],
            "run_name": run_name,
            "run_dir": str(run_dir),
            "summary_dir": str(summary_dir),
            "status": "reused",
        }

    last_checkpoint = run_dir / "weights" / "last.pt"
    resume_value: str | bool = False
    if bool(cfg.get("resume", True)) and last_checkpoint.exists():
        resume_value = str(last_checkpoint)
        print_step("resume", f"{experiment_id}: resume from {last_checkpoint}")

    train_cfg = build_train_config(
        teacher_checkpoint_path=teacher_checkpoint_path,
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
    if not dry_run:
        write_json(temp_config_path, train_cfg)
        summary_dir.mkdir(parents=True, exist_ok=True)
        if metadata_src.exists():
            shutil.copy2(metadata_src, summary_dir / "bucket_metadata.json")
    maybe_prepare_run(
        task_kind="gate",
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=int(cfg.get("epochs", 200) or 200),
        status="training_started",
        dry_run=dry_run,
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
        dry_run=dry_run,
    )
    maybe_prepare_run(
        task_kind="gate",
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=int(cfg.get("epochs", 200) or 200),
        status="training_completed",
        dry_run=dry_run,
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
        dry_run=dry_run,
    )
    if not dry_run:
        maybe_prepare_run(
            task_kind="gate",
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
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
    return {
        "experiment_id": experiment_id,
        "signal": experiment["signal"],
        "bucket": experiment["bucket"],
        "setting_name": experiment["setting_name"],
        "run_name": run_name,
        "run_dir": str(run_dir),
        "summary_dir": str(summary_dir),
        "status": "completed" if not dry_run else "planned",
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    cfg = load_json(config_path)
    print_step("task", resolve_str(cfg.get("label"), "stage-1 gate bucket pilot"))

    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200")
    split_csv = resolve_path(cfg.get("split_csv"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv")
    teacher_summary_dir = resolve_path(cfg.get("teacher_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00")
    hn_summary_csv = resolve_path(cfg.get("hn_summary_csv"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep" / "hn_sweep_summary.csv")
    pool_top_csv = resolve_path(cfg.get("pool_top_csv"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_assets" / "yolo11m_train_normal_scores" / "top_false_positive_normals.csv")
    project_root = resolve_path(cfg.get("project_root"), base=YOLOV11_ROOT / "runs" / "stage1_formal_gate_bucket_pilot")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "stage1_formal" / "gate_bucket_pilot")
    temp_config_dir = resolve_path(cfg.get("temp_config_dir"), base=REPO_ROOT / "$out" / "generated_configs" / "stage1_formal" / "gate_bucket_pilot")
    materials_root = resolve_path(cfg.get("materials_root"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot")
    score_output_dir = resolve_path(cfg.get("score_output_dir"), base=materials_root / "score_inputs")
    results_dir = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_bucket_pilot")
    scratch_root = resolve_path(cfg.get("scratch_root"), base=REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_bucket_pilot")
    dataset_view_root = resolve_path(cfg.get("dataset_view_root"), base=YOLOV11_ROOT / "datasets" / "stage1_formal_gate_bucket_pilot")

    teacher_best = load_json(teacher_summary_dir / "best_epoch_manifest.json")
    teacher_checkpoint_path = resolve_manifest_checkpoint(teacher_summary_dir, teacher_best)
    teacher_best["checkpoint_path"] = teacher_checkpoint_path

    if "fixed_budget_count" in cfg:
        fixed_budget_count = int(cfg.get("fixed_budget_count", 151) or 151)
    else:
        budget_anchor_ratio_id = resolve_str(cfg.get("budget_anchor_ratio_id"), "hn14")
        fixed_budget_count = int(resolve_ratio_row(hn_summary_csv, budget_anchor_ratio_id)["backflow_count"])

    feature_dir = scratch_root / "teacher_train_features"
    expected_rows = expected_experiments({**cfg, "fixed_budget_count": fixed_budget_count}, score_output_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    materials_root.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(
        cfg={**cfg, "fixed_budget_count": fixed_budget_count},
        config_path=config_path,
        source_dataset=source_dataset,
        split_csv=split_csv,
        teacher_summary_dir=teacher_summary_dir,
        teacher_best=teacher_best,
        teacher_checkpoint_path=teacher_checkpoint_path,
        hn_summary_csv=hn_summary_csv,
        pool_top_csv=pool_top_csv,
        feature_dir=feature_dir,
        score_output_dir=score_output_dir,
        dataset_view_root=dataset_view_root,
        results_dir=results_dir,
        registry_rows=expected_rows,
    )

    if not preflight["ready_for_full_train"] and not args.dry_run:
        raise SystemExit(f"Preflight failed. Inspect {results_dir / 'PREFLIGHT_gate_bucket_pilot.md'} before running.")

    ensure_feature_exports(
        teacher_checkpoint_path=teacher_checkpoint_path,
        source_dataset=source_dataset,
        feature_dir=feature_dir,
        cfg=cfg,
        dry_run=args.dry_run,
        rerun=args.rerun,
    )
    ensure_bucket_scores(
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_summary_dir=teacher_summary_dir,
        source_dataset=source_dataset,
        pool_top_csv=pool_top_csv,
        feature_dir=feature_dir,
        score_output_dir=score_output_dir,
        fixed_budget_count=fixed_budget_count,
        cfg=cfg,
        dry_run=args.dry_run,
        rerun=args.rerun,
    )

    registry_path = score_output_dir / "bucket_experiment_registry.csv"
    experiment_rows = expected_rows if args.dry_run or not registry_path.exists() else load_csv_rows(registry_path)

    for experiment in experiment_rows:
        ensure_dataset_view(
            source_dataset=source_dataset,
            dataset_root=dataset_view_root / str(experiment["experiment_id"]),
            candidate_scores_csv=Path(str(experiment["candidate_scores_csv"])),
            setting_id=str(experiment["experiment_id"]),
            setting_name=str(experiment["setting_name"]),
            fixed_budget_count=int(experiment["n_replay"]),
            link_mode=resolve_str(cfg.get("link_mode"), "hardlink"),
            dry_run=args.dry_run,
            rerun=args.rerun,
        )

    if args.preflight_only:
        return

    suite_rows: list[dict[str, Any]] = []
    for experiment in experiment_rows:
        suite_rows.append(
            run_experiment(
                cfg=cfg,
                config_path=config_path,
                experiment=experiment,
                teacher_checkpoint_path=teacher_checkpoint_path,
                dataset_root=dataset_view_root / str(experiment["experiment_id"]),
                project_root=project_root,
                summary_dir=materials_root / str(experiment["experiment_id"]),
                temp_config_dir=temp_config_dir,
                split_csv=split_csv,
                dry_run=args.dry_run,
                rerun=args.rerun,
                recycle_root=recycle_root,
            )
        )

    write_csv(
        results_dir / "bucket_pilot_suite_context.csv",
        ["experiment_id", "signal", "bucket", "setting_name", "run_name", "run_dir", "summary_dir", "status"],
        suite_rows,
    )
    write_json(
        results_dir / "bucket_pilot_suite_context.json",
        {
            "task_name": resolve_str(cfg.get("task_name"), "stage1_formal_gate_bucket_pilot"),
            "base_model": resolve_str(cfg.get("base_model"), "yolo11m-cls"),
            "teacher_ratio_id": resolve_str(cfg.get("teacher_ratio_id"), "hn00"),
            "teacher_checkpoint_path": teacher_checkpoint_path,
            "fixed_budget_count": fixed_budget_count,
            "rows": suite_rows,
        },
    )
    (results_dir / "bucket_pilot_suite_context.md").write_text(
        "\n".join(
            [
                "# Gate Bucket Pilot Suite Context",
                "",
                f"- base model: `{resolve_str(cfg.get('base_model'), 'yolo11m-cls')}`",
                f"- teacher ratio: `{resolve_str(cfg.get('teacher_ratio_id'), 'hn00')}`",
                f"- fixed_budget_count: `{fixed_budget_count}`",
                "",
                "| Experiment | Signal | Bucket | Status | Summary Dir |",
                "| --- | --- | --- | --- | --- |",
                *[
                    f"| {row['experiment_id']} | {row['signal']} | {row['bucket']} | {row['status']} | {row['summary_dir']} |"
                    for row in suite_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_python(
        "scripts/stage1_build_bucket_pilot_assets.py",
        [
            "--config",
            str(config_path),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
