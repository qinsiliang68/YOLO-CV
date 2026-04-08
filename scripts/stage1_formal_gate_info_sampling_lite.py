from __future__ import annotations

import argparse
import csv
import json
import math
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
    parser = argparse.ArgumentParser(description="Run the one-shot stage-1 effective-information lite ablation suite.")
    parser.add_argument("--config", required=True, help="Info-sampling-lite runtime config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing them.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing suite outputs before rerunning.")
    parser.add_argument("--preflight-only", action="store_true", help="Run setup, score, dataset, and validation checks without starting training.")
    parser.add_argument("--smoke-epochs", type=int, default=0, help="If >0, run an isolated smoke test for the selected setting(s) with this epoch count.")
    parser.add_argument("--smoke-setting", default="", help="Optional setting_id to smoke-test, e.g. A2 or A4. Defaults to A4 when --smoke-epochs > 0.")
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


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return load_csv_rows(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path_token(text: str | Path) -> str:
    return str(text).replace("\\", "/").rstrip("/").lower()


def sanitize_model_stem(model_name: str) -> str:
    return Path(model_name).stem.replace("-cls", "")


def resolve_ratio_row(summary_csv: Path, ratio_id: str) -> dict[str, str]:
    rows = load_csv_rows(summary_csv)
    match = next((row for row in rows if str(row.get("ratio_id")) == ratio_id), None)
    if match is None:
        overview_csv = summary_csv.parents[1] / "gate_hn_overview" / "stage1_formal_hn_overview.csv"
        if overview_csv.exists():
            overview_rows = load_csv_rows(overview_csv)
            match = next((row for row in overview_rows if str(row.get("ratio_id")) == ratio_id and str(row.get("model")) == "yolo11m-cls"), None)
    if match is None:
        raise SystemExit(f"Missing ratio `{ratio_id}` in {summary_csv}")
    return match


def try_resolve_ratio_row(summary_csv: Path, ratio_id: str) -> dict[str, str]:
    try:
        return resolve_ratio_row(summary_csv, ratio_id)
    except SystemExit:
        return {}


def is_setting_complete(summary_dir: Path) -> bool:
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


def build_setup_audit(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    source_dataset: Path,
    hn_summary_csv: Path,
    teacher_summary_dir: Path,
    uniform_anchor_summary_dir: Path,
    pool_top_csv: Path,
    pool_scores_csv: Path,
    pool_summary_json: Path,
    results_dir: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    teacher_manifest_path = teacher_summary_dir / "best_epoch_manifest.json"
    uniform_manifest_path = uniform_anchor_summary_dir / "best_epoch_manifest.json"
    teacher_best = load_json_if_exists(teacher_manifest_path)
    uniform_best = load_json_if_exists(uniform_manifest_path)
    pool_summary = load_json_if_exists(pool_summary_json)
    pool_top_rows = load_csv_rows_if_exists(pool_top_csv)
    fixed_budget_row = try_resolve_ratio_row(hn_summary_csv, resolve_str(cfg.get("budget_anchor_ratio_id"), "hn14")) if hn_summary_csv.exists() else {}
    teacher_row = try_resolve_ratio_row(hn_summary_csv, resolve_str(cfg.get("teacher_ratio_id"), "hn00")) if hn_summary_csv.exists() else {}
    total_train_normals = int(pool_summary.get("total_train_normals", 0))
    fixed_budget_count = (
        int(fixed_budget_row["backflow_count"])
        if "backflow_count" in fixed_budget_row
        else int(round(total_train_normals * float(fixed_budget_row.get("ratio_percent", 0.0) or 0.0) / 100.0))
    )

    teacher_checkpoint = str(teacher_best.get("checkpoint_path", ""))
    pool_teacher_checkpoint = str(pool_summary.get("weights", ""))
    required_inputs = {
        "hn_summary_csv": hn_summary_csv.exists(),
        "teacher_best_manifest": teacher_manifest_path.exists(),
        "uniform_best_manifest": uniform_manifest_path.exists(),
        "pool_top_csv": pool_top_csv.exists(),
        "pool_scores_csv": pool_scores_csv.exists(),
        "pool_summary_json": pool_summary_json.exists(),
        "source_dataset": source_dataset.exists(),
    }
    audit = {
        "task_name": resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
        "config_path": str(config_path),
        "source_dataset": str(source_dataset),
        "source_dataset_exists": source_dataset.exists(),
        "base_model": resolve_str(cfg.get("base_model"), "yolo11m-cls"),
        "teacher_ratio_id": resolve_str(cfg.get("teacher_ratio_id"), "hn00"),
        "uniform_anchor_ratio_id": resolve_str(cfg.get("uniform_anchor_ratio_id"), "hn14"),
        "budget_anchor_ratio_id": resolve_str(cfg.get("budget_anchor_ratio_id"), "hn14"),
        "teacher_checkpoint_path": teacher_checkpoint,
        "teacher_checkpoint_exists": Path(teacher_checkpoint).exists() if teacher_checkpoint else False,
        "teacher_summary_dir": str(teacher_summary_dir),
        "uniform_anchor_summary_dir": str(uniform_anchor_summary_dir),
        "teacher_best_epoch": int(teacher_best.get("epoch", -1)) if teacher_best else None,
        "teacher_best_spec_at_r995": float(teacher_best.get("spec_at_r995", 0.0)) if teacher_best else None,
        "uniform_anchor_epoch": int(uniform_best.get("epoch", -1)) if uniform_best else None,
        "uniform_anchor_spec_at_r995": float(uniform_best.get("spec_at_r995", 0.0)) if uniform_best else None,
        "pool_top_csv": str(pool_top_csv),
        "pool_scores_csv": str(pool_scores_csv),
        "pool_summary_json": str(pool_summary_json),
        "pool_top_count": len(pool_top_rows),
        "pool_summary_top_k": int(pool_summary.get("top_k", len(pool_top_rows))),
        "pool_total_train_normals": total_train_normals,
        "pool_teacher_checkpoint_path": pool_teacher_checkpoint,
        "pool_teacher_matches_hn00_teacher": bool(pool_teacher_checkpoint and teacher_checkpoint) and normalize_path_token(pool_teacher_checkpoint) == normalize_path_token(teacher_checkpoint),
        "teacher_summary_row": teacher_row,
        "budget_anchor_summary_row": fixed_budget_row,
        "fixed_budget_count": fixed_budget_count,
        "candidate_top_k": int(cfg.get("candidate_top_k", 250) or 250),
        "dry_run": bool(dry_run),
        "required_inputs": required_inputs,
        "core_inputs_ready": all(required_inputs.values()),
    }
    audit["notes"] = [
        "Teacher defaults to hn00 unless the existing hard-normal pool provenance contradicts this assumption.",
        "This suite keeps the top250 hard-normal pool fixed and only redistributes replay probability under the hn14-equivalent budget.",
        "A0/A1 are reused anchors; only A2/A3/A4 are newly trained.",
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "setup_audit.json", audit)
    (results_dir / "setup_audit.md").write_text(
        "\n".join(
            [
                "# Setup Audit",
                "",
                f"- base model: `{audit['base_model']}`",
                f"- teacher ratio: `{audit['teacher_ratio_id']}`",
                f"- teacher checkpoint: `{audit['teacher_checkpoint_path']}`",
                f"- teacher checkpoint exists: `{audit['teacher_checkpoint_exists']}`",
                f"- uniform anchor ratio: `{audit['uniform_anchor_ratio_id']}`",
                f"- budget anchor ratio: `{audit['budget_anchor_ratio_id']}`",
                f"- fixed budget count: `{audit['fixed_budget_count']}`",
                f"- fixed pool size: `{audit['pool_top_count']}`",
                f"- pool provenance matches hn00 teacher: `{audit['pool_teacher_matches_hn00_teacher']}`",
                f"- source dataset exists: `{audit['source_dataset_exists']}`",
                f"- core inputs ready: `{audit['core_inputs_ready']}`",
                "",
                "Required inputs:",
                *[f"- {key}: `{value}`" for key, value in required_inputs.items()],
                "",
                "Notes:",
                *[f"- {note}" for note in audit["notes"]],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit, teacher_best, int(audit["fixed_budget_count"])


def csv_columns_present(path: Path, required: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, required
    rows = load_csv_rows(path)
    fieldnames = list(rows[0].keys()) if rows else []
    missing = [name for name in required if name not in fieldnames]
    return len(missing) == 0, missing


def json_keys_present(path: Path, required: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, required
    payload = load_json(path)
    missing = [name for name in required if name not in payload]
    return len(missing) == 0, missing


def finite_flag(values: list[float]) -> tuple[bool, bool]:
    has_nan = any(math.isnan(float(value)) for value in values)
    has_inf = any(math.isinf(float(value)) for value in values)
    return has_nan, has_inf


def check_cleanup_scope(cleanup_roots: list[Path], protected_roots: list[Path]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    for cleanup_root in cleanup_roots:
        cleanup_norm = normalize_path_token(cleanup_root)
        for protected_root in protected_roots:
            protected_norm = normalize_path_token(protected_root)
            if cleanup_norm == protected_norm or cleanup_norm.startswith(protected_norm + "/") or protected_norm.startswith(cleanup_norm + "/"):
                violations.append({"cleanup_root": str(cleanup_root), "protected_root": str(protected_root)})
    return {"safe": not violations, "violations": violations}


def build_preflight_report(
    *,
    cfg: dict[str, Any],
    settings: list[dict[str, Any]],
    audit: dict[str, Any],
    hn_summary_csv: Path,
    teacher_summary_dir: Path,
    uniform_anchor_summary_dir: Path,
    score_output_dir: Path,
    dataset_view_root: Path,
    results_dir: Path,
    materials_root: Path,
    project_root: Path,
    recycle_root: Path,
    scratch_root: Path,
    temp_config_dir: Path,
    preflight_mode: str,
    smoke_epochs: int,
    smoke_setting: str,
) -> dict[str, Any]:
    pool_top_rows = load_csv_rows_if_exists(resolve_path(cfg.get("pool_top_csv"), base=REPO_ROOT))
    top250_paths = [normalize_path_token(row["img_rel_path"]) for row in pool_top_rows[: int(cfg.get("candidate_top_k", 250) or 250)]]
    preflight: dict[str, Any] = {
        "task_name": resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
        "mode": preflight_mode,
        "smoke_epochs": smoke_epochs,
        "smoke_setting": smoke_setting,
        "teacher": {
            "ratio_id": audit["teacher_ratio_id"],
            "checkpoint_path": audit["teacher_checkpoint_path"],
            "checkpoint_exists": audit["teacher_checkpoint_exists"],
            "summary_dir": str(teacher_summary_dir),
            "pool_teacher_matches_hn00_teacher": bool(audit["pool_teacher_matches_hn00_teacher"]),
        },
        "pool": {
            "candidate_top_k": int(cfg.get("candidate_top_k", 250) or 250),
            "pool_top_count": int(audit["pool_top_count"]),
            "pool_source_csv": str(resolve_path(cfg.get("pool_top_csv"), base=REPO_ROOT)),
            "pool_scores_csv": str(resolve_path(cfg.get("pool_scores_csv"), base=REPO_ROOT)),
            "pool_is_reused_existing_top250": True,
        },
        "budget": {
            "budget_anchor_ratio_id": audit["budget_anchor_ratio_id"],
            "fixed_budget_count": int(audit["fixed_budget_count"]),
            "uniform_anchor_ratio_id": audit["uniform_anchor_ratio_id"],
        },
        "score_checks": {},
        "dataset_checks": {},
        "schema_checks": {},
        "rerun_scope": {},
        "ready_for_full_train": False,
    }

    required_hn_cols = [
        "ratio_id",
        "best_epoch",
        "spec_at_r995",
        "spec_at_r990",
        "prec_at_r990",
        "ptr_at_r990",
        "summary_dir",
    ]
    required_best_keys = [
        "epoch",
        "checkpoint_path",
        "spec_at_r995",
        "spec_at_r990",
        "prec_at_r990",
        "ptr_at_r990",
        "tau_r995",
        "tau_r990",
        "temperature_T",
    ]
    required_new_best_keys = [
        "epoch",
        "checkpoint_path",
        "spec_at_r995",
        "spec_at_r990",
        "prec_at_r990",
        "ptr_at_r990",
    ]
    required_score_cols = [
        "setting_id",
        "image_id",
        "img_rel_path",
        "pool_rank",
        "calibrated_p",
        "R",
        "C",
        "D",
        "S",
        "pi",
        "duplication_count",
    ]

    hn_schema_ok, hn_missing = csv_columns_present(hn_summary_csv, required_hn_cols)
    teacher_schema_ok, teacher_missing = json_keys_present(teacher_summary_dir / "best_epoch_manifest.json", required_best_keys)
    uniform_schema_ok, uniform_missing = json_keys_present(uniform_anchor_summary_dir / "best_epoch_manifest.json", required_best_keys)

    preflight["schema_checks"]["anchors"] = {
        "hn_summary_csv": str(hn_summary_csv),
        "hn_summary_schema_ok": hn_schema_ok,
        "hn_summary_missing_columns": hn_missing,
        "teacher_best_manifest_ok": teacher_schema_ok,
        "teacher_best_manifest_missing": teacher_missing,
        "uniform_best_manifest_ok": uniform_schema_ok,
        "uniform_best_manifest_missing": uniform_missing,
    }

    settings_ok = True
    for setting in settings:
        setting_id = resolve_str(setting.get("setting_id"), "")
        setting_name = resolve_str(setting.get("setting_name"), setting_id.lower())
        score_csv = score_output_dir / f"{setting_id}_candidate_pool_scores.csv"
        dataset_root = dataset_view_root / setting_name
        dataset_summary_path = dataset_root / "weighted_replay_summary.json"
        summary_dir = materials_root / setting_name

        score_entry: dict[str, Any] = {
            "score_csv": str(score_csv),
            "score_csv_exists": score_csv.exists(),
        }
        dataset_entry: dict[str, Any] = {
            "dataset_root": str(dataset_root),
            "weighted_replay_summary_exists": dataset_summary_path.exists(),
        }
        schema_entry: dict[str, Any] = {
            "summary_dir": str(summary_dir),
            "best_epoch_manifest_exists": (summary_dir / "best_epoch_manifest.json").exists(),
            "epoch_gate_summary_exists": (summary_dir / "epoch_gate_summary.csv").exists(),
        }

        if score_csv.exists():
            score_ok, score_missing = csv_columns_present(score_csv, required_score_cols)
            score_rows = load_csv_rows(score_csv)
            pi_values = [float(row["pi"]) for row in score_rows]
            s_values = [float(row["S"]) for row in score_rows]
            duplication_total = int(sum(int(row["duplication_count"]) for row in score_rows))
            pool_exact_match = [normalize_path_token(row["img_rel_path"]) for row in score_rows] == top250_paths
            s_nan, s_inf = finite_flag(s_values)
            pi_nan, pi_inf = finite_flag(pi_values)
            sum_pi = float(sum(pi_values))
            score_entry.update(
                {
                    "schema_ok": score_ok,
                    "missing_columns": score_missing,
                    "row_count": len(score_rows),
                    "candidate_top_k_matches": len(score_rows) == int(cfg.get("candidate_top_k", 250) or 250),
                    "pool_exact_reuse_matches_top250": pool_exact_match,
                    "sum_pi": sum_pi,
                    "sum_pi_close_to_one": math.isclose(sum_pi, 1.0, rel_tol=0.0, abs_tol=1e-5),
                    "score_has_nan": s_nan,
                    "score_has_inf": s_inf,
                    "probability_has_nan": pi_nan,
                    "probability_has_inf": pi_inf,
                    "duplication_total": duplication_total,
                    "duplication_total_matches_budget": duplication_total == int(audit["fixed_budget_count"]),
                    "top10_cumulative_pi": float(sum(sorted(pi_values, reverse=True)[:10])),
                }
            )
        else:
            score_entry.update({"schema_ok": False, "missing_columns": required_score_cols})

        if dataset_summary_path.exists():
            dataset_summary = load_json(dataset_summary_path)
            dataset_entry.update(
                {
                    "duplicated_images": dataset_summary.get("duplicated_images"),
                    "expected_duplicated_images": dataset_summary.get("expected_duplicated_images"),
                    "duplication_total_matches_expected": dataset_summary.get("duplication_total_matches_expected"),
                }
            )

        if (summary_dir / "best_epoch_manifest.json").exists():
            best_ok, best_missing = json_keys_present(summary_dir / "best_epoch_manifest.json", required_new_best_keys)
            epoch_ok, epoch_missing = csv_columns_present(
                summary_dir / "epoch_gate_summary.csv",
                ["epoch", "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990"],
            )
            schema_entry.update(
                {
                    "best_manifest_schema_ok": best_ok,
                    "best_manifest_missing": best_missing,
                    "epoch_summary_schema_ok": epoch_ok,
                    "epoch_summary_missing": epoch_missing,
                }
            )
        else:
            schema_entry.update(
                {
                    "best_manifest_schema_ok": None,
                    "best_manifest_missing": [],
                    "epoch_summary_schema_ok": None,
                    "epoch_summary_missing": [],
                }
            )

        preflight["score_checks"][setting_id] = score_entry
        preflight["dataset_checks"][setting_id] = dataset_entry
        preflight["schema_checks"][setting_id] = schema_entry

        settings_ok = settings_ok and bool(score_entry.get("schema_ok")) and bool(score_entry.get("sum_pi_close_to_one")) and not bool(score_entry.get("score_has_nan")) and not bool(score_entry.get("score_has_inf")) and not bool(score_entry.get("probability_has_nan")) and not bool(score_entry.get("probability_has_inf")) and bool(score_entry.get("duplication_total_matches_budget")) and bool(dataset_entry.get("duplication_total_matches_expected"))

    cleanup_scope = check_cleanup_scope(
        cleanup_roots=[
            materials_root,
            results_dir,
            dataset_view_root,
            project_root,
            recycle_root,
            scratch_root,
            temp_config_dir,
        ],
        protected_roots=[
            REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep",
            REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep",
            REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_overview",
            REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_assets",
        ],
    )
    preflight["rerun_scope"] = {
        "cleanup_roots": [str(path) for path in [materials_root, results_dir, dataset_view_root, project_root, recycle_root, scratch_root, temp_config_dir]],
        "protected_hn_roots": [str(path) for path in [REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep", REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep", REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_overview", REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_assets"]],
        "safe": cleanup_scope["safe"],
        "violations": cleanup_scope["violations"],
    }

    preflight["ready_for_full_train"] = bool(
        audit["core_inputs_ready"]
        and audit["teacher_checkpoint_exists"]
        and audit["source_dataset_exists"]
        and audit["pool_teacher_matches_hn00_teacher"]
        and int(audit["pool_top_count"]) == int(cfg.get("candidate_top_k", 250) or 250)
        and settings_ok
        and cleanup_scope["safe"]
        and hn_schema_ok
        and teacher_schema_ok
        and uniform_schema_ok
    )
    preflight["checked_settings"] = [resolve_str(item.get("setting_id"), "") for item in settings]

    json_path = results_dir / "PREFLIGHT_gate_info_sampling_lite.json"
    md_path = results_dir / "PREFLIGHT_gate_info_sampling_lite.md"
    write_json(json_path, preflight)
    lines = [
        "# PREFLIGHT: Gate Info Sampling Lite",
        "",
        f"- mode: `{preflight_mode}`",
        f"- smoke_epochs: `{smoke_epochs}`",
        f"- smoke_setting: `{smoke_setting or 'none'}`",
        f"- teacher ratio: `{preflight['teacher']['ratio_id']}`",
        f"- teacher checkpoint exists: `{preflight['teacher']['checkpoint_exists']}`",
        f"- pool exact reuse source: `{preflight['pool']['pool_source_csv']}`",
        f"- fixed budget count: `{preflight['budget']['fixed_budget_count']}`",
        f"- checked settings: `{', '.join(preflight['checked_settings']) or 'none'}`",
        f"- pool provenance matches hn00 teacher: `{preflight['teacher']['pool_teacher_matches_hn00_teacher']}`",
        f"- core inputs ready: `{audit['core_inputs_ready']}`",
        f"- rerun cleanup scope safe: `{preflight['rerun_scope']['safe']}`",
        f"- ready_for_full_train: `{preflight['ready_for_full_train']}`",
        "",
        "## Required inputs",
        *[f"- {key}: `{value}`" for key, value in audit["required_inputs"].items()],
        "",
        "## Per-setting score checks",
    ]
    for setting_id, entry in preflight["score_checks"].items():
        lines.extend(
            [
                f"### {setting_id}",
                f"- score csv exists: `{entry['score_csv_exists']}`",
                f"- schema ok: `{entry.get('schema_ok')}`",
                f"- row_count: `{entry.get('row_count', 'NA')}`",
                f"- candidate_top_k_matches: `{entry.get('candidate_top_k_matches', 'NA')}`",
                f"- pool_exact_reuse_matches_top250: `{entry.get('pool_exact_reuse_matches_top250', 'NA')}`",
                f"- sum(pi): `{entry.get('sum_pi', 'NA')}`",
                f"- sum(pi) close to 1: `{entry.get('sum_pi_close_to_one', 'NA')}`",
                f"- score_has_nan/inf: `{entry.get('score_has_nan', 'NA')}` / `{entry.get('score_has_inf', 'NA')}`",
                f"- probability_has_nan/inf: `{entry.get('probability_has_nan', 'NA')}` / `{entry.get('probability_has_inf', 'NA')}`",
                f"- duplication_total matches budget: `{entry.get('duplication_total_matches_budget', 'NA')}`",
                f"- dataset duplication_total_matches_expected: `{preflight['dataset_checks'][setting_id].get('duplication_total_matches_expected', 'NA')}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return preflight


def maybe_prepare_scores(
    *,
    cfg: dict[str, Any],
    source_dataset: Path,
    score_output_dir: Path,
    scratch_root: Path,
    teacher_best: dict[str, Any],
    fixed_budget_count: int,
    pool_top_csv: Path,
    pool_scores_csv: Path,
    results_dir: Path,
    dry_run: bool,
    rerun: bool,
    recycle_root: Path,
) -> None:
    expected_outputs = [
        score_output_dir / "candidate_pool_master.csv",
        score_output_dir / "A2_candidate_pool_scores.csv",
        score_output_dir / "A3_candidate_pool_scores.csv",
        score_output_dir / "A4_candidate_pool_scores.csv",
        score_output_dir / "table_score_component_stats.csv",
        score_output_dir / "pool_source_manifest.json",
    ]
    if not rerun and all(path.exists() for path in expected_outputs):
        print_step("skip", "reuse existing info-sampling-lite score files")
        return

    feature_dir = scratch_root / "teacher_train_features"
    if rerun:
        archive_existing_path(score_output_dir, recycle_root / "score_inputs", dry_run=dry_run)
        archive_existing_path(feature_dir, recycle_root / "scratch", dry_run=dry_run)

    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            str(teacher_best["checkpoint_path"]),
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

    run_python(
        "scripts/stage1_prepare_info_sampling_lite_scores.py",
        [
            "--teacher-weights",
            str(teacher_best["checkpoint_path"]),
            "--teacher-best-manifest",
            str(resolve_path(cfg.get("teacher_summary_dir"), base=REPO_ROOT) / "best_epoch_manifest.json"),
            "--source-dataset",
            str(source_dataset),
            "--pool-top-csv",
            str(pool_top_csv),
            "--pool-scores-csv",
            str(pool_scores_csv),
            "--train-features-csv",
            str(feature_dir / "train_features.csv"),
            "--train-embeddings-npy",
            str(feature_dir / "train_embeddings.npy"),
            "--output-dir",
            str(score_output_dir),
            "--device",
            resolve_str(cfg.get("score_device"), resolve_str(cfg.get("device"), "0")),
            "--imgsz",
            str(int(cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(cfg.get("score_batch", 1) or 1)),
            "--candidate-top-k",
            str(int(cfg.get("candidate_top_k", 250) or 250)),
            "--fixed-budget-count",
            str(fixed_budget_count),
            "--alpha",
            str(float(cfg.get("alpha", 2.0) or 2.0)),
            "--density-k",
            str(int(cfg.get("density_k", 15) or 15)),
            "--kappa",
            str(float(cfg.get("kappa", 2.0) or 2.0)),
            "--seed",
            str(int(cfg.get("seed", 20260330) or 20260330)),
            "--normal-class",
            resolve_str(cfg.get("normal_class"), "Normal"),
        ],
        dry_run=dry_run,
    )

    if dry_run:
        return

    export_summary_path = feature_dir / "export_summary.json"
    if export_summary_path.exists():
        export_summary = load_json(export_summary_path)
    else:
        export_summary = {}
    write_json(
        score_output_dir / "feature_extraction_manifest.json",
        {
            "teacher_checkpoint_path": str(teacher_best["checkpoint_path"]),
            "feature_export_dir": str(feature_dir),
            "feature_export_summary": export_summary,
            "cleanup_planned": True,
        },
    )
    write_json(
        score_output_dir / "tta_manifest.json",
        {
            "K": 5,
            "variants": [
                "orig",
                "bright_down",
                "bright_up",
                "contrast_down",
                "blur_light",
            ],
            "note": "Only low-distortion semantic-preserving TTA variants are used for consistency scoring.",
        },
    )

    if feature_dir.exists():
        shutil.rmtree(feature_dir)
    write_json(
        results_dir / "scratch_cleanup.json",
        {
            "removed_paths": [str(feature_dir)],
            "reason": "large temporary feature/embedding exports are not kept in the formal working set",
        },
    )


def maybe_build_dataset(
    *,
    source_dataset: Path,
    dataset_root: Path,
    score_csv: Path,
    setting_id: str,
    setting_name: str,
    expected_duplication_total: int,
    link_mode: str,
    dry_run: bool,
    rerun: bool,
    recycle_root: Path,
) -> None:
    summary_path = dataset_root / "weighted_replay_summary.json"
    if not rerun and summary_path.exists() and dataset_root.exists():
        print_step("skip", f"{setting_name}: reuse existing dataset view")
        return
    if rerun:
        archive_existing_path(dataset_root, recycle_root / "datasets", dry_run=dry_run)
    run_python(
        "scripts/stage1_build_info_sampling_lite_dataset.py",
        [
            "--source-dataset",
            str(source_dataset),
            "--candidate-scores-csv",
            str(score_csv),
            "--output-dataset",
            str(dataset_root),
            "--setting-name",
            setting_name,
            "--setting-id",
            setting_id,
            "--expected-duplication-total",
            str(int(expected_duplication_total)),
            "--link-mode",
            link_mode,
        ],
        dry_run=dry_run,
    )


def run_setting(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    setting: dict[str, Any],
    teacher_best: dict[str, Any],
    dataset_root: Path,
    project_root: Path,
    summary_dir: Path,
    temp_config_dir: Path,
    split_csv: Path,
    epochs_override: int | None,
    dry_run: bool,
    rerun: bool,
    recycle_root: Path,
) -> dict[str, Any]:
    base_model = resolve_str(cfg.get("base_model"), "yolo11m-cls")
    model_stem = sanitize_model_stem(base_model)
    setting_id = resolve_str(setting.get("setting_id"), "")
    setting_name = resolve_str(setting.get("setting_name"), setting_id.lower())
    effective_epochs = int(epochs_override or int(cfg.get("epochs", 200) or 200))
    if epochs_override:
        run_name = f"{model_stem}_gate2_formal_{setting_name}_{effective_epochs}ep_smoke"
    else:
        run_name = f"{model_stem}_gate2_formal_{setting_name}_200ep"
    run_dir = project_root / run_name
    temp_config_path = temp_config_dir / f"{run_name}.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    if rerun:
        archive_existing_path(run_dir, recycle_root / "runs", dry_run=dry_run)
        archive_existing_path(summary_dir, recycle_root / "materials", dry_run=dry_run)

    normalize_run_dir(run_dir, dry_run=dry_run)
    maybe_prepare_run(
        task_kind="gate",
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=effective_epochs,
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

    if not rerun and is_setting_complete(summary_dir):
        print_step("skip", f"{setting_name}: evaluation already completed")
        return {
            "setting_id": setting_id,
            "setting_name": setting_name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "summary_dir": str(summary_dir),
            "status": "reused",
        }

    last_checkpoint = run_dir / "weights" / "last.pt"
    resume_value: str | bool = False
    if bool(cfg.get("resume", True)) and last_checkpoint.exists():
        resume_value = str(last_checkpoint)
        print_step("resume", f"{setting_name}: resume from {last_checkpoint}")

    train_cfg = build_train_config(
        teacher_checkpoint_path=str(teacher_best["checkpoint_path"]),
        dataset_root=dataset_root,
        project_root=project_root,
        run_name=run_name,
        epochs=effective_epochs,
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
    maybe_prepare_run(
        task_kind="gate",
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=effective_epochs,
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
        task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
        config_path=temp_config_path,
        dataset_root=dataset_root,
        run_dir=run_dir,
        summary_dir=summary_dir,
        split_manifest=split_csv,
        normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
        batch=int(cfg.get("batch", 24) or 24),
        epochs=effective_epochs,
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
            task_name=resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
            config_path=temp_config_path,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=resolve_str(cfg.get("normal_class"), "Normal"),
            batch=int(cfg.get("batch", 24) or 24),
            epochs=effective_epochs,
            status="evaluation_completed",
            dry_run=False,
        )

    return {
        "setting_id": setting_id,
        "setting_name": setting_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "summary_dir": str(summary_dir),
        "status": "completed" if not dry_run else "planned",
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    cfg = load_json(config_path)
    print_step("task", resolve_str(cfg.get("label"), "stage-1 effective-information lite ablation"))

    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200")
    hn_summary_csv = resolve_path(cfg.get("hn_summary_csv"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep" / "hn_sweep_summary.csv")
    teacher_summary_dir = resolve_path(cfg.get("teacher_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00")
    uniform_anchor_summary_dir = resolve_path(cfg.get("uniform_anchor_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn14")
    pool_top_csv = resolve_path(cfg.get("pool_top_csv"), base=REPO_ROOT)
    pool_scores_csv = resolve_path(cfg.get("pool_scores_csv"), base=REPO_ROOT)
    pool_summary_json = resolve_path(cfg.get("pool_summary_json"), base=REPO_ROOT)
    project_root = resolve_path(cfg.get("project_root"), base=YOLOV11_ROOT / "runs" / "stage1_formal_gate_info_sampling_lite")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "stage1_formal_gate_info_sampling_lite")
    temp_config_dir = resolve_path(cfg.get("temp_config_dir"), base=REPO_ROOT / "$out" / "generated_configs" / "stage1_formal" / "gate_info_sampling_lite")
    materials_root = resolve_path(cfg.get("materials_root"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_info_sampling_lite")
    score_output_dir = resolve_path(cfg.get("score_output_dir"), base=materials_root / "score_inputs")
    results_dir = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_info_sampling_lite")
    scratch_root = resolve_path(cfg.get("scratch_root"), base=REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_info_sampling_lite")
    dataset_view_root = resolve_path(cfg.get("dataset_view_root"), base=YOLOV11_ROOT / "datasets" / "stage1_formal_gate_info_sampling_lite")
    split_csv = resolve_path(cfg.get("split_csv"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv")

    smoke_setting = args.smoke_setting.strip().upper()
    if args.smoke_epochs < 0:
        raise SystemExit("--smoke-epochs must be >= 0")
    if args.smoke_epochs > 0:
        if not smoke_setting:
            smoke_setting = "A4"
        mode_suffix = f"smoke_{smoke_setting.lower()}_{int(args.smoke_epochs)}ep"
        print_step("mode", f"smoke test enabled: setting={smoke_setting} epochs={int(args.smoke_epochs)}")
        project_root = project_root / mode_suffix
        recycle_root = recycle_root / mode_suffix
        temp_config_dir = temp_config_dir / mode_suffix
        materials_root = materials_root / mode_suffix
        score_output_dir = materials_root / "score_inputs"
        results_dir = results_dir / mode_suffix
        scratch_root = scratch_root / mode_suffix
        dataset_view_root = dataset_view_root / mode_suffix

    results_dir.mkdir(parents=True, exist_ok=True)
    materials_root.mkdir(parents=True, exist_ok=True)

    audit, teacher_best, fixed_budget_count = build_setup_audit(
        cfg=cfg,
        config_path=config_path,
        source_dataset=source_dataset,
        hn_summary_csv=hn_summary_csv,
        teacher_summary_dir=teacher_summary_dir,
        uniform_anchor_summary_dir=uniform_anchor_summary_dir,
        pool_top_csv=pool_top_csv,
        pool_scores_csv=pool_scores_csv,
        pool_summary_json=pool_summary_json,
        results_dir=results_dir,
        dry_run=args.dry_run,
    )
    uniform_anchor_best = load_json_if_exists(uniform_anchor_summary_dir / "best_epoch_manifest.json")

    if audit["core_inputs_ready"] or args.dry_run:
        maybe_prepare_scores(
            cfg=cfg,
            source_dataset=source_dataset,
            score_output_dir=score_output_dir,
            scratch_root=scratch_root,
            teacher_best=teacher_best,
            fixed_budget_count=fixed_budget_count,
            pool_top_csv=pool_top_csv,
            pool_scores_csv=pool_scores_csv,
            results_dir=results_dir,
            dry_run=args.dry_run,
            rerun=args.rerun,
            recycle_root=recycle_root,
        )
    else:
        print_step("preflight", "skip score preparation because required input files are missing")

    suite_rows: list[dict[str, Any]] = [
        {
            "setting_id": "A0",
            "setting_name": "hn00",
            "run_name": str(try_resolve_ratio_row(hn_summary_csv, "hn00").get("run_name", "yolo11m_gate2_formal_200ep")),
            "run_dir": str(teacher_best.get("checkpoint_path", "")),
            "summary_dir": str(teacher_summary_dir),
            "status": "reused",
        },
        {
            "setting_id": "A1",
            "setting_name": "uniform_hn14",
            "run_name": str(try_resolve_ratio_row(hn_summary_csv, resolve_str(cfg.get("uniform_anchor_ratio_id"), "hn14")).get("run_name", "yolo11m_gate2_formal_hn14_200ep")),
            "run_dir": str(uniform_anchor_best.get("checkpoint_path", "")),
            "summary_dir": str(uniform_anchor_summary_dir),
            "status": "reused",
        },
    ]

    settings = cfg.get("settings") or []
    if not isinstance(settings, list) or not settings:
        raise SystemExit("Config field `settings` must define A2/A3/A4.")
    if smoke_setting:
        settings = [item for item in settings if resolve_str(item.get("setting_id"), "").upper() == smoke_setting]
        if not settings:
            raise SystemExit(f"Smoke setting `{smoke_setting}` not found in config settings.")

    for setting in settings:
        setting_id = resolve_str(setting.get("setting_id"), "")
        setting_name = resolve_str(setting.get("setting_name"), setting_id.lower())
        dataset_root = dataset_view_root / setting_name
        score_csv = score_output_dir / f"{setting_id}_candidate_pool_scores.csv"
        summary_dir = materials_root / setting_name

        if audit["core_inputs_ready"] or args.dry_run:
            maybe_build_dataset(
                source_dataset=source_dataset,
                dataset_root=dataset_root,
                score_csv=score_csv,
                setting_id=setting_id,
                setting_name=setting_name,
                expected_duplication_total=fixed_budget_count,
                link_mode=resolve_str(cfg.get("link_mode"), "hardlink"),
                dry_run=args.dry_run,
                rerun=args.rerun,
                recycle_root=recycle_root,
            )
        else:
            print_step("preflight", f"skip dataset build for {setting_id} because required input files are missing")

    preflight_mode = "dry_run" if args.dry_run else ("preflight_only" if args.preflight_only else ("smoke" if args.smoke_epochs > 0 else "full_run"))
    preflight = build_preflight_report(
        cfg=cfg,
        settings=settings,
        audit=audit,
        hn_summary_csv=hn_summary_csv,
        teacher_summary_dir=teacher_summary_dir,
        uniform_anchor_summary_dir=uniform_anchor_summary_dir,
        score_output_dir=score_output_dir,
        dataset_view_root=dataset_view_root,
        results_dir=results_dir,
        materials_root=materials_root,
        project_root=project_root,
        recycle_root=recycle_root,
        scratch_root=scratch_root,
        temp_config_dir=temp_config_dir,
        preflight_mode=preflight_mode,
        smoke_epochs=int(args.smoke_epochs),
        smoke_setting=smoke_setting,
    )
    print_step("preflight", f"ready_for_full_train={preflight['ready_for_full_train']}")

    if args.preflight_only:
        return

    if not preflight["ready_for_full_train"] and not args.dry_run:
        raise SystemExit(
            "Preflight failed. Inspect research/results/stage1_formal/gate_info_sampling_lite/"
            "PREFLIGHT_gate_info_sampling_lite.md before smoke/full runs."
        )

    epochs_override = int(args.smoke_epochs) if args.smoke_epochs > 0 else None

    for setting in settings:
        setting_id = resolve_str(setting.get("setting_id"), "")
        setting_name = resolve_str(setting.get("setting_name"), setting_id.lower())
        dataset_root = dataset_view_root / setting_name
        summary_dir = materials_root / setting_name

        suite_rows.append(
            run_setting(
                cfg=cfg,
                config_path=config_path,
                setting=setting,
                teacher_best=teacher_best,
                dataset_root=dataset_root,
                project_root=project_root,
                summary_dir=summary_dir,
                temp_config_dir=temp_config_dir,
                split_csv=split_csv,
                epochs_override=epochs_override,
                dry_run=args.dry_run,
                rerun=args.rerun,
                recycle_root=recycle_root,
            )
        )

    write_csv(
        results_dir / "suite_context.csv",
        ["setting_id", "setting_name", "run_name", "run_dir", "summary_dir", "status"],
        suite_rows,
    )
    write_json(
        results_dir / "suite_context.json",
        {
            "task_name": resolve_str(cfg.get("task_name"), "stage1_formal_gate_info_sampling_lite"),
            "base_model": resolve_str(cfg.get("base_model"), "yolo11m-cls"),
            "teacher_ratio_id": audit["teacher_ratio_id"],
            "uniform_anchor_ratio_id": audit["uniform_anchor_ratio_id"],
            "budget_anchor_ratio_id": audit["budget_anchor_ratio_id"],
            "fixed_budget_count": fixed_budget_count,
            "mode": preflight_mode,
            "smoke_epochs": int(args.smoke_epochs),
            "smoke_setting": smoke_setting,
            "score_output_dir": str(score_output_dir),
            "materials_root": str(materials_root),
            "results_dir": str(results_dir),
            "dataset_view_root": str(dataset_view_root),
            "project_root": str(project_root),
            "rows": suite_rows,
        },
    )
    (results_dir / "suite_context.md").write_text(
        "\n".join(
            [
                "# Info-Sampling Lite Suite Context",
                "",
                f"- base model: `{resolve_str(cfg.get('base_model'), 'yolo11m-cls')}`",
                f"- teacher ratio: `{audit['teacher_ratio_id']}`",
                f"- uniform anchor: `{audit['uniform_anchor_ratio_id']}`",
                f"- fixed budget count: `{fixed_budget_count}`",
                "",
                "| Setting | Status | Summary Dir |",
                "| --- | --- | --- |",
                *[
                    f"| {row['setting_id']} / {row['setting_name']} | {row['status']} | {row['summary_dir']} |"
                    for row in suite_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if args.smoke_epochs > 0:
        write_json(
            results_dir / "SMOKE_gate_info_sampling_lite.json",
            {
                "mode": "smoke",
                "smoke_epochs": int(args.smoke_epochs),
                "smoke_setting": smoke_setting,
                "preflight_ready_for_full_train": preflight["ready_for_full_train"],
                "suite_rows": suite_rows,
            },
        )
        (results_dir / "SMOKE_gate_info_sampling_lite.md").write_text(
            "\n".join(
                [
                    "# Smoke Gate Info Sampling Lite",
                    "",
                    f"- smoke_setting: `{smoke_setting}`",
                    f"- smoke_epochs: `{int(args.smoke_epochs)}`",
                    f"- preflight ready_for_full_train: `{preflight['ready_for_full_train']}`",
                    "",
                    "| Setting | Status | Summary Dir |",
                    "| --- | --- | --- |",
                    *[
                        f"| {row['setting_id']} / {row['setting_name']} | {row['status']} | {row['summary_dir']} |"
                        for row in suite_rows
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return

    run_python(
        "scripts/stage1_build_info_sampling_lite_paper_assets.py",
        [
            "--config",
            str(config_path),
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
