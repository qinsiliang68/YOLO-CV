from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
YOLOV11_ROOT = REPO_ROOT / "YOLOv11"
DEFAULT_ENTRY_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "main_entry.json"
BUILTIN_STAGE1_ENTRY_CONFIG = {
    "task": "stage1_gate_rcis_suite",
    "score_device": "0",
    "top_k": 22,
    "score_batch": 1,
    "score_chunk_size": 32,
    "ptsg_eval_config": r"YOLOv11\configs\runtime\stage1_gate_ptsg_eval.json",
    "ptsg_nextwave_config": r"YOLOv11\configs\runtime\stage1_gate_ptsg_nextwave.json",
    "stage1_embed_supcon_config": r"YOLOv11\configs\runtime\stage1_gate_embedding_supcon_eval.json",
    "stage1_maxfilter_suite_config": r"YOLOv11\configs\runtime\stage1_gate_maxfilter_suite.json",
    "stage1_rcis_suite_config": r"YOLOv11\configs\runtime\stage1_gate_rcis_suite.json",
    "stage1_formal_gate_capacity_config": r"YOLOv11\configs\runtime\stage1_formal_gate_capacity.json",
    "stage1_formal_cls6_capacity_config": r"YOLOv11\configs\runtime\stage1_formal_cls6_capacity.json",
    "stage1_formal_gate_hn_n_sweep_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_n_sweep.json",
    "stage1_formal_gate_hn_s_sweep_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_s_sweep.json",
    "stage1_formal_gate_hn_m_sweep_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_m_sweep.json",
    "stage1_formal_gate_hn_l_sweep_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_l_sweep.json",
    "stage1_formal_gate_hn_x_sweep_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_x_sweep.json",
    "stage1_formal_gate_hn_x_crosscheck_config": r"YOLOv11\configs\runtime\stage1_formal_gate_hn_x_crosscheck.json",
    "stage1_formal_gate_info_sampling_lite_config": r"YOLOv11\configs\runtime\stage1_formal_gate_info_sampling_lite.json",
    "stage1_formal_gate_bucket_pilot_config": r"YOLOv11\configs\runtime\stage1_formal_gate_bucket_pilot.json",
    "stage1_formal_gate_value_g1_config": r"YOLOv11\configs\runtime\stage1_formal_gate_value_g1.json",
    "stage1_formal_gate_value_g2_config": r"YOLOv11\configs\runtime\stage1_formal_gate_value_g2.json",
    "stage1_formal_gate_value_g3_config": r"YOLOv11\configs\runtime\stage1_formal_gate_value_g3.json",
    "stage1_formal_gate_value_g4_config": r"YOLOv11\configs\runtime\stage1_formal_gate_value_g4.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the current YOLO-CV training task."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Config JSON for the selected task. When omitted, the active entry config is used.",
    )
    parser.add_argument(
        "--task",
        choices=(
            "auto",
            "stage1_gate_ptsg_eval",
            "stage1_gate_ptsg_nextwave",
            "stage1_gate_embed_supcon",
            "stage1_gate_maxfilter_suite",
            "stage1_gate_rcis_suite",
            "stage1_formal_gate_capacity",
            "stage1_formal_cls6_capacity",
            "stage1_formal_gate_hn_n_sweep",
            "stage1_formal_gate_hn_s_sweep",
            "stage1_formal_gate_hn_m_sweep",
            "stage1_formal_gate_hn_l_sweep",
            "stage1_formal_gate_hn_x_sweep",
            "stage1_formal_gate_hn_x_crosscheck",
            "stage1_formal_gate_hn_all",
            "stage1_formal_gate_hn_ns_all",
            "stage1_formal_gate_info_sampling_lite",
            "stage1_formal_gate_bucket_pilot",
            "stage1_formal_gate_value_g1",
            "stage1_formal_gate_value_g2",
            "stage1_formal_gate_value_g3",
            "stage1_formal_gate_value_g4",
            "stage1_formal_gate_value_g0_g4_all",
        ),
        default="auto",
        help="Task to run. 'auto' reads YOLOv11/configs/runtime/main_entry.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without training.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing runs and rerun all configured models.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight validation for task pipelines that support it.")
    parser.add_argument("--smoke-epochs", type=int, default=0, help="Run an isolated smoke test with a short epoch count for supported tasks.")
    parser.add_argument("--smoke-setting", default="", help="Optional setting id for smoke tests, e.g. A4.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def load_entry_config(path: Path) -> dict:
    config = dict(BUILTIN_STAGE1_ENTRY_CONFIG)
    if path.exists():
        config.update(load_json(path))
        print_step("config", f"loaded entry config: {path}")
    else:
        print_step("config", f"missing entry config, using built-in defaults: {path}")
    return config


def resolve_path(value: str | None, *, base: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return base.resolve()
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_str(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def run_python(script: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script), *args]
    print_step("run", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    if os.name == "nt":
        env.setdefault("PIN_MEMORY", "False")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def archive_existing_run(run_dir: Path, recycle_root: Path, dry_run: bool) -> None:
    if not run_dir.exists():
        return
    destination = recycle_root / run_dir.name
    print_step("archive", f"{run_dir} -> {destination}")
    if dry_run:
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))


def run_stage1_ptsg(entry_cfg: dict, dry_run: bool) -> None:
    config_path = resolve_path(
        entry_cfg.get("ptsg_eval_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_ptsg_eval.json",
    )
    ptsg_cfg = load_json(config_path)
    output_dir = resolve_path(
        ptsg_cfg.get("output_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_ptsg" / "yolo11l_gate2_hn02",
    )
    label = resolve_str(ptsg_cfg.get("label"), Path(resolve_str(ptsg_cfg.get("weights"), "stage1_ptsg")).stem)

    train_features_csv = output_dir / "train_features.csv"
    train_embeddings_npy = output_dir / "train_embeddings.npy"
    val_features_csv = output_dir / "val_features.csv"
    val_embeddings_npy = output_dir / "val_embeddings.npy"

    print_step("task", f"stage1_gate_ptsg_eval ({label})")
    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            resolve_str(ptsg_cfg.get("weights"), ""),
            "--data-root",
            resolve_str(ptsg_cfg.get("data_root"), ""),
            "--output-dir",
            str(output_dir),
            "--device",
            resolve_str(ptsg_cfg.get("device"), "0"),
            "--imgsz",
            str(int(ptsg_cfg.get("imgsz", 640) or 640)),
            "--batch",
            str(int(ptsg_cfg.get("batch", 4) or 4)),
            "--chunk-size",
            str(int(ptsg_cfg.get("chunk_size", 32) or 32)),
            "--normal-class",
            resolve_str(ptsg_cfg.get("normal_class"), "Normal"),
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_build_ptsg_bank.py",
        [
            "--train-features-csv",
            str(train_features_csv),
            "--train-embeddings-npy",
            str(train_embeddings_npy),
            "--output-dir",
            str(output_dir),
            "--normal-class",
            resolve_str(ptsg_cfg.get("normal_class"), "Normal"),
            "--hn-manifest",
            resolve_str(ptsg_cfg.get("hn_manifest"), ""),
            "--hn-weight",
            str(float(ptsg_cfg.get("hn_weight", 3.0) or 3.0)),
        ],
        dry_run=dry_run,
    )
    eval_args = [
        "--val-features-csv",
        str(val_features_csv),
        "--val-embeddings-npy",
        str(val_embeddings_npy),
        "--val-split-csv",
        resolve_str(ptsg_cfg.get("split_csv"), ""),
        "--normal-proto",
        str(output_dir / "normal_proto.npy"),
        "--abnormal-proto",
        str(output_dir / "abnormal_proto.npy"),
        "--output-dir",
        str(output_dir),
        "--normal-class",
        resolve_str(ptsg_cfg.get("normal_class"), "Normal"),
        "--alpha",
        str(float(ptsg_cfg.get("alpha", 1.0) or 1.0)),
        "--beta",
        str(float(ptsg_cfg.get("beta", 1.0) or 1.0)),
        "--gamma",
        str(float(ptsg_cfg.get("gamma", 0.5) or 0.5)),
    ]
    hn_proto = output_dir / "normal_proto_hn_aware.npy"
    if dry_run or hn_proto.exists():
        eval_args.extend(["--hn-aware-normal-proto", str(hn_proto)])
    run_python("scripts/stage1_eval_ptsg.py", eval_args, dry_run=dry_run)


def run_stage1_ptsg_nextwave(entry_cfg: dict, dry_run: bool) -> None:
    config_path = resolve_path(
        entry_cfg.get("ptsg_nextwave_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_ptsg_nextwave.json",
    )
    ptsg_cfg = load_json(config_path)
    output_dir = resolve_path(
        ptsg_cfg.get("output_dir"),
        base=REPO_ROOT / "research" / "results" / "stage1_ptsg_nextwave",
    )
    label = resolve_str(ptsg_cfg.get("label"), "yolo11l-cls + hn02 multi-prototype trust")

    print_step("task", f"stage1_gate_ptsg_nextwave ({label})")
    run_python(
        "scripts/stage1_eval_ptsg_nextwave.py",
        [
            "--train-features-csv",
            resolve_str(ptsg_cfg.get("train_features_csv"), ""),
            "--train-embeddings-npy",
            resolve_str(ptsg_cfg.get("train_embeddings_npy"), ""),
            "--val-features-csv",
            resolve_str(ptsg_cfg.get("val_features_csv"), ""),
            "--val-embeddings-npy",
            resolve_str(ptsg_cfg.get("val_embeddings_npy"), ""),
            "--val-split-csv",
            resolve_str(ptsg_cfg.get("split_csv"), ""),
            "--output-dir",
            str(output_dir),
            "--normal-class",
            resolve_str(ptsg_cfg.get("normal_class"), "Normal"),
            "--alpha",
            str(float(ptsg_cfg.get("alpha", 1.0) or 1.0)),
            "--beta",
            str(float(ptsg_cfg.get("beta", 1.0) or 1.0)),
            "--delta",
            str(float(ptsg_cfg.get("delta", 0.5) or 0.5)),
            "--seed",
            str(int(ptsg_cfg.get("seed", 20260330) or 20260330)),
            "--kmeans-max-iter",
            str(int(ptsg_cfg.get("kmeans_max_iter", 50) or 50)),
            "--temperature-max-iter",
            str(int(ptsg_cfg.get("temperature_max_iter", 200) or 200)),
            "--k-values",
            *[str(int(value)) for value in ptsg_cfg.get("k_values", [4, 8])],
        ],
        dry_run=dry_run,
    )


def run_stage1_ptsg_material_eval(
    *,
    weights_path: Path,
    data_root: Path,
    output_dir: Path,
    split_csv: str,
    normal_class: str,
    device: str,
    imgsz: int,
    batch: int,
    chunk_size: int,
    alpha: float,
    beta: float,
    gamma: float,
    hn_manifest: str,
    hn_weight: float,
    dry_run: bool,
) -> None:
    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            str(weights_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--device",
            device,
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--chunk-size",
            str(chunk_size),
            "--normal-class",
            normal_class,
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_build_ptsg_bank.py",
        [
            "--train-features-csv",
            str(output_dir / "train_features.csv"),
            "--train-embeddings-npy",
            str(output_dir / "train_embeddings.npy"),
            "--output-dir",
            str(output_dir),
            "--normal-class",
            normal_class,
            "--hn-manifest",
            hn_manifest,
            "--hn-weight",
            str(hn_weight),
        ],
        dry_run=dry_run,
    )
    eval_args = [
        "--val-features-csv",
        str(output_dir / "val_features.csv"),
        "--val-embeddings-npy",
        str(output_dir / "val_embeddings.npy"),
        "--val-split-csv",
        split_csv,
        "--normal-proto",
        str(output_dir / "normal_proto.npy"),
        "--abnormal-proto",
        str(output_dir / "abnormal_proto.npy"),
        "--output-dir",
        str(output_dir),
        "--normal-class",
        normal_class,
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--gamma",
        str(gamma),
    ]
    hn_proto = output_dir / "normal_proto_hn_aware.npy"
    if dry_run or hn_proto.exists():
        eval_args.extend(["--hn-aware-normal-proto", str(hn_proto)])
    run_python("scripts/stage1_eval_ptsg.py", eval_args, dry_run=dry_run)


def run_stage1_maxfilter_suite(entry_cfg: dict, dry_run: bool) -> None:
    config_path = resolve_path(
        entry_cfg.get("stage1_maxfilter_suite_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_maxfilter_suite.json",
    )
    suite_cfg = load_json(config_path)
    print_step("task", f"stage1_gate_maxfilter_suite ({resolve_str(suite_cfg.get('label'), 'stage1 maxfilter suite')})")

    source_dataset = resolve_path(
        suite_cfg.get("source_dataset"),
        base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200",
    )
    base_hn_dataset = resolve_path(
        suite_cfg.get("base_hn_dataset"),
        base=YOLOV11_ROOT / "datasets" / "stage1_gate_hn_backflow" / "yolo11l_gate2_hn02",
    )
    miner_weights = resolve_path(
        suite_cfg.get("miner_weights"),
        base=YOLOV11_ROOT / "runs" / "cls_gate_hn_sweep" / "yolo11l_gate2_train7200_hn02" / "weights" / "best.pt",
    )
    scoring_output_dir = resolve_path(
        suite_cfg.get("scoring_output_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_gate_maxfilter" / "miner_yolo11l_gate2_hn02",
    )
    hard_mining_dataset = resolve_path(
        suite_cfg.get("hard_mining_dataset"),
        base=YOLOV11_ROOT / "datasets" / "stage1_gate_maxfilter" / "yolo11l_gate2_hn02_hardmix",
    )
    defect_oversample_dataset = resolve_path(
        suite_cfg.get("defect_oversample_dataset"),
        base=YOLOV11_ROOT / "datasets" / "stage1_gate_maxfilter" / "yolo11l_gate2_hn02_defectos",
    )
    baseline_dir = resolve_path(
        suite_cfg.get("baseline_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_ptsg" / "yolo11l_gate2_hn02",
    )
    results_dir = resolve_path(
        suite_cfg.get("results_dir"),
        base=REPO_ROOT / "research" / "results" / "stage1_gate_maxfilter_suite",
    )
    recycle_root = resolve_path(
        suite_cfg.get("recycle_root"),
        base=REPO_ROOT / "_recycle_bin" / "stage1_gate_maxfilter",
    )

    normal_class = resolve_str(suite_cfg.get("normal_class"), "Normal")
    device = resolve_str(suite_cfg.get("device"), "0")
    imgsz = int(suite_cfg.get("imgsz", 640) or 640)
    batch = int(suite_cfg.get("batch", 2) or 2)
    chunk_size = int(suite_cfg.get("chunk_size", 16) or 16)
    score_batch = int(suite_cfg.get("score_batch", 1) or 1)
    score_chunk_size = int(suite_cfg.get("score_chunk_size", 32) or 32)
    alpha = float(suite_cfg.get("alpha", 1.0) or 1.0)
    beta = float(suite_cfg.get("beta", 1.0) or 1.0)
    gamma = float(suite_cfg.get("gamma", 0.5) or 0.5)
    hn_manifest = resolve_str(suite_cfg.get("hn_manifest"), "")
    hn_weight = float(suite_cfg.get("hn_weight", 3.0) or 3.0)

    run_python(
        "scripts/stage1_score_train_samples.py",
        [
            "--weights",
            str(miner_weights),
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(scoring_output_dir),
            "--device",
            device,
            "--imgsz",
            str(imgsz),
            "--batch",
            str(score_batch),
            "--chunk-size",
            str(score_chunk_size),
            "--normal-class",
            normal_class,
        ],
        dry_run=dry_run,
    )

    scores_csv = scoring_output_dir / "train_sample_scores.csv"
    run_python(
        "scripts/stage1_build_augmented_gate_dataset.py",
        [
            "--source-dataset",
            str(source_dataset),
            "--scores-csv",
            str(scores_csv),
            "--output-dataset",
            str(hard_mining_dataset),
            "--normal-class",
            normal_class,
            "--hard-negative-top-k",
            str(int(suite_cfg.get("hard_negative_top_k", 22) or 22)),
            "--hard-negative-repeat",
            str(int(suite_cfg.get("hard_negative_repeat", 1) or 1)),
            "--hard-positive-top-k",
            str(int(suite_cfg.get("hard_positive_top_k", 22) or 22)),
            "--hard-positive-repeat",
            str(int(suite_cfg.get("hard_positive_repeat", 1) or 1)),
            "--abnormal-repeat-all",
            "0",
            "--link-mode",
            resolve_str(suite_cfg.get("link_mode"), "hardlink"),
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_build_augmented_gate_dataset.py",
        [
            "--source-dataset",
            str(source_dataset),
            "--scores-csv",
            str(scores_csv),
            "--output-dataset",
            str(defect_oversample_dataset),
            "--normal-class",
            normal_class,
            "--hard-negative-top-k",
            str(int(suite_cfg.get("hard_negative_top_k", 22) or 22)),
            "--hard-negative-repeat",
            str(int(suite_cfg.get("hard_negative_repeat", 1) or 1)),
            "--hard-positive-top-k",
            "0",
            "--hard-positive-repeat",
            "0",
            "--abnormal-repeat-all",
            str(int(suite_cfg.get("abnormal_repeat_all", 1) or 1)),
            "--link-mode",
            resolve_str(suite_cfg.get("link_mode"), "hardlink"),
        ],
        dry_run=dry_run,
    )

    compare_args = [
        "--baseline-dir",
        str(baseline_dir),
        "--baseline-label",
        "H0 current best hn02 + P2",
        "--output-dir",
        str(results_dir),
    ]

    for experiment in suite_cfg.get("experiments", []):
        train_config_path = resolve_path(
            experiment.get("train_config"),
            base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_l_hn_selective.json",
        )
        train_cfg = load_json(train_config_path)
        run_project = resolve_path(train_cfg.get("project"), base=YOLOV11_ROOT / "runs" / "stage1_gate_maxfilter")
        run_name = resolve_str(train_cfg.get("name"), train_config_path.stem)
        run_dir = run_project / run_name
        weights_path = run_dir / "weights" / "best.pt"
        data_root = resolve_path(train_cfg.get("data"), base=base_hn_dataset)
        materials_dir = resolve_path(
            experiment.get("materials_dir"),
            base=REPO_ROOT / "research" / "materials" / "stage1_gate_maxfilter" / run_name,
        )

        archive_existing_run(run_dir, recycle_root, dry_run=dry_run)
        run_python("scripts/stage1_gate_train.py", ["--config", str(train_config_path)], dry_run=dry_run)
        run_stage1_ptsg_material_eval(
            weights_path=weights_path,
            data_root=data_root,
            output_dir=materials_dir,
            split_csv=resolve_str(suite_cfg.get("split_csv"), ""),
            normal_class=normal_class,
            device=device,
            imgsz=imgsz,
            batch=batch,
            chunk_size=chunk_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hn_manifest=hn_manifest,
            hn_weight=hn_weight,
            dry_run=dry_run,
        )
        compare_args.extend(["--experiment", f"{resolve_str(experiment.get('label'), run_name)}::{materials_dir}"])

    run_python("scripts/stage1_compare_maxfilter_suite.py", compare_args, dry_run=dry_run)


def run_stage1_rcis_suite(entry_cfg: dict, dry_run: bool) -> None:
    config_path = resolve_path(
        entry_cfg.get("stage1_rcis_suite_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_rcis_suite.json",
    )
    suite_cfg = load_json(config_path)
    print_step("task", f"stage1_gate_rcis_suite ({resolve_str(suite_cfg.get('label'), 'stage1 RCIS suite')})")

    source_dataset = resolve_path(
        suite_cfg.get("source_dataset"),
        base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200",
    )
    miner_weights = resolve_path(
        suite_cfg.get("miner_weights"),
        base=YOLOV11_ROOT / "runs" / "stage1_gate_maxfilter" / "yolo11l_gate2_hn02_hardmix" / "weights" / "best.pt",
    )
    baseline_dir = resolve_path(
        suite_cfg.get("baseline_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_gate_maxfilter" / "yolo11l_gate2_hn02_hardmix",
    )
    scoring_output_dir = resolve_path(
        suite_cfg.get("scoring_output_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_gate_rcis" / "miner_yolo11l_gate2_hn02_hardmix",
    )
    feature_output_dir = resolve_path(
        suite_cfg.get("feature_output_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_gate_rcis" / "source_features_yolo11l_gate2_hn02_hardmix",
    )
    results_dir = resolve_path(
        suite_cfg.get("results_dir"),
        base=REPO_ROOT / "research" / "results" / "stage1_gate_rcis_suite",
    )
    recycle_root = resolve_path(
        suite_cfg.get("recycle_root"),
        base=REPO_ROOT / "_recycle_bin" / "stage1_gate_rcis",
    )

    normal_class = resolve_str(suite_cfg.get("normal_class"), "Normal")
    device = resolve_str(suite_cfg.get("device"), "0")
    imgsz = int(suite_cfg.get("imgsz", 640) or 640)
    batch = int(suite_cfg.get("batch", 2) or 2)
    chunk_size = int(suite_cfg.get("chunk_size", 16) or 16)
    score_batch = int(suite_cfg.get("score_batch", 1) or 1)
    score_chunk_size = int(suite_cfg.get("score_chunk_size", 32) or 32)
    alpha = float(suite_cfg.get("alpha", 1.0) or 1.0)
    beta = float(suite_cfg.get("beta", 1.0) or 1.0)
    gamma = float(suite_cfg.get("gamma", 0.5) or 0.5)
    hn_manifest = resolve_str(suite_cfg.get("hn_manifest"), "")
    hn_weight = float(suite_cfg.get("hn_weight", 3.0) or 3.0)
    reference_variant = resolve_str(suite_cfg.get("reference_variant"), "P0")

    run_python(
        "scripts/stage1_score_train_samples.py",
        [
            "--weights",
            str(miner_weights),
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(scoring_output_dir),
            "--device",
            device,
            "--imgsz",
            str(imgsz),
            "--batch",
            str(score_batch),
            "--chunk-size",
            str(score_chunk_size),
            "--normal-class",
            normal_class,
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_export_gate_features.py",
        [
            "--weights",
            str(miner_weights),
            "--data-root",
            str(source_dataset),
            "--output-dir",
            str(feature_output_dir),
            "--device",
            device,
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--chunk-size",
            str(chunk_size),
            "--normal-class",
            normal_class,
        ],
        dry_run=dry_run,
    )

    compare_args = [
        "--baseline-dir",
        str(baseline_dir),
        "--baseline-label",
        "G4 current best HardMix + P0",
        "--output-dir",
        str(results_dir),
    ]

    for experiment in suite_cfg.get("experiments", []):
        dataset_root = resolve_path(
            experiment.get("dataset"),
            base=YOLOV11_ROOT / "datasets" / "stage1_gate_rcis" / resolve_str(experiment.get("name"), "rcis"),
        )
        materials_dir = resolve_path(
            experiment.get("materials_dir"),
            base=REPO_ROOT / "research" / "materials" / "stage1_gate_rcis" / resolve_str(experiment.get("name"), "rcis"),
        )
        train_config_path = resolve_path(
            experiment.get("train_config"),
            base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_l_rcis_full.json",
        )
        train_cfg = load_json(train_config_path)
        run_project = resolve_path(train_cfg.get("project"), base=YOLOV11_ROOT / "runs" / "stage1_gate_rcis")
        run_name = resolve_str(train_cfg.get("name"), train_config_path.stem)
        run_dir = run_project / run_name
        weights_path = run_dir / "weights" / "best.pt"
        rcis_args = experiment.get("rcis_args") or {}

        build_args = [
            "--source-dataset",
            str(source_dataset),
            "--scores-csv",
            str(scoring_output_dir / "train_sample_scores.csv"),
            "--train-features-csv",
            str(feature_output_dir / "train_features.csv"),
            "--train-embeddings-npy",
            str(feature_output_dir / "train_embeddings.npy"),
            "--reference-material-dir",
            str(baseline_dir),
            "--output-dataset",
            str(dataset_root),
            "--normal-class",
            normal_class,
            "--reference-variant",
            reference_variant,
            "--normal-clusters",
            str(int(suite_cfg.get("normal_clusters", 24) or 24)),
            "--abnormal-clusters",
            str(int(suite_cfg.get("abnormal_clusters", 12) or 12)),
            "--cluster-iter",
            str(int(suite_cfg.get("cluster_iter", 30) or 30)),
            "--sigmoid-gain",
            str(float(suite_cfg.get("sigmoid_gain", 4.0) or 4.0)),
            "--normal-wmin",
            str(float(suite_cfg.get("normal_wmin", 0.25) or 0.25)),
            "--normal-wmax",
            str(float(suite_cfg.get("normal_wmax", 3.0) or 3.0)),
            "--abnormal-wmin",
            str(float(suite_cfg.get("abnormal_wmin", 1.0) or 1.0)),
            "--abnormal-wmax",
            str(float(suite_cfg.get("abnormal_wmax", 4.0) or 4.0)),
            "--link-mode",
            resolve_str(suite_cfg.get("link_mode"), "hardlink"),
            "--seed",
            str(int(suite_cfg.get("seed", 20260330) or 20260330)),
        ]
        for key, value in rcis_args.items():
            build_args.extend([f"--{str(key).replace('_', '-')}", str(value)])

        run_python("scripts/stage1_build_rcis_dataset.py", build_args, dry_run=dry_run)
        archive_existing_run(run_dir, recycle_root, dry_run=dry_run)
        run_python("scripts/stage1_gate_train.py", ["--config", str(train_config_path)], dry_run=dry_run)
        run_stage1_ptsg_material_eval(
            weights_path=weights_path,
            data_root=dataset_root,
            output_dir=materials_dir,
            split_csv=resolve_str(suite_cfg.get("split_csv"), ""),
            normal_class=normal_class,
            device=device,
            imgsz=imgsz,
            batch=batch,
            chunk_size=chunk_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hn_manifest=hn_manifest,
            hn_weight=hn_weight,
            dry_run=dry_run,
        )
        compare_args.extend(["--experiment", f"{resolve_str(experiment.get('label'), run_name)}::{materials_dir}"])

    run_python("scripts/stage1_compare_rcis_suite.py", compare_args, dry_run=dry_run)


def run_stage1_formal_capacity(entry_cfg: dict, *, task_kind: str, dry_run: bool, rerun: bool) -> None:
    key = "stage1_formal_gate_capacity_config" if task_kind == "gate" else "stage1_formal_cls6_capacity_config"
    base_name = "stage1_formal_gate_capacity.json" if task_kind == "gate" else "stage1_formal_cls6_capacity.json"
    config_path = resolve_path(
        entry_cfg.get(key),
        base=YOLOV11_ROOT / "configs" / "runtime" / base_name,
    )
    run_python(
        "scripts/stage1_formal_capacity_suite.py",
        [
            "--config",
            str(config_path),
            "--task-kind",
            task_kind,
            *(["--dry-run"] if dry_run else []),
            *(["--rerun"] if rerun else []),
        ],
        dry_run=False,
    )


def run_stage1_formal_hn_suite(entry_cfg: dict, *, variant: str, dry_run: bool, rerun: bool) -> None:
    variant_map = {
        "n": ("stage1_formal_gate_hn_n_sweep_config", "stage1_formal_gate_hn_n_sweep.json"),
        "s": ("stage1_formal_gate_hn_s_sweep_config", "stage1_formal_gate_hn_s_sweep.json"),
        "m": ("stage1_formal_gate_hn_m_sweep_config", "stage1_formal_gate_hn_m_sweep.json"),
        "l": ("stage1_formal_gate_hn_l_sweep_config", "stage1_formal_gate_hn_l_sweep.json"),
        "x": ("stage1_formal_gate_hn_x_sweep_config", "stage1_formal_gate_hn_x_sweep.json"),
        "x_crosscheck": ("stage1_formal_gate_hn_x_crosscheck_config", "stage1_formal_gate_hn_x_crosscheck.json"),
    }
    if variant not in variant_map:
        raise SystemExit(f"Unsupported formal HN variant: {variant}")
    key, base_name = variant_map[variant]
    config_path = resolve_path(
        entry_cfg.get(key),
        base=YOLOV11_ROOT / "configs" / "runtime" / base_name,
    )
    run_python(
        "scripts/stage1_formal_hn_sweep.py",
        [
            "--config",
            str(config_path),
            *(["--dry-run"] if dry_run else []),
            *(["--rerun"] if rerun else []),
        ],
        dry_run=False,
    )


def run_stage1_formal_hn_all(entry_cfg: dict, *, dry_run: bool, rerun: bool) -> None:
    run_stage1_formal_hn_suite(entry_cfg, variant="m", dry_run=dry_run, rerun=rerun)
    run_stage1_formal_hn_suite(entry_cfg, variant="x_crosscheck", dry_run=dry_run, rerun=rerun)


def run_stage1_formal_hn_ns_all(entry_cfg: dict, *, dry_run: bool, rerun: bool) -> None:
    run_stage1_formal_hn_suite(entry_cfg, variant="n", dry_run=dry_run, rerun=rerun)
    run_stage1_formal_hn_suite(entry_cfg, variant="s", dry_run=dry_run, rerun=rerun)


def run_stage1_formal_gate_info_sampling_lite(
    entry_cfg: dict,
    *,
    dry_run: bool,
    rerun: bool,
    preflight_only: bool,
    smoke_epochs: int,
    smoke_setting: str,
) -> None:
    config_path = resolve_path(
        entry_cfg.get("stage1_formal_gate_info_sampling_lite_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_formal_gate_info_sampling_lite.json",
    )
    run_python(
        "scripts/stage1_formal_gate_info_sampling_lite.py",
        [
            "--config",
            str(config_path),
            *(["--dry-run"] if dry_run else []),
            *(["--rerun"] if rerun else []),
            *(["--preflight-only"] if preflight_only else []),
            *(["--smoke-epochs", str(int(smoke_epochs))] if int(smoke_epochs) > 0 else []),
            *(["--smoke-setting", smoke_setting] if str(smoke_setting).strip() else []),
        ],
        dry_run=False,
    )


def run_stage1_formal_gate_bucket_pilot(
    entry_cfg: dict,
    *,
    dry_run: bool,
    rerun: bool,
    preflight_only: bool,
) -> None:
    config_path = resolve_path(
        entry_cfg.get("stage1_formal_gate_bucket_pilot_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_formal_gate_bucket_pilot.json",
    )
    run_python(
        "scripts/stage1_formal_gate_bucket_pilot.py",
        [
            "--config",
            str(config_path),
            *(["--dry-run"] if dry_run else []),
            *(["--rerun"] if rerun else []),
            *(["--preflight-only"] if preflight_only else []),
        ],
        dry_run=False,
    )


def run_stage1_formal_gate_value_control(
    entry_cfg: dict,
    *,
    variant: str,
    dry_run: bool,
    rerun: bool,
    preflight_only: bool,
    smoke_epochs: int,
    smoke_setting: str,
) -> None:
    variant_map = {
        "g1": ("stage1_formal_gate_value_g1_config", "stage1_formal_gate_value_g1.json"),
        "g2": ("stage1_formal_gate_value_g2_config", "stage1_formal_gate_value_g2.json"),
        "g3": ("stage1_formal_gate_value_g3_config", "stage1_formal_gate_value_g3.json"),
        "g4": ("stage1_formal_gate_value_g4_config", "stage1_formal_gate_value_g4.json"),
    }
    if variant not in variant_map:
        raise SystemExit(f"Unsupported gate value control variant: {variant}")
    key, base_name = variant_map[variant]
    config_path = resolve_path(
        entry_cfg.get(key),
        base=YOLOV11_ROOT / "configs" / "runtime" / base_name,
    )
    run_python(
        "scripts/stage1_formal_gate_info_sampling_lite.py",
        [
            "--config",
            str(config_path),
            *(["--dry-run"] if dry_run else []),
            *(["--rerun"] if rerun else []),
            *(["--preflight-only"] if preflight_only else []),
            *(["--smoke-epochs", str(int(smoke_epochs))] if int(smoke_epochs) > 0 else []),
            *(["--smoke-setting", smoke_setting] if str(smoke_setting).strip() else []),
        ],
        dry_run=False,
    )


def run_stage1_formal_gate_value_controls_all(
    entry_cfg: dict,
    *,
    dry_run: bool,
    rerun: bool,
    preflight_only: bool,
) -> None:
    for variant in ("g1", "g2", "g3", "g4"):
        run_stage1_formal_gate_value_control(
            entry_cfg,
            variant=variant,
            dry_run=dry_run,
            rerun=rerun,
            preflight_only=preflight_only,
            smoke_epochs=0,
            smoke_setting="",
        )


def run_stage1_embed_supcon(entry_cfg: dict, dry_run: bool) -> None:
    config_path = resolve_path(
        entry_cfg.get("stage1_embed_supcon_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_embedding_supcon_eval.json",
    )
    embed_cfg = load_json(config_path)
    train_config_path = resolve_path(
        embed_cfg.get("train_config"),
        base=YOLOV11_ROOT / "configs" / "runtime" / "stage1_gate_l_hn02_supcon.json",
    )
    train_cfg = load_json(train_config_path)

    project_dir = Path(resolve_str(train_cfg.get("project"), ""))
    run_name = resolve_str(train_cfg.get("name"), "yolo11l_gate2_hn02_supcon")
    run_dir = project_dir / run_name
    recycle_root = resolve_path(
        embed_cfg.get("recycle_root"),
        base=REPO_ROOT / "_recycle_bin" / "stage1_gate_embed",
    )
    output_dir = resolve_path(
        embed_cfg.get("output_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_embedding_gate" / run_name,
    )
    baseline_dir = resolve_path(
        embed_cfg.get("baseline_dir"),
        base=REPO_ROOT / "research" / "materials" / "stage1_ptsg" / "yolo11l_gate2_hn02",
    )
    results_dir = resolve_path(
        embed_cfg.get("results_dir"),
        base=REPO_ROOT / "research" / "results" / "stage1_embedding_gate",
    )
    weights_path = run_dir / "weights" / "best.pt"

    print_step("task", f"stage1_gate_embed_supcon ({resolve_str(embed_cfg.get('label'), run_name)})")
    archive_existing_run(run_dir, recycle_root, dry_run=dry_run)
    run_python("scripts/stage1_gate_train.py", ["--config", str(train_config_path)], dry_run=dry_run)
    run_stage1_ptsg_material_eval(
        weights_path=weights_path,
        data_root=resolve_path(train_cfg.get("data"), base=YOLOV11_ROOT / "datasets" / "stage1_gate_hn_backflow" / "yolo11l_gate2_hn02"),
        output_dir=output_dir,
        split_csv=resolve_str(embed_cfg.get("split_csv"), ""),
        normal_class=resolve_str(embed_cfg.get("normal_class"), "Normal"),
        device=resolve_str(embed_cfg.get("device"), "0"),
        imgsz=int(embed_cfg.get("imgsz", 640) or 640),
        batch=int(embed_cfg.get("batch", 2) or 2),
        chunk_size=int(embed_cfg.get("chunk_size", 16) or 16),
        alpha=float(embed_cfg.get("alpha", 1.0) or 1.0),
        beta=float(embed_cfg.get("beta", 1.0) or 1.0),
        gamma=float(embed_cfg.get("gamma", 0.5) or 0.5),
        hn_manifest=resolve_str(embed_cfg.get("hn_manifest"), ""),
        hn_weight=float(embed_cfg.get("hn_weight", 3.0) or 3.0),
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_compare_embedding_gate.py",
        [
            "--baseline-dir",
            str(baseline_dir),
            "--candidate-dir",
            str(output_dir),
            "--output-dir",
            str(results_dir),
        ],
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()
    entry_config_path = resolve_path(args.config, base=DEFAULT_ENTRY_CONFIG) if args.config else DEFAULT_ENTRY_CONFIG
    entry_cfg = load_entry_config(entry_config_path)
    task_name = resolve_str(entry_cfg.get("task"), BUILTIN_STAGE1_ENTRY_CONFIG["task"]) if args.task == "auto" else args.task

    if task_name == "stage1_gate_ptsg_eval":
        run_stage1_ptsg(entry_cfg, dry_run=args.dry_run)
        return

    if task_name == "stage1_gate_ptsg_nextwave":
        run_stage1_ptsg_nextwave(entry_cfg, dry_run=args.dry_run)
        return

    if task_name == "stage1_gate_embed_supcon":
        run_stage1_embed_supcon(entry_cfg, dry_run=args.dry_run)
        return

    if task_name == "stage1_gate_maxfilter_suite":
        run_stage1_maxfilter_suite(entry_cfg, dry_run=args.dry_run)
        return

    if task_name == "stage1_gate_rcis_suite":
        run_stage1_rcis_suite(entry_cfg, dry_run=args.dry_run)
        return

    if task_name == "stage1_formal_gate_capacity":
        run_stage1_formal_capacity(entry_cfg, task_kind="gate", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_cls6_capacity":
        run_stage1_formal_capacity(entry_cfg, task_kind="cls6", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_m_sweep":
        run_stage1_formal_hn_suite(entry_cfg, variant="m", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_l_sweep":
        run_stage1_formal_hn_suite(entry_cfg, variant="l", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_x_sweep":
        run_stage1_formal_hn_suite(entry_cfg, variant="x", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_x_crosscheck":
        run_stage1_formal_hn_suite(entry_cfg, variant="x_crosscheck", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_all":
        run_stage1_formal_hn_all(entry_cfg, dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_n_sweep":
        run_stage1_formal_hn_suite(entry_cfg, variant="n", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_s_sweep":
        run_stage1_formal_hn_suite(entry_cfg, variant="s", dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_hn_ns_all":
        run_stage1_formal_hn_ns_all(entry_cfg, dry_run=args.dry_run, rerun=args.rerun)
        return

    if task_name == "stage1_formal_gate_info_sampling_lite":
        run_stage1_formal_gate_info_sampling_lite(
            entry_cfg,
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
            smoke_epochs=args.smoke_epochs,
            smoke_setting=args.smoke_setting,
        )
        return
    if task_name == "stage1_formal_gate_bucket_pilot":
        run_stage1_formal_gate_bucket_pilot(
            entry_cfg,
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
        )
        return
    if task_name == "stage1_formal_gate_value_g1":
        run_stage1_formal_gate_value_control(
            entry_cfg,
            variant="g1",
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
            smoke_epochs=args.smoke_epochs,
            smoke_setting=args.smoke_setting,
        )
        return
    if task_name == "stage1_formal_gate_value_g2":
        run_stage1_formal_gate_value_control(
            entry_cfg,
            variant="g2",
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
            smoke_epochs=args.smoke_epochs,
            smoke_setting=args.smoke_setting,
        )
        return
    if task_name == "stage1_formal_gate_value_g3":
        run_stage1_formal_gate_value_control(
            entry_cfg,
            variant="g3",
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
            smoke_epochs=args.smoke_epochs,
            smoke_setting=args.smoke_setting,
        )
        return
    if task_name == "stage1_formal_gate_value_g4":
        run_stage1_formal_gate_value_control(
            entry_cfg,
            variant="g4",
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
            smoke_epochs=args.smoke_epochs,
            smoke_setting=args.smoke_setting,
        )
        return
    if task_name == "stage1_formal_gate_value_g0_g4_all":
        run_stage1_formal_gate_value_controls_all(
            entry_cfg,
            dry_run=args.dry_run,
            rerun=args.rerun,
            preflight_only=args.preflight_only,
        )
        return
    raise SystemExit(f"Unsupported task: {task_name}")


if __name__ == "__main__":
    main()
