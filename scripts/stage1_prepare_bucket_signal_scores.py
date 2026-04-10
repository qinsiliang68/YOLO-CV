from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pipeline_common import ensure_yolov11_importable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Layer-1 single-signal bucket pilot inputs for stage-1 gate replay.")
    parser.add_argument("--teacher-weights", required=True, help="Teacher checkpoint used for one-shot scoring.")
    parser.add_argument("--teacher-best-manifest", required=True, help="Teacher best_epoch_manifest.json.")
    parser.add_argument("--source-dataset", required=True, help="Binary gate dataset root.")
    parser.add_argument("--pool-top-csv", required=True, help="Existing top-k hard-normal pool manifest.")
    parser.add_argument("--train-features-csv", required=True, help="Train feature CSV from stage1_export_gate_features.py.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Train embedding matrix from stage1_export_gate_features.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived bucket score files.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--candidate-top-k", type=int, default=250, help="Candidate pool size.")
    parser.add_argument("--fixed-budget-count", type=int, required=True, help="Fixed extra-normal budget aligned with HN14.")
    parser.add_argument("--bucket-count", type=int, default=5, help="Number of equal-width buckets per signal.")
    parser.add_argument("--density-k", type=int, default=15, help="k for cosine-neighbor density support.")
    parser.add_argument("--seed", type=int, default=20260330, help="Base random seed.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    return parser.parse_args()


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object expected: {path}")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_rel_path(text: str) -> str:
    return str(text).replace("\\", "/")


def image_id_from_rel_path(rel_path: str) -> str:
    path = Path(rel_path)
    return f"{path.parent.name}_{path.stem}"


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def build_tta_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    return [
        ("orig", rgb),
        ("bright_down", ImageEnhance.Brightness(rgb).enhance(0.92)),
        ("bright_up", ImageEnhance.Brightness(rgb).enhance(1.08)),
        ("contrast_down", ImageEnhance.Contrast(rgb).enhance(0.92)),
        ("blur_light", rgb.filter(ImageFilter.GaussianBlur(radius=0.8))),
    ]


def predict_variant_probs(
    model,
    variants: list[tuple[str, Image.Image]],
    imgsz: int,
    batch: int,
    device: str,
    normal_class: str,
) -> list[tuple[str, float]]:
    use_half = device.lower() != "cpu"
    sources = [np.asarray(image, dtype=np.uint8) for _name, image in variants]
    results = model.predict(
        source=sources,
        stream=False,
        verbose=False,
        imgsz=imgsz,
        batch=min(batch, len(sources)),
        device=device,
        half=use_half,
    )
    rows: list[tuple[str, float]] = []
    for (name, _image), result in zip(variants, results, strict=True):
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        class_names = result.names
        if isinstance(class_names, dict):
            normal_index = next((int(idx) for idx, class_name in class_names.items() if class_name == normal_class), None)
        else:
            normal_index = next((idx for idx, class_name in enumerate(class_names) if class_name == normal_class), None)
        if normal_index is None:
            raise SystemExit(f"Normal class '{normal_class}' not found in class names: {class_names}")
        p_normal = float(probs[normal_index])
        rows.append((name, float(1.0 - p_normal)))
    if use_half:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return rows


def density_support(embeddings: np.ndarray, k: int) -> np.ndarray:
    count = int(embeddings.shape[0])
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    if count == 1:
        return np.ones((1,), dtype=np.float32)
    normalized = embeddings.astype(np.float32, copy=False)
    normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k_eff = max(1, min(int(k), count - 1))
    top_indices = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_sim = np.zeros((count,), dtype=np.float32)
    for idx in range(count):
        mean_sim[idx] = float(np.mean(similarity[idx, top_indices[idx]]))
    return mean_sim


def robust_zscore_then_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty((0,), dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-6)
    z = (values - median) / scale
    min_val = float(np.min(z))
    max_val = float(np.max(z))
    if math.isclose(min_val, max_val, rel_tol=0.0, abs_tol=1e-9):
        return np.ones_like(z, dtype=np.float32)
    normalized = (z - min_val) / (max_val - min_val)
    return np.clip(normalized.astype(np.float32), 0.0, 1.0)


def deterministic_uniform_counts(n_items: int, total: int, seed: int) -> list[int]:
    if n_items <= 0:
        return []
    base = int(total) // int(n_items)
    remainder = int(total) % int(n_items)
    counts = [base for _ in range(n_items)]
    if remainder <= 0:
        return counts
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_items).tolist()
    for idx in order[:remainder]:
        counts[int(idx)] += 1
    return counts


def side_flag(prob: float, tau: float) -> int:
    return int(float(prob) >= float(tau))


def choose_anchor_tau(prob: float, tau_995: float, tau_990: float) -> tuple[str, float]:
    dist_995 = abs(float(prob) - float(tau_995))
    dist_990 = abs(float(prob) - float(tau_990))
    if dist_995 <= dist_990:
        return "tau_r995", float(tau_995)
    return "tau_r990", float(tau_990)


def main() -> None:
    args = parse_args()
    ensure_yolov11_importable()
    from ultralytics import YOLO

    source_dataset = Path(args.source_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_manifest = load_json(Path(args.teacher_best_manifest).resolve())
    tau_995 = float(teacher_manifest["tau_r995"])
    tau_990 = float(teacher_manifest["tau_r990"])
    temperature = float(teacher_manifest["temperature_T"])
    s_r = max(abs(tau_995 - tau_990), 1e-6)
    teacher_checkpoint = Path(args.teacher_weights).resolve()

    pool_top_rows = load_csv_rows(Path(args.pool_top_csv).resolve())
    feature_rows = load_csv_rows(Path(args.train_features_csv).resolve())
    embeddings = np.load(Path(args.train_embeddings_npy).resolve())

    candidate_rows = [
        {
            "img_path": str(row["img_path"]),
            "img_rel_path": normalize_rel_path(str(row["img_rel_path"])),
            "pool_score_anchor": float(row["p_abnormal"]),
            "pool_rank": index,
        }
        for index, row in enumerate(pool_top_rows[: int(args.candidate_top_k)], start=1)
    ]
    candidate_lookup = {row["img_rel_path"]: row for row in candidate_rows}

    feature_lookup: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        if str(row.get("split", "")) != "train":
            continue
        if str(row.get("gt_label", "")) != args.normal_class:
            continue
        rel_path = normalize_rel_path(str(row.get("img_rel_path", "")))
        if rel_path in candidate_lookup:
            feature_lookup[rel_path] = row

    if len(candidate_rows) != int(args.candidate_top_k):
        raise SystemExit(f"Expected {int(args.candidate_top_k)} pool rows, got {len(candidate_rows)}")
    if int(args.candidate_top_k) % int(args.bucket_count) != 0:
        raise SystemExit("--candidate-top-k must be divisible by --bucket-count for equal-width buckets.")

    model = YOLO(str(teacher_checkpoint), task="classify")
    master_rows: list[dict[str, Any]] = []
    candidate_embeddings: list[np.ndarray] = []
    print_step("data", f"scoring fixed pool of {len(candidate_rows)} hard-normal candidates")
    for pool_row in candidate_rows:
        rel_path = str(pool_row["img_rel_path"])
        image_path = source_dataset / Path(rel_path)
        if not image_path.exists():
            raise SystemExit(f"Candidate image missing: {image_path}")
        feature_row = feature_lookup.get(rel_path)
        if feature_row is None:
            raise SystemExit(f"Missing feature row for candidate: {rel_path}")
        embedding_index = int(feature_row["embedding_index"])
        candidate_embeddings.append(embeddings[embedding_index].astype(np.float32, copy=False))

        with Image.open(image_path) as image:
            variants = build_tta_variants(image)
        variant_probs = predict_variant_probs(model, variants, int(args.imgsz), int(args.batch), str(args.device), args.normal_class)
        raw_probs = [float(prob) for _name, prob in variant_probs]
        raw_logits = [safe_prob_to_logit(prob) for prob in raw_probs]
        cal_logits = [float(logit) / float(temperature) for logit in raw_logits]
        cal_probs = [sigmoid(value) for value in cal_logits]
        p_anchor = float(cal_probs[0])
        tau_label, tau_anchor = choose_anchor_tau(p_anchor, tau_995, tau_990)
        q_i = float(
            np.mean(
                [
                    float(side_flag(prob, tau_anchor) == side_flag(p_anchor, tau_anchor))
                    for prob in cal_probs
                ]
            )
        )
        logit_var = float(np.var(np.asarray(cal_logits, dtype=np.float32)))
        r_score = max(
            math.exp(-abs(p_anchor - tau_995) / s_r),
            math.exp(-abs(p_anchor - tau_990) / s_r),
        )
        master_rows.append(
            {
                "image_id": image_id_from_rel_path(rel_path),
                "img_rel_path": rel_path,
                "pool_rank": int(pool_row["pool_rank"]),
                "pool_score_anchor": round(float(pool_row["pool_score_anchor"]), 6),
                "embedding_index": embedding_index,
                "calibrated_p": round(p_anchor, 6),
                "tau_anchor_label": tau_label,
                "tau_anchor_value": round(float(tau_anchor), 6),
                "tta_side_consistency": round(q_i, 6),
                "tta_logit_variance": round(logit_var, 6),
                "R": round(float(r_score), 6),
            }
        )

    density_raw = density_support(np.stack(candidate_embeddings), int(args.density_k))
    density_scaled = robust_zscore_then_minmax(density_raw)
    variance_values = np.asarray([float(row["tta_logit_variance"]) for row in master_rows], dtype=np.float32)
    s_c = max(float(np.median(variance_values)), 1e-6)

    for row, d_raw, d_scaled in zip(master_rows, density_raw.tolist(), density_scaled.tolist(), strict=True):
        c_score = float(row["tta_side_consistency"]) * math.exp(-float(row["tta_logit_variance"]) / s_c)
        row["C"] = round(float(c_score), 6)
        row["D_raw"] = round(float(d_raw), 6)
        row["D"] = round(float(d_scaled), 6)

    master_fieldnames = [
        "image_id",
        "img_rel_path",
        "pool_rank",
        "pool_score_anchor",
        "embedding_index",
        "calibrated_p",
        "tau_anchor_label",
        "tau_anchor_value",
        "tta_side_consistency",
        "tta_logit_variance",
        "R",
        "C",
        "D_raw",
        "D",
    ]
    write_csv(output_dir / "candidate_signal_master.csv", master_fieldnames, master_rows)

    experiment_rows: list[dict[str, Any]] = []
    experiment_fieldnames = [
        "experiment_id",
        "signal",
        "bucket",
        "bucket_rank_start",
        "bucket_rank_end",
        "setting_name",
        "candidate_scores_csv",
        "metadata_json",
        "n_unique",
        "n_replay",
    ]
    score_fieldnames = [
        "setting_id",
        "setting_name",
        "signal",
        "bucket",
        "bucket_rank",
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
        "selected_flag",
    ]

    bucket_size = int(args.candidate_top_k) // int(args.bucket_count)
    experiments_dir = output_dir / "experiments"
    metadata_dir = output_dir / "metadata"
    signal_stats: dict[str, Any] = {
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_best_manifest": str(Path(args.teacher_best_manifest).resolve()),
        "tau_r995": tau_995,
        "tau_r990": tau_990,
        "temperature_T": temperature,
        "s_R": s_r,
        "s_C": s_c,
        "candidate_top_k": int(args.candidate_top_k),
        "bucket_count": int(args.bucket_count),
        "bucket_size": bucket_size,
        "density_k": int(args.density_k),
        "fixed_budget_count": int(args.fixed_budget_count),
        "seed": int(args.seed),
        "signals": {},
    }

    for signal_name in ("R", "C", "D"):
        ranked = sorted(master_rows, key=lambda row: (float(row[signal_name]), -int(row["pool_rank"])), reverse=True)
        signal_stats["signals"][signal_name] = {
            "top5_image_ids": [row["image_id"] for row in ranked[:5]],
            "bottom5_image_ids": [row["image_id"] for row in ranked[-5:]],
            "value_min": round(min(float(row[signal_name]) for row in ranked), 6),
            "value_max": round(max(float(row[signal_name]) for row in ranked), 6),
        }
        for bucket_index in range(int(args.bucket_count)):
            rank_start = bucket_index * bucket_size
            rank_end = rank_start + bucket_size
            bucket_rows = ranked[rank_start:rank_end]
            bucket_name = f"Q{bucket_index + 1}"
            experiment_id = f"{signal_name}-{bucket_name}"
            setting_name = f"{signal_name.lower()}_{bucket_name.lower()}_bucket_uniform151"
            counts = deterministic_uniform_counts(len(bucket_rows), int(args.fixed_budget_count), int(args.seed) + (100 * (bucket_index + 1)) + ord(signal_name))
            score_rows: list[dict[str, Any]] = []
            for local_rank, (row, duplication_count) in enumerate(zip(bucket_rows, counts, strict=True), start=1):
                score_rows.append(
                    {
                        "setting_id": experiment_id,
                        "setting_name": setting_name,
                        "signal": signal_name,
                        "bucket": bucket_name,
                        "bucket_rank": local_rank,
                        "image_id": row["image_id"],
                        "img_rel_path": row["img_rel_path"],
                        "pool_rank": row["pool_rank"],
                        "calibrated_p": row["calibrated_p"],
                        "R": row["R"],
                        "C": row["C"],
                        "D": row["D"],
                        "S": row[signal_name],
                        "pi": round(1.0 / len(bucket_rows), 8),
                        "duplication_count": int(duplication_count),
                        "selected_flag": int(duplication_count > 0),
                    }
                )
            candidate_scores_csv = experiments_dir / f"{experiment_id}_candidate_scores.csv"
            metadata_json = metadata_dir / f"{experiment_id}_bucket_metadata.json"
            write_csv(candidate_scores_csv, score_fieldnames, score_rows)
            write_json(
                metadata_json,
                {
                    "experiment_id": experiment_id,
                    "signal": signal_name,
                    "bucket": bucket_name,
                    "bucket_rank_start": rank_start + 1,
                    "bucket_rank_end": rank_end,
                    "n_unique": len(bucket_rows),
                    "n_replay": int(sum(counts)),
                    "selection_mode": "bucket_uniform_replay",
                    "score_source": str(output_dir / "candidate_signal_master.csv"),
                    "selected_image_ids": [row["image_id"] for row in bucket_rows],
                    "selected_rel_paths": [row["img_rel_path"] for row in bucket_rows],
                },
            )
            experiment_rows.append(
                {
                    "experiment_id": experiment_id,
                    "signal": signal_name,
                    "bucket": bucket_name,
                    "bucket_rank_start": rank_start + 1,
                    "bucket_rank_end": rank_end,
                    "setting_name": setting_name,
                    "candidate_scores_csv": str(candidate_scores_csv),
                    "metadata_json": str(metadata_json),
                    "n_unique": len(bucket_rows),
                    "n_replay": int(sum(counts)),
                }
            )

    g0_rng = np.random.default_rng(int(args.seed) + 9999)
    g0_indices = sorted(int(index) for index in g0_rng.choice(len(master_rows), size=int(args.fixed_budget_count), replace=False).tolist())
    g0_rows: list[dict[str, Any]] = []
    for local_rank, master_index in enumerate(g0_indices, start=1):
        row = master_rows[master_index]
        g0_rows.append(
            {
                "setting_id": "G0",
                "setting_name": "g0_uniform_random_pool250",
                "signal": "baseline",
                "bucket": "G0",
                "bucket_rank": local_rank,
                "image_id": row["image_id"],
                "img_rel_path": row["img_rel_path"],
                "pool_rank": row["pool_rank"],
                "calibrated_p": row["calibrated_p"],
                "R": row["R"],
                "C": row["C"],
                "D": row["D"],
                "S": row["pool_score_anchor"],
                "pi": round(1.0 / len(master_rows), 8),
                "duplication_count": 1,
                "selected_flag": 1,
            }
        )
    g0_candidate_scores_csv = experiments_dir / "G0_candidate_scores.csv"
    g0_metadata_json = metadata_dir / "G0_bucket_metadata.json"
    write_csv(g0_candidate_scores_csv, score_fieldnames, g0_rows)
    write_json(
        g0_metadata_json,
        {
            "experiment_id": "G0",
            "signal": "baseline",
            "bucket": "G0",
            "bucket_rank_start": 1,
            "bucket_rank_end": int(args.fixed_budget_count),
            "n_unique": int(args.fixed_budget_count),
            "n_replay": int(args.fixed_budget_count),
            "selection_mode": "uniform_random_without_replacement_from_pool250",
            "score_source": str(output_dir / "candidate_signal_master.csv"),
            "selected_image_ids": [row["image_id"] for row in g0_rows],
            "selected_rel_paths": [row["img_rel_path"] for row in g0_rows],
        },
    )
    experiment_rows.append(
        {
            "experiment_id": "G0",
            "signal": "baseline",
            "bucket": "G0",
            "bucket_rank_start": 1,
            "bucket_rank_end": int(args.fixed_budget_count),
            "setting_name": "g0_uniform_random_pool250",
            "candidate_scores_csv": str(g0_candidate_scores_csv),
            "metadata_json": str(g0_metadata_json),
            "n_unique": int(args.fixed_budget_count),
            "n_replay": int(args.fixed_budget_count),
        }
    )

    write_csv(output_dir / "bucket_experiment_registry.csv", experiment_fieldnames, experiment_rows)
    write_json(output_dir / "signal_stats.json", signal_stats)
    (output_dir / "signal_stats.md").write_text(
        "\n".join(
            [
                "# Bucket Pilot Signal Statistics",
                "",
                f"- candidate_top_k: `{int(args.candidate_top_k)}`",
                f"- bucket_count: `{int(args.bucket_count)}`",
                f"- bucket_size: `{bucket_size}`",
                f"- fixed_budget_count: `{int(args.fixed_budget_count)}`",
                f"- tau_r995: `{tau_995:.6f}`",
                f"- tau_r990: `{tau_990:.6f}`",
                f"- temperature_T: `{temperature:.6f}`",
                f"- s_R: `{s_r:.6f}`",
                f"- s_C: `{s_c:.6f}`",
                "",
                "| Signal | Min | Max | Top-5 IDs | Bottom-5 IDs |",
                "| --- | ---: | ---: | --- | --- |",
            ]
            + [
                "| {signal} | {value_min:.6f} | {value_max:.6f} | {top} | {bottom} |".format(
                    signal=signal_name,
                    value_min=signal_stats["signals"][signal_name]["value_min"],
                    value_max=signal_stats["signals"][signal_name]["value_max"],
                    top="; ".join(signal_stats["signals"][signal_name]["top5_image_ids"]),
                    bottom="; ".join(signal_stats["signals"][signal_name]["bottom5_image_ids"]),
                )
                for signal_name in ("R", "C", "D")
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print_step("done", f"wrote bucket pilot inputs to {output_dir}")


if __name__ == "__main__":
    main()
