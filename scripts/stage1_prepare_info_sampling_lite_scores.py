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
    parser = argparse.ArgumentParser(description="Prepare static gate-aligned lite scores for the fixed HN14 candidate pool.")
    parser.add_argument("--teacher-weights", required=True, help="Teacher checkpoint used for one-shot scoring.")
    parser.add_argument("--teacher-best-manifest", required=True, help="Teacher best_epoch_manifest.json.")
    parser.add_argument("--source-dataset", required=True, help="Binary gate dataset root.")
    parser.add_argument("--pool-top-csv", required=True, help="Existing top250 hard-normal pool manifest.")
    parser.add_argument("--pool-scores-csv", required=True, help="Existing full train-normal score ranking for uniform HN14 reference.")
    parser.add_argument("--train-features-csv", required=True, help="Train feature CSV from stage1_export_gate_features.py.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Train embedding matrix from stage1_export_gate_features.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived score files.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--candidate-top-k", type=int, default=250, help="Candidate pool size.")
    parser.add_argument("--fixed-budget-count", type=int, required=True, help="Fixed extra-normal budget aligned with HN14.")
    parser.add_argument("--alpha", type=float, default=2.0, help="Exponent for the threshold-centered risk term.")
    parser.add_argument("--density-k", type=int, default=15, help="k for cosine-neighbor density support.")
    parser.add_argument("--kappa", type=float, default=2.0, help="Nonlinear amplification used for replay probability.")
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


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normalize_rel_path(text: str) -> str:
    return str(text).replace("\\", "/")


def path_stub(rel_path: str) -> str:
    return Path(rel_path).as_posix()


def image_id_from_rel_path(rel_path: str) -> str:
    path = Path(rel_path)
    parent = path.parent.name
    return f"{parent}_{path.stem}"


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
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k_eff = max(1, min(int(k), count - 1))
    top_indices = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_sim = np.zeros((count,), dtype=np.float32)
    for idx in range(count):
        values = similarity[idx, top_indices[idx]]
        mean_sim[idx] = float(np.mean(values))
    finite = mean_sim[np.isfinite(mean_sim)]
    if finite.size == 0:
        return np.ones((count,), dtype=np.float32)
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    if math.isclose(min_val, max_val, rel_tol=0.0, abs_tol=1e-9):
        return np.ones((count,), dtype=np.float32)
    normalized_support = (mean_sim - min_val) / (max_val - min_val)
    return np.clip(normalized_support, 0.0, 1.0)


def compute_gini(probabilities: list[float]) -> float:
    values = np.asarray(sorted(float(value) for value in probabilities), dtype=np.float64)
    if values.size == 0:
        return 0.0
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    index = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * values) / (values.size * total)) - ((values.size + 1.0) / values.size))


def effective_count(probabilities: list[float]) -> float:
    values = np.asarray([max(float(value), 1e-12) for value in probabilities], dtype=np.float64)
    entropy = -float(np.sum(values * np.log(values)))
    return float(math.exp(entropy))


def finite_flags(values: list[float]) -> tuple[bool, bool]:
    has_nan = any(math.isnan(float(value)) for value in values)
    has_inf = any(math.isinf(float(value)) for value in values)
    return has_nan, has_inf


def rank_pool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["pool_score_anchor"]), reverse=True)


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
    teacher_checkpoint = Path(args.teacher_weights).resolve()

    pool_top_rows = load_csv_rows(Path(args.pool_top_csv).resolve())
    pool_scores_rows = load_csv_rows(Path(args.pool_scores_csv).resolve())
    feature_rows = load_csv_rows(Path(args.train_features_csv).resolve())
    embeddings = np.load(Path(args.train_embeddings_npy).resolve())

    candidate_rows = rank_pool_rows(
        [
            {
                "img_path": str(row["img_path"]),
                "img_rel_path": normalize_rel_path(str(row["img_rel_path"])),
                "pool_score_anchor": float(row["p_abnormal"]),
                "pool_rank": index,
            }
            for index, row in enumerate(pool_top_rows[: args.candidate_top_k], start=1)
        ]
    )
    candidate_lookup = {row["img_rel_path"]: row for row in candidate_rows}
    uniform_rows = sorted(pool_scores_rows, key=lambda row: float(row["p_abnormal"]), reverse=True)
    uniform_anchor_rows = uniform_rows[: int(args.fixed_budget_count)]

    feature_lookup: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        if str(row.get("split", "")) != "train":
            continue
        if str(row.get("gt_label", "")) != args.normal_class:
            continue
        feature_lookup[normalize_rel_path(str(row.get("img_rel_path", "")))] = row

    model = YOLO(str(teacher_checkpoint), task="classify")
    master_rows: list[dict[str, Any]] = []
    candidate_embeddings: list[np.ndarray] = []
    candidate_paths: list[str] = []
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
        candidate_paths.append(rel_path)

        with Image.open(image_path) as image:
            variants = build_tta_variants(image)
        variant_probs = predict_variant_probs(model, variants, args.imgsz, args.batch, str(args.device), args.normal_class)
        raw_probs = [prob for _name, prob in variant_probs]
        raw_logits = [safe_prob_to_logit(prob) for prob in raw_probs]
        cal_probs = [sigmoid(logit / temperature) for logit in raw_logits]
        cal_logits = [safe_prob_to_logit(prob) for prob in cal_probs]
        p_mean = float(np.mean(cal_probs))
        logit_var = float(np.var(np.asarray(cal_logits, dtype=np.float32)))
        r_raw = max(0.0, (p_mean - tau_995) / max(1.0 - tau_995, 1e-6))
        r_score = r_raw ** float(args.alpha)
        master_rows.append(
            {
                "image_id": image_id_from_rel_path(rel_path),
                "img_rel_path": rel_path,
                "path_stub": path_stub(rel_path),
                "pool_rank": int(pool_row["pool_rank"]),
                "pool_score_anchor": round(float(pool_row["pool_score_anchor"]), 6),
                "embedding_index": embedding_index,
                "calibrated_p": round(p_mean, 6),
                "raw_tta_variance": round(logit_var, 6),
                "R": round(r_score, 6),
            }
        )

    support = density_support(np.stack(candidate_embeddings), int(args.density_k))
    variances = [float(row["raw_tta_variance"]) for row in master_rows]
    sigma_c = float(np.median(np.asarray(variances, dtype=np.float32))) if variances else 1.0
    sigma_c = max(sigma_c, 1e-6)
    for row, d_score in zip(master_rows, support, strict=True):
        c_score = math.exp(-float(row["raw_tta_variance"]) / (sigma_c ** 2))
        row["C"] = round(float(c_score), 6)
        row["D"] = round(float(d_score), 6)

    variant_defs = [
        ("A2", "weighted_hn14_risk_only", "risk_only"),
        ("A3", "weighted_hn14_risk_consistency", "risk_consistency"),
        ("A4", "weighted_hn14_risk_consistency_density", "risk_consistency_density"),
    ]
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    stats_rows: list[dict[str, Any]] = []
    variant_details: dict[str, Any] = {}

    for variant_index, (setting_id, setting_name, score_variant) in enumerate(variant_defs, start=1):
        rows_for_variant: list[dict[str, Any]] = []
        raw_weights: list[float] = []
        for row in master_rows:
            r_val = float(row["R"])
            c_val = float(row["C"])
            d_val = float(row["D"])
            if score_variant == "risk_only":
                s_val = max(r_val, 1e-9)
            elif score_variant == "risk_consistency":
                s_val = max(r_val * c_val, 1e-9) ** 0.5
            else:
                s_val = max(r_val * c_val * d_val, 1e-9) ** (1.0 / 3.0)
            raw_weight = (s_val + 1e-8) ** float(args.kappa)
            raw_weights.append(float(raw_weight))
            row_copy = dict(row)
            row_copy["setting_id"] = setting_id
            row_copy["setting_name"] = setting_name
            row_copy["score_variant"] = score_variant
            row_copy["S"] = round(float(s_val), 6)
            rows_for_variant.append(row_copy)

        weight_sum = float(sum(raw_weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise SystemExit(f"{setting_id}: invalid raw weight sum {weight_sum}")
        probabilities = [weight / weight_sum for weight in raw_weights]
        score_values = [float(row["S"]) for row in rows_for_variant]
        has_score_nan, has_score_inf = finite_flags(score_values)
        has_prob_nan, has_prob_inf = finite_flags(probabilities)
        if has_score_nan or has_score_inf or has_prob_nan or has_prob_inf:
            raise SystemExit(
                f"{setting_id}: non-finite values detected "
                f"(score_nan={has_score_nan}, score_inf={has_score_inf}, "
                f"prob_nan={has_prob_nan}, prob_inf={has_prob_inf})"
            )
        rng = np.random.default_rng(int(args.seed) + variant_index)
        duplication_counts = rng.multinomial(int(args.fixed_budget_count), probabilities)
        for row_copy, prob, duplication in zip(rows_for_variant, probabilities, duplication_counts, strict=True):
            row_copy["pi"] = round(float(prob), 8)
            row_copy["duplication_count"] = int(duplication)
            row_copy["selected_flag"] = int(duplication > 0)
        rows_for_variant.sort(key=lambda item: (int(item["duplication_count"]), float(item["S"]), float(item["calibrated_p"])), reverse=True)
        variant_rows[setting_id] = rows_for_variant

        top_ids = [row["image_id"] for row in sorted(rows_for_variant, key=lambda item: float(item["S"]), reverse=True)[:10]]
        bottom_ids = [row["image_id"] for row in sorted(rows_for_variant, key=lambda item: float(item["S"]))[:10]]
        top10_cumulative_pi = float(sum(float(row["pi"]) for row in sorted(rows_for_variant, key=lambda item: float(item["S"]), reverse=True)[:10]))
        probability_sum = float(sum(probabilities))
        nonzero_probability_count = int(sum(1 for value in probabilities if float(value) > 0.0))
        stats_rows.append(
            {
                "setting": setting_name,
                "score_min": round(min(float(row["S"]) for row in rows_for_variant), 6),
                "score_median": round(float(np.median([float(row["S"]) for row in rows_for_variant])), 6),
                "score_max": round(max(float(row["S"]) for row in rows_for_variant), 6),
                "prob_min": round(min(float(row["pi"]) for row in rows_for_variant), 8),
                "prob_median": round(float(np.median([float(row["pi"]) for row in rows_for_variant])), 8),
                "prob_max": round(max(float(row["pi"]) for row in rows_for_variant), 8),
                "effective_count": round(effective_count([float(row["pi"]) for row in rows_for_variant]), 4),
                "gini": round(compute_gini([float(row["pi"]) for row in rows_for_variant]), 6),
                "prob_sum": round(probability_sum, 10),
                "top10_cumulative_pi": round(top10_cumulative_pi, 8),
                "nonzero_probability_count": nonzero_probability_count,
                "top10_ids": ";".join(top_ids),
                "bottom10_ids": ";".join(bottom_ids),
            }
        )
        variant_details[setting_id] = {
            "setting_name": setting_name,
            "score_variant": score_variant,
            "selected_unique_count": int(sum(1 for row in rows_for_variant if int(row["selected_flag"]) > 0)),
            "duplicate_total": int(sum(int(row["duplication_count"]) for row in rows_for_variant)),
            "probability_sum": probability_sum,
            "top10_cumulative_pi": top10_cumulative_pi,
            "nonzero_probability_count": nonzero_probability_count,
            "score_has_nan": has_score_nan,
            "score_has_inf": has_score_inf,
            "probability_has_nan": has_prob_nan,
            "probability_has_inf": has_prob_inf,
            "top10_ids": top_ids,
            "bottom10_ids": bottom_ids,
        }

    master_fieldnames = [
        "image_id",
        "img_rel_path",
        "path_stub",
        "pool_rank",
        "pool_score_anchor",
        "embedding_index",
        "calibrated_p",
        "raw_tta_variance",
        "R",
        "C",
        "D",
    ]
    write_csv(output_dir / "candidate_pool_master.csv", master_fieldnames, master_rows)

    variant_fieldnames = [
        "setting_id",
        "setting_name",
        "score_variant",
        "image_id",
        "img_rel_path",
        "path_stub",
        "pool_rank",
        "pool_score_anchor",
        "embedding_index",
        "calibrated_p",
        "raw_tta_variance",
        "R",
        "C",
        "D",
        "S",
        "pi",
        "duplication_count",
        "selected_flag",
    ]
    for setting_id, rows in variant_rows.items():
        write_csv(output_dir / f"{setting_id}_candidate_pool_scores.csv", variant_fieldnames, rows)
        stats_payload = {
            "setting_id": setting_id,
            "details": variant_details[setting_id],
            "fixed_budget_count": int(args.fixed_budget_count),
            "candidate_top_k": int(args.candidate_top_k),
            "teacher_tau_r995": tau_995,
            "teacher_tau_r990": tau_990,
            "teacher_temperature": temperature,
            "alpha": float(args.alpha),
            "kappa": float(args.kappa),
            "density_k": int(args.density_k),
            "sigma_c": sigma_c,
        }
        write_json(output_dir / f"{setting_id}_score_stats.json", stats_payload)
        (output_dir / f"{setting_id}_score_stats.md").write_text(
            "\n".join(
                [
                    f"# {variant_details[setting_id]['setting_name']}",
                    "",
                    f"- candidate_top_k: `{args.candidate_top_k}`",
                    f"- fixed_budget_count: `{args.fixed_budget_count}`",
                    f"- selected_unique_count: `{variant_details[setting_id]['selected_unique_count']}`",
                    f"- duplicate_total: `{variant_details[setting_id]['duplicate_total']}`",
                    f"- probability_sum: `{variant_details[setting_id]['probability_sum']:.10f}`",
                    f"- top10_cumulative_pi: `{variant_details[setting_id]['top10_cumulative_pi']:.8f}`",
                    f"- nonzero_probability_count: `{variant_details[setting_id]['nonzero_probability_count']}`",
                    f"- sigma_c: `{sigma_c:.6f}`",
                    f"- top10_ids: `{'; '.join(variant_details[setting_id]['top10_ids'])}`",
                    f"- bottom10_ids: `{'; '.join(variant_details[setting_id]['bottom10_ids'])}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    write_csv(
        output_dir / "uniform_hn14_reference.csv",
        ["uniform_rank", "img_rel_path", "image_id", "path_stub", "p_abnormal"],
        [
            {
                "uniform_rank": index,
                "img_rel_path": normalize_rel_path(str(row["img_rel_path"])),
                "image_id": image_id_from_rel_path(normalize_rel_path(str(row["img_rel_path"]))),
                "path_stub": path_stub(normalize_rel_path(str(row["img_rel_path"]))),
                "p_abnormal": round(float(row["p_abnormal"]), 6),
            }
            for index, row in enumerate(uniform_anchor_rows, start=1)
        ],
    )
    write_csv(
        output_dir / "table_score_component_stats.csv",
        [
            "setting",
            "score_min",
            "score_median",
            "score_max",
            "prob_min",
            "prob_median",
            "prob_max",
            "effective_count",
            "gini",
            "prob_sum",
            "top10_cumulative_pi",
            "nonzero_probability_count",
            "top10_ids",
            "bottom10_ids",
        ],
        stats_rows,
    )
    write_json(
        output_dir / "table_score_component_stats.json",
        {
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_tau_r995": tau_995,
            "teacher_tau_r990": tau_990,
            "teacher_temperature": temperature,
            "sigma_c": sigma_c,
            "rows": stats_rows,
        },
    )
    (output_dir / "table_score_component_stats.md").write_text(
        "\n".join(
            [
                "# Lite Score Component Statistics",
                "",
                "| Setting | Score Min | Score Median | Score Max | Prob Min | Prob Median | Prob Max | Effective Count | Gini | Prob Sum | Top10 Cumulative Pi | Nonzero Prob Count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            + [
                "| {setting} | {score_min:.6f} | {score_median:.6f} | {score_max:.6f} | {prob_min:.8f} | {prob_median:.8f} | {prob_max:.8f} | {effective_count:.4f} | {gini:.6f} | {prob_sum:.10f} | {top10_cumulative_pi:.8f} | {nonzero_probability_count} |".format(
                    **row
                )
                for row in stats_rows
            ]
            + [
                "",
                f"- sigma_c (pool median variance): `{sigma_c:.6f}`",
                f"- teacher tau_R99.5: `{tau_995:.6f}`",
                f"- teacher tau_R99.0: `{tau_990:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "pool_source_manifest.json",
        {
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_best_manifest": str(Path(args.teacher_best_manifest).resolve()),
            "source_dataset": str(source_dataset),
            "pool_top_csv": str(Path(args.pool_top_csv).resolve()),
            "pool_scores_csv": str(Path(args.pool_scores_csv).resolve()),
            "candidate_top_k": int(args.candidate_top_k),
            "fixed_budget_count": int(args.fixed_budget_count),
            "teacher_tau_r995": tau_995,
            "teacher_tau_r990": tau_990,
            "teacher_temperature": temperature,
            "sigma_c": sigma_c,
        },
    )
    print_step("done", f"wrote lite candidate scores to {output_dir}")


if __name__ == "__main__":
    main()
