from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RCIS stage-1 dataset with information-driven sampling weights.")
    parser.add_argument("--source-dataset", required=True, help="Original binary gate dataset root.")
    parser.add_argument("--scores-csv", required=True, help="Train-side score CSV exported by stage1_score_train_samples.py.")
    parser.add_argument("--train-features-csv", required=True, help="Train feature CSV exported by stage1_export_gate_features.py.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Train embedding matrix exported by stage1_export_gate_features.py.")
    parser.add_argument("--reference-material-dir", required=True, help="Existing evaluated material dir for thresholds and temperature.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--reference-variant", default="auto", help="Threshold source variant, e.g. P0/P2 or auto.")
    parser.add_argument("--alpha-boundary", type=float, default=1.0, help="Boundary-score coefficient.")
    parser.add_argument("--beta-uncertainty", type=float, default=0.2, help="Entropy / uncertainty coefficient.")
    parser.add_argument("--gamma-flip", type=float, default=0.0, help="P0/P2 disagreement coefficient.")
    parser.add_argument("--delta-rarity", type=float, default=0.4, help="Rarity coefficient.")
    parser.add_argument("--eta-hardness", type=float, default=0.9, help="Hard-type boost coefficient.")
    parser.add_argument("--mu-redundancy", type=float, default=0.8, help="Redundancy penalty coefficient.")
    parser.add_argument("--nu-quality", type=float, default=0.0, help="Quality penalty coefficient.")
    parser.add_argument("--boundary-band", type=float, default=0.05, help="Operating-point boundary bandwidth.")
    parser.add_argument("--gap-scale", type=float, default=0.08, help="Scale for P0/P2 disagreement normalization.")
    parser.add_argument("--normal-clusters", type=int, default=24, help="Normal-cluster count for rarity / redundancy.")
    parser.add_argument("--abnormal-clusters", type=int, default=12, help="Abnormal-cluster count for rarity / redundancy.")
    parser.add_argument("--cluster-iter", type=int, default=30, help="KMeans iteration cap.")
    parser.add_argument("--sigmoid-gain", type=float, default=4.0, help="Gain for weight mapping.")
    parser.add_argument("--normal-wmin", type=float, default=0.25, help="Normal lower weight bound.")
    parser.add_argument("--normal-wmax", type=float, default=3.0, help="Normal upper weight bound.")
    parser.add_argument("--abnormal-wmin", type=float, default=1.0, help="Abnormal lower weight bound.")
    parser.add_argument("--abnormal-wmax", type=float, default=4.0, help="Abnormal upper weight bound.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="Output materialization mode.")
    parser.add_argument("--seed", type=int, default=20260330, help="Deterministic seed for clustering and fractional copy rounding.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def binary_entropy(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    entropy = -(clipped * math.log(clipped) + (1.0 - clipped) * math.log(1.0 - clipped))
    return entropy / math.log(2.0)


def stable_unit_float(key: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def copies_from_weight(weight: float, key: str, seed: int) -> int:
    if weight <= 0.0:
        return 0
    draw = stable_unit_float(key, seed)
    if weight < 1.0:
        return 1 if draw < weight else 0
    integer = int(math.floor(weight))
    fraction = weight - integer
    return integer + (1 if draw < fraction else 0)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def fit_class_clusters(
    embeddings: np.ndarray,
    indices: list[int],
    cluster_count: int,
    seed: int,
    iter_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset = embeddings[indices]
    if subset.shape[0] == 0:
        raise SystemExit("Cannot cluster an empty class subset for RCIS.")
    k = max(1, min(cluster_count, subset.shape[0]))
    if k == 1:
        labels = np.zeros((subset.shape[0],), dtype=np.int32)
        centroids = subset[:1].copy()
        return labels, centroids
    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(subset, k=k, minit="points", iter=iter_count, seed=rng)
    return labels.astype(np.int32), np.asarray(centroids, dtype=np.float32)


def compute_cluster_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=int(centroids.shape[0])).astype(np.float32)
    min_count = float(counts.min())
    max_count = float(counts.max())
    normalized = normalize_embeddings(embeddings)
    centroid_norm = normalize_embeddings(centroids)
    centroid_cos = np.sum(normalized * centroid_norm[labels], axis=1)
    centroid_cos = np.clip((centroid_cos + 1.0) * 0.5, 0.0, 1.0)

    if max_count <= min_count:
        size_density = np.ones_like(counts[labels], dtype=np.float32)
        size_rarity = np.zeros_like(counts[labels], dtype=np.float32)
    else:
        size_density = (counts[labels] - min_count) / (max_count - min_count)
        size_rarity = 1.0 - size_density

    centroid_dist = 1.0 - centroid_cos
    rarity = np.clip(0.7 * size_rarity + 0.3 * centroid_dist, 0.0, 1.0)
    redundancy = np.clip(0.7 * size_density + 0.3 * centroid_cos, 0.0, 1.0)
    return rarity.astype(np.float32), redundancy.astype(np.float32)


def load_reference_config(material_dir: Path, variant: str) -> dict[str, Any]:
    best_path = material_dir / "best_ptsg_config.json"
    if not best_path.exists():
        raise SystemExit(f"Missing best_ptsg_config.json under {material_dir}")
    best_cfg = json.loads(best_path.read_text(encoding="utf-8"))
    chosen_variant = str(best_cfg.get("best_variant", "P0")) if variant == "auto" else variant.upper()
    threshold_path = material_dir / chosen_variant.lower() / "threshold_summary.json"
    if not threshold_path.exists():
        raise SystemExit(f"Missing threshold summary for variant {chosen_variant}: {threshold_path}")
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return {
        "variant": chosen_variant,
        "temperature": float(best_cfg.get("temperature", 1.0) or 1.0),
        "alpha": float(best_cfg.get("alpha", 1.0) or 1.0),
        "beta": float(best_cfg.get("beta", 1.0) or 1.0),
        "gamma": float(best_cfg.get("gamma", 0.5) or 0.5),
        "tau_r995": float(threshold_payload["operating_points"]["recall_ge_99_5"]["threshold"]),
        "tau_r990": float(threshold_payload["operating_points"]["recall_ge_99_0"]["threshold"]),
    }


def boundary_score(prob: float, tau_r995: float, tau_r990: float, band: float) -> float:
    band = max(float(band), 1e-6)
    near_995 = math.exp(-abs(prob - tau_r995) / band)
    near_990 = math.exp(-abs(prob - tau_r990) / band)
    return float(max(near_995, near_990))


def quality_penalty(image_path: Path) -> tuple[float, dict[str, float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return 1.0, {"blur_penalty": 1.0, "exposure_penalty": 1.0, "contrast_penalty": 1.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_value = float(gray.mean())
    std_value = float(gray.std())

    blur_penalty = min(max((60.0 - blur_value) / 60.0, 0.0), 1.0)
    dark_penalty = min(max((45.0 - mean_value) / 45.0, 0.0), 1.0)
    bright_penalty = min(max((mean_value - 215.0) / 40.0, 0.0), 1.0)
    exposure_penalty = max(dark_penalty, bright_penalty)
    contrast_penalty = min(max((35.0 - std_value) / 35.0, 0.0), 1.0)
    penalty = float((blur_penalty + exposure_penalty + contrast_penalty) / 3.0)
    return penalty, {
        "blur_penalty": round(blur_penalty, 6),
        "exposure_penalty": round(exposure_penalty, 6),
        "contrast_penalty": round(contrast_penalty, 6),
    }


def build_train_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        lookup[str(row["img_rel_path"]).replace("\\", "/")] = row
    return lookup


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    output_root = Path(args.output_dataset).resolve()
    reference_dir = Path(args.reference_material_dir).resolve()
    scores_rows = load_rows(Path(args.scores_csv).resolve())
    feature_rows = load_rows(Path(args.train_features_csv).resolve())
    embeddings = np.load(Path(args.train_embeddings_npy).resolve()).astype(np.float32)
    reference_cfg = load_reference_config(reference_dir, args.reference_variant)
    signal_flags = {
        "boundary_enabled": bool(args.alpha_boundary > 0),
        "uncertainty_enabled": bool(args.beta_uncertainty > 0),
        "flip_enabled": bool(args.gamma_flip > 0),
        "rarity_enabled": bool(args.delta_rarity > 0),
        "hardness_enabled": bool(args.eta_hardness > 0),
        "redundancy_enabled": bool(args.mu_redundancy > 0),
        "quality_enabled": bool(args.nu_quality > 0),
    }

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_feature_rows = [row for row in feature_rows if row.get("split") == "train"]
    feature_lookup = build_train_lookup(train_feature_rows)
    score_lookup = build_train_lookup(scores_rows)
    merged_rows: list[dict[str, Any]] = []
    for img_rel_path, score_row in score_lookup.items():
        feature_row = feature_lookup.get(img_rel_path)
        if feature_row is None:
            continue
        row = {**feature_row, **score_row}
        row["img_rel_path"] = img_rel_path
        row["embedding_index"] = int(feature_row["embedding_index"])
        merged_rows.append(row)

    if not merged_rows:
        raise SystemExit("No overlapping train rows between scores CSV and feature CSV.")

    embeddings_norm = normalize_embeddings(embeddings)
    normal_indices = [int(row["embedding_index"]) for row in merged_rows if row["gt_label"] == args.normal_class]
    abnormal_indices = [int(row["embedding_index"]) for row in merged_rows if row["gt_label"] != args.normal_class]
    if signal_flags["flip_enabled"]:
        normal_proto = embeddings[normal_indices].mean(axis=0)
        abnormal_proto = embeddings[abnormal_indices].mean(axis=0)
    else:
        normal_proto = None
        abnormal_proto = None

    normal_labels, normal_centroids = fit_class_clusters(
        embeddings_norm,
        normal_indices,
        args.normal_clusters,
        args.seed,
        args.cluster_iter,
    )
    abnormal_labels, abnormal_centroids = fit_class_clusters(
        embeddings_norm,
        abnormal_indices,
        args.abnormal_clusters,
        args.seed + 1,
        args.cluster_iter,
    )
    normal_rarity, normal_redundancy = compute_cluster_scores(embeddings_norm[normal_indices], normal_labels, normal_centroids)
    abnormal_rarity, abnormal_redundancy = compute_cluster_scores(embeddings_norm[abnormal_indices], abnormal_labels, abnormal_centroids)

    class_signal_lookup: dict[int, dict[str, float]] = {}
    for local_idx, embedding_index in enumerate(normal_indices):
        class_signal_lookup[int(embedding_index)] = {
            "rarity": float(normal_rarity[local_idx]),
            "redundancy": float(normal_redundancy[local_idx]),
        }
    for local_idx, embedding_index in enumerate(abnormal_indices):
        class_signal_lookup[int(embedding_index)] = {
            "rarity": float(abnormal_rarity[local_idx]),
            "redundancy": float(abnormal_redundancy[local_idx]),
        }

    sample_rows: list[dict[str, Any]] = []
    info_scores_by_class: dict[str, list[float]] = {"normal": [], "abnormal": []}
    for row in merged_rows:
        image_path = Path(row["img_path"])
        p_abnormal_raw = float(row["p_abnormal"])
        logit_raw = safe_prob_to_logit(p_abnormal_raw)
        p_abnormal_cal = sigmoid(logit_raw / reference_cfg["temperature"])

        embedding_index = int(row["embedding_index"])
        b_score = boundary_score(p_abnormal_cal, reference_cfg["tau_r995"], reference_cfg["tau_r990"], args.boundary_band)
        u_score = binary_entropy(p_abnormal_cal)
        if signal_flags["flip_enabled"]:
            embedding = embeddings[embedding_index]
            d_normal = float(np.linalg.norm(embedding - normal_proto))
            d_abnormal = float(np.linalg.norm(embedding - abnormal_proto))
            trust_normal = d_abnormal / (d_normal + d_abnormal + 1e-8)
            p2_safe = sigmoid(reference_cfg["alpha"] * (1.0 - p_abnormal_cal) + reference_cfg["beta"] * trust_normal)
            p2_abnormal = 1.0 - p2_safe
            disagree_995 = int((p_abnormal_cal >= reference_cfg["tau_r995"]) != (p2_abnormal >= reference_cfg["tau_r995"]))
            disagree_990 = int((p_abnormal_cal >= reference_cfg["tau_r990"]) != (p2_abnormal >= reference_cfg["tau_r990"]))
            flip_score = min(
                1.0,
                0.5 * max(disagree_995, disagree_990)
                + 0.5 * min(1.0, abs(p2_abnormal - p_abnormal_cal) / max(args.gap_scale, 1e-6)),
            )
            p2_abnormal_proxy = p2_abnormal
        else:
            flip_score = 0.0
            p2_abnormal_proxy = p_abnormal_cal

        class_key = "normal" if row["gt_label"] == args.normal_class else "abnormal"
        if class_key == "normal":
            h_score = p_abnormal_cal
        else:
            h_score = max(1.0 - p_abnormal_cal, b_score)

        class_signals = class_signal_lookup[embedding_index]
        r_score = float(class_signals["rarity"])
        d_score = float(class_signals["redundancy"])
        if signal_flags["quality_enabled"]:
            q_score, q_detail = quality_penalty(image_path)
        else:
            q_score = 0.0
            q_detail = {
                "blur_penalty": 0.0,
                "exposure_penalty": 0.0,
                "contrast_penalty": 0.0,
            }

        info_score = (
            args.alpha_boundary * b_score
            + args.beta_uncertainty * u_score
            + args.gamma_flip * flip_score
            + args.delta_rarity * r_score
            + args.eta_hardness * h_score
            - args.mu_redundancy * d_score
            - args.nu_quality * q_score
        )
        info_scores_by_class[class_key].append(info_score)
        sample_rows.append(
            {
                "img_rel_path": row["img_rel_path"],
                "img_path": str(image_path),
                "gt_label": row["gt_label"],
                "class_key": class_key,
                "p_abnormal_raw": round(p_abnormal_raw, 6),
                "p_abnormal_cal": round(p_abnormal_cal, 6),
                "p2_abnormal_proxy": round(p2_abnormal_proxy, 6),
                "boundary_score": round(b_score, 6),
                "uncertainty_score": round(u_score, 6),
                "flip_score": round(flip_score, 6),
                "rarity_score": round(r_score, 6),
                "hardness_score": round(h_score, 6),
                "redundancy_penalty": round(d_score, 6),
                "quality_penalty": round(q_score, 6),
                "blur_penalty": q_detail["blur_penalty"],
                "exposure_penalty": q_detail["exposure_penalty"],
                "contrast_penalty": q_detail["contrast_penalty"],
                "info_score": round(info_score, 6),
            }
        )

    class_medians = {
        key: float(np.median(values)) if values else 0.0
        for key, values in info_scores_by_class.items()
    }

    sampling_rows: list[dict[str, Any]] = []
    train_output_root = output_root / "train"
    val_output_root = output_root / "val"
    kept_unique = 0
    total_materialized = 0

    # Mirror validation split unchanged.
    for image_path in sorted((source_root / "val").rglob("*")) if (source_root / "val").exists() else []:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = val_output_root / image_path.relative_to(source_root / "val")
        materialize(image_path, target, args.link_mode)

    for row in sample_rows:
        class_key = str(row["class_key"])
        info_score = float(row["info_score"])
        median = class_medians[class_key]
        if class_key == "normal":
            weight = args.normal_wmin + (args.normal_wmax - args.normal_wmin) * sigmoid(args.sigmoid_gain * (info_score - median))
        else:
            weight = args.abnormal_wmin + (args.abnormal_wmax - args.abnormal_wmin) * sigmoid(args.sigmoid_gain * (info_score - median))

        rel_path = Path(row["img_rel_path"])
        source_image = Path(row["img_path"])
        copies = copies_from_weight(weight, row["img_rel_path"], args.seed)
        kept_unique += int(copies > 0)
        if copies > 0:
            first_target = train_output_root / rel_path.relative_to("train")
            materialize(source_image, first_target, args.link_mode)
            total_materialized += 1
            for replica_idx in range(2, copies + 1):
                extra_target = first_target.with_name(f"{first_target.stem}_rcis{replica_idx - 1}{first_target.suffix}")
                materialize(source_image, extra_target, args.link_mode)
                total_materialized += 1

        sampling_rows.append(
            {
                **row,
                "info_median_class": round(median, 6),
                "sampling_weight": round(weight, 6),
                "copies": copies,
            }
        )

    sampling_rows.sort(key=lambda item: (item["sampling_weight"], item["info_score"]), reverse=True)
    write_csv(output_root / "rcis_sampling_manifest.csv", sampling_rows)

    summary = {
        "source_dataset": str(source_root),
        "reference_material_dir": str(reference_dir),
        "reference_variant": reference_cfg["variant"],
        "temperature": reference_cfg["temperature"],
        "tau_r995": reference_cfg["tau_r995"],
        "tau_r990": reference_cfg["tau_r990"],
        "normal_class": args.normal_class,
        "normal_clusters": args.normal_clusters,
        "abnormal_clusters": args.abnormal_clusters,
        "seed": args.seed,
        "coefficients": {
            "alpha_boundary": args.alpha_boundary,
            "beta_uncertainty": args.beta_uncertainty,
            "gamma_flip": args.gamma_flip,
            "delta_rarity": args.delta_rarity,
            "eta_hardness": args.eta_hardness,
            "mu_redundancy": args.mu_redundancy,
            "nu_quality": args.nu_quality,
        },
        "signal_flags": signal_flags,
        "weight_bounds": {
            "normal": [args.normal_wmin, args.normal_wmax],
            "abnormal": [args.abnormal_wmin, args.abnormal_wmax],
        },
        "train_original_count": len(sample_rows),
        "train_kept_unique": kept_unique,
        "train_materialized_total": total_materialized,
        "normal_original_count": sum(1 for row in sample_rows if row["class_key"] == "normal"),
        "abnormal_original_count": sum(1 for row in sample_rows if row["class_key"] == "abnormal"),
        "normal_materialized_total": sum(int(row["copies"]) for row in sampling_rows if row["class_key"] == "normal"),
        "abnormal_materialized_total": sum(int(row["copies"]) for row in sampling_rows if row["class_key"] == "abnormal"),
    }
    (output_root / "rcis_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_step("done", f"wrote {output_root}")


if __name__ == "__main__":
    main()
