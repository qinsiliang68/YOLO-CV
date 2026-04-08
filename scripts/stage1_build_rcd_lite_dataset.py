from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-budget RCD-Lite stage-1 gate dataset.")
    parser.add_argument("--source-dataset", required=True, help="Original binary gate dataset root.")
    parser.add_argument("--scores-csv", required=True, help="CSV exported by stage1_score_train_normals_rcd_lite.py.")
    parser.add_argument("--train-features-csv", required=True, help="Train feature CSV from stage1_export_gate_features.py.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Train embedding matrix from stage1_export_gate_features.py.")
    parser.add_argument("--anchor-best-manifest", required=True, help="Anchor best_epoch_manifest.json.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root.")
    parser.add_argument("--candidate-top-k", type=int, default=250, help="Candidate risky-normal pool size.")
    parser.add_argument("--fixed-budget-count", type=int, required=True, help="Fixed extra-normal duplication budget.")
    parser.add_argument("--r-sigma", type=float, default=0.03, help="Sigmoid width for threshold-centered relevance.")
    parser.add_argument("--r-beta", type=float, default=0.75, help="Logit-distance decay for threshold-centered relevance.")
    parser.add_argument("--c-sigma", type=float, default=0.85, help="Variance decay for TTA consistency.")
    parser.add_argument("--rknn-k", type=int, default=10, help="k for reverse-kNN structural support.")
    parser.add_argument("--kappa", type=float, default=2.0, help="Nonlinear amplification for duplication probabilities.")
    parser.add_argument("--gallery-top-n", type=int, default=0, help="Optional number of selected normals to copy into a gallery. Default 0 keeps only CSV/JSON paths.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="File materialization mode.")
    parser.add_argument("--seed", type=int, default=20260330, help="Random seed for fixed-budget allocation.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normalize_rel_path(text: str) -> str:
    return str(text).replace("\\", "/")


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


def mirror_dataset(source_root: Path, target_root: Path, mode: str) -> int:
    count = 0
    for split_name in ("train", "val"):
        split_root = source_root / split_name
        if not split_root.exists():
            continue
        for image_path in split_root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target_path = target_root / image_path.relative_to(source_root)
            materialize(image_path, target_path, mode)
            count += 1
    return count


def reverse_knn_support(embeddings: np.ndarray, k: int) -> np.ndarray:
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
    k_eff = max(1, min(k, count - 1))
    knn_indices = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    support = np.zeros((count,), dtype=np.float32)
    for row in knn_indices:
        support[row] += 1.0
    max_support = float(np.max(support))
    if max_support <= 0.0:
        return np.ones((count,), dtype=np.float32)
    return support / max_support


def copy_gallery(rows: list[dict[str, Any]], source_root: Path, gallery_dir: Path, limit: int) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows[:limit], start=1):
        source = source_root / Path(str(row["img_rel_path"]))
        if not source.exists():
            continue
        score = float(row["S_score"])
        dup = int(row["duplication_count"])
        target = gallery_dir / f"{index:03d}_dup{dup:03d}_{score:.4f}_{source.name}"
        shutil.copy2(source, target)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores_rows = load_csv_rows(Path(args.scores_csv).resolve())
    feature_rows = load_csv_rows(Path(args.train_features_csv).resolve())
    embeddings = np.load(Path(args.train_embeddings_npy).resolve())
    anchor_best = json.loads(Path(args.anchor_best_manifest).resolve().read_text(encoding="utf-8"))

    feature_lookup: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        if str(row.get("split", "")) != "train":
            continue
        if str(row.get("gt_label", "")) != "Normal":
            continue
        feature_lookup[normalize_rel_path(str(row.get("img_rel_path", "")))] = row

    candidate_rows = sorted(
        scores_rows,
        key=lambda item: float(item["p_abnormal_cal_mean"]),
        reverse=True,
    )[: args.candidate_top_k]

    tau_r995 = float(anchor_best["tau_r995"])
    tau_logit = safe_prob_to_logit(tau_r995)
    joined_rows: list[dict[str, Any]] = []
    candidate_embeddings: list[np.ndarray] = []
    missing_features = 0
    for rank_index, row in enumerate(candidate_rows, start=1):
        rel_path = normalize_rel_path(str(row["img_rel_path"]))
        feature_row = feature_lookup.get(rel_path)
        if feature_row is None:
            missing_features += 1
            continue
        embedding_index = int(feature_row["embedding_index"])
        embedding = embeddings[embedding_index].astype(np.float32, copy=False)
        p_abnormal = float(row["p_abnormal_cal_mean"])
        logit_mean = safe_prob_to_logit(p_abnormal)
        logit_var = float(row["logit_abnormal_cal_var"])
        r_score = sigmoid((p_abnormal - tau_r995) / max(float(args.r_sigma), 1e-6)) * math.exp(
            -abs(logit_mean - tau_logit) / max(float(args.r_beta), 1e-6)
        )
        c_score = math.exp(-logit_var / max(float(args.c_sigma) ** 2, 1e-6))
        joined_rows.append(
            {
                "rank_in_candidate_pool": rank_index,
                "img_rel_path": rel_path,
                "img_path": str(row["img_path"]),
                "heuristic_group": str(row.get("heuristic_group", "")),
                "heuristic_reason": str(row.get("heuristic_reason", "")),
                "embedding_index": embedding_index,
                "p_abnormal_cal_mean": round(p_abnormal, 6),
                "p_abnormal_cal_max": round(float(row["p_abnormal_cal_max"]), 6),
                "p_abnormal_cal_min": round(float(row["p_abnormal_cal_min"]), 6),
                "logit_abnormal_cal_mean": round(logit_mean, 6),
                "logit_abnormal_cal_var": round(logit_var, 6),
                "R_score": round(r_score, 6),
                "C_score": round(c_score, 6),
            }
        )
        candidate_embeddings.append(embedding)

    if not joined_rows:
        raise SystemExit("No RCD-Lite candidates could be joined with train embeddings.")

    support_scores = reverse_knn_support(np.stack(candidate_embeddings), int(args.rknn_k))
    raw_weights: list[float] = []
    for row, d_score in zip(joined_rows, support_scores, strict=True):
        row["D_score"] = round(float(d_score), 6)
        combined = max(float(row["R_score"]) * float(row["C_score"]) * float(d_score), 1e-9)
        score = combined ** (1.0 / 3.0)
        row["S_score"] = round(float(score), 6)
        raw_weight = (score + 1e-9) ** float(args.kappa)
        raw_weights.append(float(raw_weight))

    weight_sum = float(sum(raw_weights))
    probabilities = [weight / weight_sum for weight in raw_weights]
    rng = np.random.default_rng(int(args.seed))
    duplication_counts = rng.multinomial(int(args.fixed_budget_count), probabilities)

    selected_rows: list[dict[str, Any]] = []
    for row, prob, count in zip(joined_rows, probabilities, duplication_counts, strict=True):
        row["selection_prob"] = round(float(prob), 8)
        row["duplication_count"] = int(count)
        row["selected_flag"] = int(count > 0)
        if count > 0:
            selected_rows.append(row)

    selected_rows = sorted(
        selected_rows,
        key=lambda item: (int(item["duplication_count"]), float(item["S_score"]), float(item["p_abnormal_cal_mean"])),
        reverse=True,
    )
    summary = {
        "anchor_best_manifest": str(Path(args.anchor_best_manifest).resolve()),
        "source_dataset": str(Path(args.source_dataset).resolve()),
        "candidate_top_k": int(args.candidate_top_k),
        "fixed_budget_count": int(args.fixed_budget_count),
        "tau_r995": tau_r995,
        "tau_r990": float(anchor_best["tau_r990"]),
        "temperature_T": float(anchor_best["temperature_T"]),
        "selected_unique_count": len(selected_rows),
        "missing_feature_rows": missing_features,
        "r_sigma": float(args.r_sigma),
        "r_beta": float(args.r_beta),
        "c_sigma": float(args.c_sigma),
        "rknn_k": int(args.rknn_k),
        "kappa": float(args.kappa),
    }
    return joined_rows, summary


def duplicate_selected_rows(
    selected_rows: list[dict[str, Any]],
    source_root: Path,
    target_root: Path,
    mode: str,
) -> list[dict[str, Any]]:
    duplicated: list[dict[str, Any]] = []
    for row in selected_rows:
        count = int(row["duplication_count"])
        if count <= 0:
            continue
        relative = Path(str(row["img_rel_path"]))
        source_path = source_root / relative
        stem = relative.stem
        suffix = relative.suffix
        for replica_index in range(1, count + 1):
            target_path = target_root / relative.parent / f"{stem}_rcd{replica_index:03d}{suffix}"
            materialize(source_path, target_path, mode)
            duplicated.append(
                {
                    "img_rel_path": str(relative).replace("\\", "/"),
                    "source_path": str(source_path),
                    "duplicated_path": str(target_path),
                    "duplication_index": replica_index,
                    "duplication_count_total": count,
                    "selection_prob": row["selection_prob"],
                    "R_score": row["R_score"],
                    "C_score": row["C_score"],
                    "D_score": row["D_score"],
                    "S_score": row["S_score"],
                }
            )
    return duplicated


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    target_root = Path(args.output_dataset).resolve()
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    candidate_rows, summary = build_rows(args)
    selected_rows = [row for row in candidate_rows if int(row["selected_flag"]) > 0]
    mirrored_count = mirror_dataset(source_root, target_root, args.link_mode)
    duplicated_rows = duplicate_selected_rows(selected_rows, source_root, target_root, args.link_mode)
    gallery_top_n = max(int(args.gallery_top_n), 0)
    if gallery_top_n > 0:
        copy_gallery(selected_rows, source_root, target_root / "rcd_selected_gallery", gallery_top_n)

    table_fields = [
        "rank_in_candidate_pool",
        "img_rel_path",
        "heuristic_group",
        "heuristic_reason",
        "embedding_index",
        "p_abnormal_cal_mean",
        "p_abnormal_cal_max",
        "p_abnormal_cal_min",
        "logit_abnormal_cal_mean",
        "logit_abnormal_cal_var",
        "R_score",
        "C_score",
        "D_score",
        "S_score",
        "selection_prob",
        "duplication_count",
        "selected_flag",
    ]
    write_csv(target_root / "rcd_score_table.csv", table_fields, candidate_rows)
    write_csv(target_root / "rcd_selected_normals.csv", table_fields, selected_rows)
    write_csv(
        target_root / "rcd_duplications.csv",
        [
            "img_rel_path",
            "source_path",
            "duplicated_path",
            "duplication_index",
            "duplication_count_total",
            "selection_prob",
            "R_score",
            "C_score",
            "D_score",
            "S_score",
        ],
        duplicated_rows,
    )

    summary.update(
        {
            "mirrored_images": mirrored_count,
            "duplicated_images": len(duplicated_rows),
            "selected_gallery_size": min(len(selected_rows), gallery_top_n),
            "output_dataset": str(target_root),
            "link_mode": args.link_mode,
        }
    )
    (target_root / "rcd_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_step(
        "done",
        f"wrote {target_root} with budget={summary['fixed_budget_count']} selected_unique={summary['selected_unique_count']}",
    )


if __name__ == "__main__":
    main()
