from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normal/abnormal prototype banks for stage-1 PTSG.")
    parser.add_argument("--train-features-csv", required=True, help="Train feature index CSV exported by stage1_export_gate_features.py.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Train embedding matrix exported by stage1_export_gate_features.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for prototype bank files.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--hn-manifest", default="", help="Optional HN manifest CSV for HN-aware normal prototype.")
    parser.add_argument("--hn-weight", type=float, default=3.0, help="Extra weight for HN samples in the HN-aware bank.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_hn_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys: set[str] = set()
    for row in rows:
        rel_path = str(row.get("img_rel_path", "")).replace("\\", "/").strip()
        if rel_path:
            keys.add(rel_path)
    return keys


def mean_prototype(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        raise SystemExit("Cannot build prototype from empty embedding set.")
    return embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)


def weighted_normal_prototype(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    normal_class: str,
    hn_keys: set[str],
    hn_weight: float,
) -> np.ndarray:
    total = np.zeros((embeddings.shape[1],), dtype=np.float64)
    total_weight = 0.0
    for row in rows:
        if row["gt_label"] != normal_class:
            continue
        index = int(row["embedding_index"])
        vector = embeddings[index].astype(np.float64)
        weight = hn_weight if str(row["img_rel_path"]).replace("\\", "/") in hn_keys else 1.0
        total += vector * weight
        total_weight += weight
    if total_weight <= 0:
        raise SystemExit("No normal embeddings available for HN-aware prototype.")
    return (total / total_weight).astype(np.float32)


def main() -> None:
    args = parse_args()
    train_rows = load_feature_rows(Path(args.train_features_csv).resolve())
    train_embeddings = np.load(Path(args.train_embeddings_npy).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_indices = [int(row["embedding_index"]) for row in train_rows if row["gt_label"] == args.normal_class]
    abnormal_indices = [int(row["embedding_index"]) for row in train_rows if row["gt_label"] != args.normal_class]

    normal_proto = mean_prototype(train_embeddings[normal_indices])
    abnormal_proto = mean_prototype(train_embeddings[abnormal_indices])
    np.save(output_dir / "normal_proto.npy", normal_proto)
    np.save(output_dir / "abnormal_proto.npy", abnormal_proto)

    hn_summary: dict[str, Any] = {"hn_manifest": "", "hn_weight": args.hn_weight, "hn_match_count": 0}
    if args.hn_manifest:
        hn_path = Path(args.hn_manifest).resolve()
        hn_keys = load_hn_keys(hn_path)
        if hn_keys:
            normal_proto_hn = weighted_normal_prototype(train_rows, train_embeddings, args.normal_class, hn_keys, args.hn_weight)
            np.save(output_dir / "normal_proto_hn_aware.npy", normal_proto_hn)
            hn_summary = {
                "hn_manifest": str(hn_path),
                "hn_weight": args.hn_weight,
                "hn_match_count": sum(1 for row in train_rows if str(row["img_rel_path"]).replace("\\", "/") in hn_keys),
            }

    summary = {
        "train_features_csv": str(Path(args.train_features_csv).resolve()),
        "train_embeddings_npy": str(Path(args.train_embeddings_npy).resolve()),
        "normal_class": args.normal_class,
        "normal_count": len(normal_indices),
        "abnormal_count": len(abnormal_indices),
        "embedding_dim": int(train_embeddings.shape[1]) if train_embeddings.ndim == 2 else 0,
        **hn_summary,
    }
    (output_dir / "prototype_bank_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
