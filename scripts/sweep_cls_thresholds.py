from __future__ import annotations

import argparse
import math
from pathlib import Path

from pipeline_common import YOLOV11_ROOT, ensure_yolov11_importable, load_json_config, resolve_relative_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SOURCE_CONFIG = "YOLOv11/configs/runtime/cls_source_cls6.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep abnormal-confidence thresholds for a classification model and write metrics to a txt report."
    )
    parser.add_argument("--config", default=DEFAULT_SOURCE_CONFIG, help="Source classification config JSON.")
    parser.add_argument("--weights", default="", help="Override trained classification weights.")
    parser.add_argument("--source", default="", help="Override validation image root. Defaults to <data>/val from config.")
    parser.add_argument("--normal-class", default="Normal", help="Class name treated as normal.")
    parser.add_argument("--device", default="0", help="Inference device. Defaults to GPU 0.")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size. Must be <= 32.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Override image size. Defaults to config value.")
    parser.add_argument("--min-threshold", type=float, default=0.01, help="Minimum threshold to evaluate.")
    parser.add_argument("--max-threshold", type=float, default=0.99, help="Maximum threshold to evaluate.")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold step.")
    parser.add_argument("--output", default="", help="Output txt path. Defaults to research/results/<run>/threshold_sweep.txt.")
    return parser.parse_args()


def resolve_config_path(path: str) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    return config_path.resolve()


def require_batch_limit(batch: int) -> None:
    if batch <= 0 or batch > 32:
        raise SystemExit(f"Batch must be between 1 and 32, got {batch}.")


def float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise SystemExit(f"Step must be positive, got {step}.")
    count = int(round((stop - start) / step)) + 1
    values = [round(start + i * step, 4) for i in range(max(count, 0))]
    return [value for value in values if value <= stop + 1e-9]


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def infer_weights_path(cfg: dict, override: str) -> Path:
    if override:
        return Path(override).resolve()
    project = resolve_relative_path(cfg.get("project"), YOLOV11_ROOT)
    name = str(cfg.get("name") or "").strip()
    if not project or not name:
        raise SystemExit("Could not infer weights path from config; pass --weights explicitly.")
    return Path(project) / name / "weights" / "best.pt"


def infer_source_root(cfg: dict, override: str) -> Path:
    if override:
        return Path(override).resolve()
    data_root = resolve_relative_path(cfg.get("data"), YOLOV11_ROOT)
    if not data_root:
        raise SystemExit("Could not infer validation source from config; pass --source explicitly.")
    return Path(data_root) / "val"


def infer_output_path(cfg: dict, override: str, weights_path: Path) -> Path:
    if override:
        return Path(override).resolve()
    run_name = weights_path.parents[1].name
    return (Path(__file__).resolve().parents[1] / "research" / "results" / run_name / "threshold_sweep.txt").resolve()


def gather_images(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Validation image root does not exist: {root}")
    images = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise SystemExit(f"No images found under {root}")
    return images


def metric(num: int, den: int) -> float:
    return num / den if den else 0.0


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def find_normal_index(names: dict | list, normal_class: str) -> int:
    if isinstance(names, dict):
        pairs = [(int(idx), name) for idx, name in names.items()]
    else:
        pairs = list(enumerate(names))
    for idx, name in pairs:
        if str(name) == normal_class:
            return int(idx)
    raise SystemExit(f"Normal class '{normal_class}' not found in model names: {names}")


def collect_scores(
    weights_path: Path,
    image_paths: list[Path],
    normal_class: str,
    device: str,
    batch: int,
    imgsz: int,
) -> list[tuple[float, bool]]:
    ensure_yolov11_importable()
    from ultralytics import YOLO

    model = YOLO(str(weights_path), task="classify")
    scored: list[tuple[float, bool]] = []

    for chunk in chunked(image_paths, batch):
        chunk_results = model.predict(
            source=[str(path) for path in chunk],
            verbose=False,
            stream=False,
            batch=len(chunk),
            device=device,
            imgsz=imgsz,
        )
        for path, result in zip(chunk, chunk_results, strict=True):
            normal_idx = find_normal_index(result.names, normal_class)
            probs = result.probs.data.tolist()
            p_normal = float(probs[normal_idx])
            abnormal_conf = 1.0 - p_normal
            is_abnormal = path.parent.name != normal_class
            scored.append((abnormal_conf, is_abnormal))
    return scored


def compute_rows(scores: list[tuple[float, bool]], thresholds: list[float]) -> list[dict[str, int | float]]:
    actual_abnormal = sum(1 for _, is_abnormal in scores if is_abnormal)
    actual_normal = len(scores) - actual_abnormal
    rows: list[dict[str, int | float]] = []

    for threshold in thresholds:
        tp = fn = fp = tn = 0
        for abnormal_conf, is_abnormal in scores:
            pred_abnormal = abnormal_conf >= threshold
            if is_abnormal and pred_abnormal:
                tp += 1
            elif is_abnormal and not pred_abnormal:
                fn += 1
            elif not is_abnormal and pred_abnormal:
                fp += 1
            else:
                tn += 1

        recall = metric(tp, actual_abnormal)
        precision = metric(tp, tp + fp)
        specificity = metric(tn, actual_normal)
        accuracy = metric(tp + tn, len(scores))
        f1 = metric(2 * precision * recall, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": recall,
                "precision": precision,
                "specificity": specificity,
                "accuracy": accuracy,
                "f1": f1,
                "normal_filtered": tn,
                "normal_left": fp,
                "abnormal_kept": tp,
                "abnormal_missed": fn,
            }
        )
    return rows


def write_report(
    output_path: Path,
    weights_path: Path,
    source_root: Path,
    device: str,
    batch: int,
    imgsz: int,
    normal_class: str,
    scores: list[tuple[float, bool]],
    rows: list[dict[str, int | float]],
) -> None:
    actual_abnormal = sum(1 for _, is_abnormal in scores if is_abnormal)
    actual_normal = len(scores) - actual_abnormal
    best_f1 = max(rows, key=lambda row: row["f1"])
    best_recall = max(rows, key=lambda row: row["recall"])
    best_accuracy = max(rows, key=lambda row: row["accuracy"])

    header = [
        "Classification Threshold Sweep",
        f"weights={weights_path}",
        f"source={source_root}",
        f"device={device}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"normal_class={normal_class}",
        f"total_images={len(scores)}",
        f"actual_abnormal={actual_abnormal}",
        f"actual_normal={actual_normal}",
        "",
        f"best_f1_threshold={best_f1['threshold']:.2f} f1={format_percent(best_f1['f1'])} recall={format_percent(best_f1['recall'])} precision={format_percent(best_f1['precision'])}",
        f"best_recall_threshold={best_recall['threshold']:.2f} recall={format_percent(best_recall['recall'])} specificity={format_percent(best_recall['specificity'])}",
        f"best_accuracy_threshold={best_accuracy['threshold']:.2f} accuracy={format_percent(best_accuracy['accuracy'])}",
        "",
        "threshold\ttp\tfn\tfp\ttn\trecall%\tprecision%\tspecificity%\taccuracy%\tf1%\tnormal_filtered\tnormal_left\tabnormal_kept\tabnormal_missed",
    ]

    lines = header[:]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    f"{row['threshold']:.2f}",
                    str(int(row["tp"])),
                    str(int(row["fn"])),
                    str(int(row["fp"])),
                    str(int(row["tn"])),
                    format_percent(float(row["recall"])),
                    format_percent(float(row["precision"])),
                    format_percent(float(row["specificity"])),
                    format_percent(float(row["accuracy"])),
                    format_percent(float(row["f1"])),
                    str(int(row["normal_filtered"])),
                    str(int(row["normal_left"])),
                    str(int(row["abnormal_kept"])),
                    str(int(row["abnormal_missed"])),
                ]
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_batch_limit(args.batch)

    config_path = resolve_config_path(args.config)
    cfg = load_json_config(config_path)
    weights_path = infer_weights_path(cfg, args.weights)
    source_root = infer_source_root(cfg, args.source)
    imgsz = args.imgsz if args.imgsz > 0 else int(cfg.get("imgsz") or 640)
    output_path = infer_output_path(cfg, args.output, weights_path)
    thresholds = float_range(args.min_threshold, args.max_threshold, args.step)
    image_paths = gather_images(source_root)

    print(f"[data] images={len(image_paths)} abnormal={sum(1 for p in image_paths if p.parent.name != args.normal_class)} normal={sum(1 for p in image_paths if p.parent.name == args.normal_class)}")
    print(f"[run] weights={weights_path} device={args.device} batch={args.batch} imgsz={imgsz}")
    scores = collect_scores(
        weights_path=weights_path,
        image_paths=image_paths,
        normal_class=args.normal_class,
        device=args.device,
        batch=args.batch,
        imgsz=imgsz,
    )
    rows = compute_rows(scores, thresholds)
    write_report(
        output_path=output_path,
        weights_path=weights_path,
        source_root=source_root,
        device=args.device,
        batch=args.batch,
        imgsz=imgsz,
        normal_class=args.normal_class,
        scores=scores,
        rows=rows,
    )
    print(f"[report] {output_path}")


if __name__ == "__main__":
    main()
