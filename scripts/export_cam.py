from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pipeline_common import STRUCT6_CLASSES, ensure_yolov11_importable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> tuple[torch.Tensor, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return logits.detach(), cam.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Grad-CAM overlays for a classification model.")
    parser.add_argument("--weights", required=True, help="Path to a trained classification weights file.")
    parser.add_argument("--source", required=True, help="Image file or directory to process.")
    parser.add_argument("--output", required=True, help="Directory to save CAM outputs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Classification image size.")
    parser.add_argument("--crop-fraction", type=float, default=1.0, help="Crop fraction used by classify_transforms.")
    parser.add_argument("--label-manifest", default="", help="Optional CSV with filename -> target class mapping.")
    parser.add_argument("--device", default="", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay heatmap alpha.")
    parser.add_argument("--save-npy", action="store_true", help="Save raw heatmap arrays as .npy files.")
    parser.add_argument("--limit", type=int, default=0, help="Optional image count limit for quick debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and exit.")
    return parser.parse_args()


def collect_images(source: Path) -> tuple[list[Path], Path]:
    if source.is_file():
        return [source], source.parent
    images = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    return images, source


def load_label_lookup(path: str) -> dict[str, str]:
    if not path:
        return {}
    lookup: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename") or row.get("image") or row.get("relative_path") or ""
            label = row.get("assigned_class") or row.get("label") or row.get("class") or ""
            if filename and label:
                lookup[filename] = label
    return lookup


def find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def blend_overlay(image_bgr: np.ndarray, cam_uint8: np.ndarray, alpha: float) -> np.ndarray:
    heatmap_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)


def normalize_names(names) -> list[str]:
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    images, source_root = collect_images(source_path)
    if args.limit > 0:
        images = images[: args.limit]

    if args.dry_run:
        print(f"images={len(images)} source_root={source_root} output={output_root}")
        return

    ensure_yolov11_importable()
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms

    label_lookup = load_label_lookup(args.label_manifest)
    model_wrapper = YOLO(args.weights, task="classify")
    model = model_wrapper.model
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    names = normalize_names(getattr(model, "names", getattr(model_wrapper, "names", STRUCT6_CLASSES)))
    name_to_index = {name: idx for idx, name in enumerate(names)}
    transform = classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
    grad_cam = GradCAM(model, find_last_conv(model))

    overlays_dir = output_root / "overlays"
    heatmaps_dir = output_root / "heatmaps"
    raw_dir = output_root / "raw"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "relative_path",
                "source_path",
                "predicted_class",
                "predicted_index",
                "confidence",
                "target_class",
                "target_index",
                "heatmap_path",
                "overlay_path",
            ],
        )
        writer.writeheader()

        for image_path in images:
            relative_path = image_path.relative_to(source_root)
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

            with torch.enable_grad():
                logits, cam = grad_cam(tensor, 0)
                probs = torch.softmax(logits, dim=1)[0]
                predicted_index = int(probs.argmax().item())
                predicted_class = names[predicted_index]
                target_class = label_lookup.get(relative_path.as_posix()) or label_lookup.get(image_path.name) or predicted_class
                target_index = name_to_index.get(target_class, predicted_index)
                logits, cam = grad_cam(tensor, target_index)
                probs = torch.softmax(logits, dim=1)[0]
                confidence = float(probs[target_index].item())

            cam_uint8 = (cam * 255.0).clip(0, 255).astype(np.uint8)
            cam_resized = cv2.resize(cam_uint8, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            overlay = blend_overlay(image_bgr, cam_resized, alpha=args.alpha)

            overlay_path = overlays_dir / relative_path
            heatmap_path = heatmaps_dir / relative_path
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(heatmap_path), cam_resized)

            if args.save_npy:
                raw_path = (raw_dir / relative_path).with_suffix(".npy")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, cam)

            writer.writerow(
                {
                    "filename": image_path.name,
                    "relative_path": relative_path.as_posix(),
                    "source_path": str(image_path),
                    "predicted_class": predicted_class,
                    "predicted_index": predicted_index,
                    "confidence": f"{confidence:.6f}",
                    "target_class": target_class,
                    "target_index": target_index,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )


if __name__ == "__main__":
    main()
