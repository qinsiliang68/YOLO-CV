# Training Machine Runbook

This runbook is for the machine that will actually train models. The intended flow is:
`git pull`, sync the `uv` environment, move datasets into fixed local-only paths, then run.

## 1. Environment Bootstrap

```powershell
cd C:\GitHub\YOLO-CV
.\scripts\setup.ps1 -Backend cu128
.\scripts\check.ps1
```

Default one-click pipeline entrypoint:

```powershell
uv run --no-sync main.py
```

`main.py` runs the default CLS6 source pretraining, target classification fine-tuning, CAM export,
and pseudo-box generation pipeline. It stops early with a clear error if the target-domain inputs
have not been placed into the expected local-only paths yet.

Available `-Backend` values:

- `cu128`: default choice for recent NVIDIA CUDA 12.8 compatible drivers
- `cu126`: use if the training machine is better matched to CUDA 12.6
- `cpu`: CPU-only debugging

`setup.ps1` does three things:

1. Runs `uv sync --frozen` with the selected Torch backend
2. Stores the chosen backend in `.uv-torch-backend`
3. Creates the local-only dataset directory skeleton

## 1.1 Default Hardware Profile

The committed runtime defaults are tuned for a workstation with an RTX 3090 24 GB.

- Source classification defaults: `imgsz=224`, `batch=128`, `workers=8`
- Target classification defaults: `imgsz=224`, `batch=128`, `workers=8`
- Detection defaults: `imgsz=640`, `batch=16`, `workers=8`

These defaults assume the current baseline models:

- `yolo11n-cls.pt` for classification
- `yolo11n.pt` for detection

If you later switch to a larger model such as `yolo11s-cls.pt` or `yolo11s.pt`, lower batch size first.
Safe fallback points are:

- classification: `batch=64`
- detection: `batch=8`

## 2. Fixed Dataset Paths

### Source Classification Pretraining

- `YOLOv11/datasets/sewerml_hla_cls3_focus/train/...`
- `YOLOv11/datasets/sewerml_hla_cls3_focus/val/...`
- `YOLOv11/datasets/sewerml_hla_cls6_focus/train/...`
- `YOLOv11/datasets/sewerml_hla_cls6_focus/val/...`

### Target Classification Fine-Tuning

- `YOLOv11/datasets/struct6_cls_target/train/<ClassName>/...`
- `YOLOv11/datasets/struct6_cls_target/val/<ClassName>/...`
- `YOLOv11/datasets/struct6_cls_target/test/<ClassName>/...`

Recommended class names:

- `Normal`
- `CrackBreak`
- `SurfaceDamage`
- `Deformation`
- `JointDislocation`
- `Intrusion`
- `Infiltration`

### Reviewed Detection Dataset

- `YOLOv11/datasets/struct6_det_reviewed/images/train/...`
- `YOLOv11/datasets/struct6_det_reviewed/images/val/...`
- `YOLOv11/datasets/struct6_det_reviewed/images/test/...`
- `YOLOv11/datasets/struct6_det_reviewed/labels/train/...`
- `YOLOv11/datasets/struct6_det_reviewed/labels/val/...`
- `YOLOv11/datasets/struct6_det_reviewed/labels/test/...`

### CAM Pseudo-Box Dataset

- `YOLOv11/datasets/struct6_det_pseudo/images/train/...`
- `YOLOv11/datasets/struct6_det_pseudo/images/val/...`
- `YOLOv11/datasets/struct6_det_pseudo/images/test/...`
- `YOLOv11/datasets/struct6_det_pseudo/labels/train/...`
- `YOLOv11/datasets/struct6_det_pseudo/labels/val/...`
- `YOLOv11/datasets/struct6_det_pseudo/labels/test/...`

### Raw and Intermediate Local Data

- `data/sewerml/raw_reference`
- `data/foshan/images`
- `data/foshan/labels_cls`
- `data/foshan/cam_outputs`
- `data/foshan/pseudo_boxes`
- `data/foshan/reviewed_boxes`
- `data/local/images`
- `data/local/labels_cls`
- `data/local/cam_outputs`
- `data/local/pseudo_boxes`
- `data/local/reviewed_boxes`
- `data/local/inference_samples`
- `data/normal/images`

## 3. Main Commands

### Source Classification Pretraining

```powershell
.\scripts\cls_pretrain.ps1 -Config YOLOv11/configs/runtime/cls_source_cls6.json
```

### Target Classification Fine-Tuning

```powershell
.\scripts\cls_finetune_target.ps1 -Config YOLOv11/configs/runtime/cls_target_struct6.json
```

### Export CAM

```powershell
.\scripts\export_cam.ps1 `
  -Weights C:\GitHub\YOLO-CV\YOLOv11\runs\classify\struct6_target_from_cls6\weights\best.pt `
  -Source C:\GitHub\YOLO-CV\YOLOv11\datasets\struct6_cls_target\val `
  -Output C:\GitHub\YOLO-CV\data\local\cam_outputs `
  -LabelManifest C:\GitHub\YOLO-CV\data\local\labels_cls\val_manifest.csv
```

### Convert CAM To Pseudo Boxes

```powershell
.\scripts\cam_to_pseudobox.ps1 `
  -CamManifest C:\GitHub\YOLO-CV\data\local\cam_outputs\manifest.csv `
  -Output C:\GitHub\YOLO-CV\YOLOv11\datasets\struct6_det_pseudo `
  -Thresholds C:\GitHub\YOLO-CV\research\cam_threshold_template.json `
  -KeepNormal
```

### Generate Research Figures

```powershell
.\scripts\plot_research_figures.ps1 -Mode sewerml
.\scripts\plot_research_figures.ps1 -Mode cam-review
.\scripts\plot_research_figures.ps1 -Mode train-metrics `
  -ResultsCsv C:\GitHub\YOLO-CV\YOLOv11\runs\train\struct6_reviewed_baseline\results.csv `
  -Output C:\GitHub\YOLO-CV\outputs\figures\training\struct6_reviewed_baseline.png `
  -Title "Struct6 Reviewed Baseline"
```

Default SewerML figure output:

- `outputs/figures/sewerml/01_raw_label_distribution.png`
- `outputs/figures/sewerml/02_alignment_summary.png`
- `outputs/figures/sewerml/03_struct6_cooccurrence.png`
- `outputs/figures/sewerml/04_waterlevel_profile.png`
- `outputs/figures/sewerml/05_group_share_donut.png`
- `outputs/figures/sewerml/06_multilabel_cardinality.png`
- `outputs/figures/sewerml/07_struct6_by_split.png`
- `outputs/figures/sewerml/08_top_label_pairs.png`

### Train The First Structure-6 Detector

```powershell
.\YOLOv11\scripts\train.ps1
```

Default files:

- `YOLOv11/configs/runtime/train_detect_struct6_reviewed.json`
- `YOLOv11/configs/datasets/struct6_detect_reviewed.yaml`

Pseudo-box baseline alternative:

- `YOLOv11/configs/runtime/train_detect_struct6_pseudo.json`
- `YOLOv11/configs/datasets/struct6_detect_pseudo.yaml`

### Validate Or Test

```powershell
.\YOLOv11\scripts\val.ps1
.\YOLOv11\scripts\test.ps1
```

### Predict

```powershell
.\YOLOv11\scripts\predict.ps1 -Source C:\GitHub\YOLO-CV\data\local\inference_samples
```

## 4. Ground Rules

- Data never goes to Git; code and configs do.
- All training commands run through the root `uv` project.
- Local `YOLOv11` source edits take effect immediately because the scripts import the repo checkout directly.
- If you change CUDA backend, rerun `.\scripts\setup.ps1 -Backend <cpu|cu126|cu128>`.
