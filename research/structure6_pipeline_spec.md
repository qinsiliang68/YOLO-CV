# Structure-6 Pipeline Spec

This spec freezes the label system and directory layout for the first runnable version of the
classification-to-CAM-to-detection pipeline.

## Final Detection Classes

The detector task is fixed to six structural defect classes:

| ID | Detector class | Source labels | Notes |
| --- | --- | --- | --- |
| 0 | CrackBreak | RB | Crack, break, collapse style wall-body failure. |
| 1 | SurfaceDamage | OB, PF | Surface damage, corrosion-like appearance, or production/manufacturing defect. |
| 2 | Deformation | DE | Geometric deformation of the pipe body. |
| 3 | JointDislocation | FS | Joint displacement, dislocation, or similar interface offset. |
| 4 | Intrusion | IS | Joint or pipe intrusion by sealing material or similar structure. |
| 5 | Infiltration | IN | Leakage or infiltration expression. |

`Normal` is kept as an explicit class only for classification and gating, not for detector outputs.

## Source Pretraining Datasets

Use the raw SewerML library to extract one compact single-label source set as the source-domain pretraining data:

- `YOLOv11/datasets/sewerml_cls6_train3000`

Do not train the first version of the pipeline on raw multi-label SewerML.

## Decision Policy For CAM

CAM outputs are candidate boxes only. They are not assumed to be detector-ready labels.

Per-class review rules:

- Usable rate `>= 70%`: CAM main route
- Usable rate `40% - 70%`: CAM + manual seed boxes
- Usable rate `< 40%`: switch the class to manual seed boxes early

Usable rate means:

`(direct_use + minor_edit_use) / inspected_total`

## Directory Layout

The repository keeps code in Git and keeps datasets local-only.

Tracked in Git:

- `research/`
- `scripts/`
- `YOLOv11/configs/`

Local-only data positions:

- `data/sewerml/annotations`
- `data/sewerml/images_all`
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
- `data/normal/images`

Derived training datasets that stay local:

- `YOLOv11/datasets/sewerml_cls6_train3000`
- `YOLOv11/datasets/struct6_cls_target`
- `YOLOv11/datasets/struct6_det_pseudo`
- `YOLOv11/datasets/struct6_det_reviewed`

## Execution Order

1. Extract `sewerml_cls6_train3000` from the raw SewerML library.
2. Run source classification pretraining on `sewerml_cls6_train3000`.
3. Fine-tune on target-domain classification data.
4. Export CAM heatmaps.
5. Convert CAM heatmaps to pseudo boxes.
6. Review per-class pseudo boxes and decide CAM/manual route.
7. Build detector dataset and train the first detector baseline on another machine.
