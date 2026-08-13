# YOLO-CV

Start here for scope control:

- `CURRENT_RESEARCH.md` is the current source-of-truth map.
- `artifacts/README.md` explains which artifact folders are current results,
  supporting run evidence, or pending OOF prediction outputs.
- `docs/README.md` lists the current documentation entrypoints.
- `docs/stage1_sctsr_v4/IMPLEMENTATION_GUIDE.md` is the entrypoint for the
  isolated SCTSR v4 implementation. It is implementation-only, formal training
  remains disabled, and it does not supersede historical Stage1 results.

When a number can refer to several datasets or experiments, name the scope first
before using it.  Historical material under `_recycle_bin/` is audit-only unless
explicitly requested.

This workspace has been reset for the final dataset:

- new active direction: one final dataset, not a v1/v2/v3/v4 sequence
- target defect pool: `PF / DE / FS / RB / AF / OB`, mapped to CJJ
  181-2012 defect categories
- first-stage classification is a six-class multi-label task
- final local dataset root:
  `data/final_sewerml_dataset/`
- sampling seed is fixed to `20260606`
- raw image source is preserved outside the repository at
  `C:\Sewer-ML\sewerml_train_images\`
- source annotation CSVs are preserved locally under
  `YOLOv11/datasets/sewerml_annotations/` and are not tracked by Git

Old v1/v3/stage-1 materials, thesis drafts, scripts, runtime configs and evidence
notes were archived under:

`_recycle_bin/pre_final_dataset_reset/`

Start with:

- `_recycle_bin/pre_final_dataset_reset/CONCLUSIONS.md`
- `_recycle_bin/pre_final_dataset_reset/FILE_MANIFEST.md`
- `_recycle_bin/pre_final_dataset_reset/essay/docs/essay3.tex`
- `_recycle_bin/pre_final_dataset_reset/top_level/SAMPLING_PROTOCOL.md`

## Current Dataset

The current final dataset uses one canonical image copy under
`data/final_sewerml_dataset/Det/images/{split}` and reproducibility manifests
under `data/final_sewerml_dataset/manifests/`.

Splits:

- `train`: 60,000 images, sampled last, primary-balanced with 10,000 images for
  each primary class (`PF / DE / FS / RB / AF / OB`)
- `val_cal`: 20,000 images, natural long-tail distribution, used only for
  calibration fitting
- `val_op`: 20,000 images, natural long-tail distribution, used only for
  operating-threshold selection after calibration
- `test`: 20,000 images, natural long-tail distribution, held out for final
  evaluation

All four splits must be filename-disjoint. The manifest CSV files are the source
of truth for reproducing which images were sampled, where they came from in the
original CSV, and which labels they carry.

For first-stage classification, `train_primary_class` records why a training
image was selected into the balanced training pool. It is not the only training
label. Training labels must use the original multi-hot columns
`PF / DE / FS / RB / AF / OB`.

## CJJ 181 Mapping Rules

The six active SewerML labels are selected by mapping SewerML/Fotomanualen
labels to defect categories used in CJJ 181-2012 sewer inspection practice. This
mapping was established in the archived thesis/protocol materials and is carried
forward as the current label policy.

| CJJ category | CJJ code | SewerML label | Current role |
| --- | --- | --- | --- |
| 正常 | -- | `Defect = 0` | negative/background reference for gate-style views |
| 破裂 | `PL` | `PF` | target defect |
| 变形 | `BX` | `DE` | target defect |
| 错口 | `CK` | `FS` | target defect |
| 腐蚀 | `FS` | `RB` | target defect |
| 沉积 | `CJ` | `AF` | target defect |
| 障碍物 | `ZW` | `OB` | target defect |

The current first-stage classifier trains only the six target defect labels.
Normal is listed here only as the background/negative definition for binary
gate-style views or future comparison experiments.

Important naming rule: SewerML `FS` maps to CJJ `CK` (错口). CJJ `FS` means
腐蚀 and maps to SewerML `RB`. Do not compare these two `FS` codes by name
alone.

Quality and non-target labels:

- `OK` and `PH` are not CJJ defect targets. Pure OK/PH images without any of the
  six target classes are excluded from the final target pool.
- If an image has a target class and also has `OK` or `PH`, keep the image and
  preserve the original OK/PH columns in the manifest for audit.
- Holdout/non-target SewerML labels are not used as first-stage target classes.
  If they co-occur with a target class, the image may remain in the dataset, but
  the first-stage supervised label vector is still only
  `PF / DE / FS / RB / AF / OB`.

## Active Rules

Do not delete or move:

- `C:\Sewer-ML\sewerml_train_images\`
- local SewerML source annotation CSVs under `YOLOv11/datasets/sewerml_annotations/`
- YOLO source code under `YOLOv11/`
- pretrained base weights in the repository root

The repository `data/` folder currently contains the local final sampled dataset
and is ignored by Git except for reproducibility manifests under
`data/final_sewerml_dataset/manifests/`. Do not stage or upload large image
datasets or raw source annotation tables into the repository history.

SewerML is multi-label. New manifests should preserve all original label columns
and may add a `primary_class` only for export/folder layout convenience.
