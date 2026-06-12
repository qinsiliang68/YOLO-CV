# YOLO-CV

This workspace has been reset for the final dataset:

- new active direction: one final dataset, not a v1/v2/v3/v4 sequence
- target defect pool: `PF / DE / FS / RB / AF / OB`, 15,000 images per class
- additional Normal images will be sampled according to the next gate/detection
  experiment design
- raw image source is preserved outside the repository at
  `C:\Sewer-ML\sewerml_train_images\`
- source annotation CSV is preserved at
  `YOLOv11/datasets/sewerml_annotations/SewerML_Train.csv`

Old v1/v3/stage-1 materials, thesis drafts, scripts, runtime configs and evidence
notes were archived under:

`_recycle_bin/pre_final_dataset_reset/`

Start with:

- `_recycle_bin/pre_final_dataset_reset/CONCLUSIONS.md`
- `_recycle_bin/pre_final_dataset_reset/FILE_MANIFEST.md`

## Active Rules

Do not delete or move:

- `C:\Sewer-ML\sewerml_train_images\`
- `YOLOv11/datasets/sewerml_annotations/SewerML_Train.csv`
- YOLO source code under `YOLOv11/`
- pretrained base weights in the repository root

New generated datasets should use a new path, for example:

- `research/materials/final_dataset/`
- `research/results/final_dataset/`
- `C:\Sewer-ML\final_dataset\`

The repository `data/` folder must not contain large image datasets. Keep all
large raw, intermediate and exported datasets under `C:\Sewer-ML\` so editors do
not scan hundreds of thousands of files inside the repository.

SewerML is multi-label. New manifests should preserve all original label columns
and may add a `primary_class` only for export/folder layout convenience.
