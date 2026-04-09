# stage1 info-sampling-lite PNG manifest usage

This manifest set records image file lists and mapping rules for `stage1_formal_gate_info_sampling_lite`.

## Files

- `stage1_info_sampling_lite_png_file_manifest.csv`
- `stage1_info_sampling_lite_png_rebuild_map.csv`

## Scope

- `stage1_info_sampling_lite_png_file_manifest.csv`:
  - full PNG list under `H:\stage1_info_sampling_lite_materials`
  - includes path, file size, and timestamp
- `stage1_info_sampling_lite_png_rebuild_map.csv`:
  - mapping for dataset-view PNGs under `YOLOv11\datasets\stage1_formal_gate_info_sampling_lite`
  - maps each view image to `sewerml_gate2_train7200` by `split/class_name/file_name`
  - includes `source_exists_in_base_dataset` validation flag

## Important note

CSV files preserve list and mapping metadata only. They do not store pixel content.
Rebuild requires the base dataset to remain available.
