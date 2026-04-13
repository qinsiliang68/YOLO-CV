# Stage-1 HN Rebuild Indexes

- These CSVs store the image-level backflow selections relative to the hn00 baseline.
- If the HN dataset view directories are deleted, rebuild each ratio by taking the gate base dataset and reassigning the listed `image_id` values from `train/Normal` to `train/Abnormal`.
- `stage1_hn_backflow_selection_summary.csv` checks manifest count vs derived count.
