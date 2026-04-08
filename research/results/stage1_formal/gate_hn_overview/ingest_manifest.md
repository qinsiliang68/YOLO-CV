# Stage-1 HN Ingest Manifest

- Source zip: `C:\Users\28898\Desktop\stage1_HN_materials.zip`
- Source extracted root: `C:\Users\28898\Desktop\stage1_HN_materials_extracted\stage1_HN_materials`
- Repo target root: `C:\GitHub\YOLO-CV`
- Archive type: `non_pt_materials_only`

This ingest absorbed the non-PT HN materials into the repo-local formal directories. The corresponding PT checkpoints live outside this zip and were not part of the ingest.

## Imported Materials
- `research/materials/stage1_formal/gate_hn_assets`
- `research/materials/stage1_formal/gate_hn_m_sweep`
- `research/materials/stage1_formal/gate_hn_x_crosscheck`

## Imported Results
- `research/results/stage1_formal/gate_hn_m_sweep`
- `research/results/stage1_formal/gate_hn_x_crosscheck`
- `research/results/stage1_formal/gate_hn_overview`

## Coverage
- `yolo11m` raw + summary coverage: `hn00, hn02, hn04, hn06, hn08, hn10, hn12`
- `yolo11m` manifest-only coverage: `hn14, hn16, hn18, hn20`
- `yolo11x` manifest-only coverage: `hn00, hn02`

## Summary Counts
- `yolo11m` HN rows: `11`
- `yolo11x` HN rows: `2`
- HN overview rows: `13`

## Intentionally Omitted From Repo Copy
- The original desktop zip after successful ingest
- The extracted desktop temp directory after successful ingest
- `per_epoch_gate` raw trees from `yolo11m` to keep the repo working set smaller
- All PT checkpoints, because they are not present in this archive
