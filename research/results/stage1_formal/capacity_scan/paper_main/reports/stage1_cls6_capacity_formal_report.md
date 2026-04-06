# Stage-1 CLS6 Capacity Formal Report

## Protocol
- Auxiliary task: `source-side six-class capacity scan`
- Batch size: `24`
- Epochs: `200`
- Checkpoint policy: save every epoch
- Formal ranking: `Accuracy > AUROC > AUPRC`

## Main Table
See `paper_main/tables/table_stage1_cls6_capacity_main.md`.

## Final Ranking
1. `yolo11x-cls`
2. `yolo11m-cls`
3. `yolo11l-cls`
4. `yolo11s-cls`
5. `yolo11n-cls`

## Interpretation as a Source-Side Representation Reference
- `yolo11x-cls` is the source-side six-class leader under the formal auxiliary rule.
- The six-class ordering is useful for characterizing representation quality on the source task.
- However, it should be interpreted strictly as an auxiliary reference rather than the official selection criterion for the stage-1 gate.
