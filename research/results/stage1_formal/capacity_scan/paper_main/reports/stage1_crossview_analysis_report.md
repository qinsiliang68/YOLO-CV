# Stage-1 Cross-View Analysis Report

## Why CLS6 and Binary Gate Should Not Be Conflated
The formal stage-1 objective is direct binary-gate model selection under a recall-constrained filtering setting. The source-side six-class task serves only as an auxiliary representation view and should not be conflated with the official gate objective.

## Rank-Mismatch Analysis
See `paper_main/tables/table_stage1_crossview_rank_gap.md`.

## Implications for Stage-1 Model Selection
- `yolo11x-cls` leads the auxiliary cls6 view, but `yolo11m-cls` remains the direct binary-gate leader.
- `yolo11l-cls`, `yolo11s-cls`, and `yolo11n-cls` keep the same relative order across both views, indicating that the main disagreement is concentrated in the two strongest models.
- This mismatch supports the claim that source-side six-class ranking cannot substitute for direct binary-gate selection.

## Cross-View Table
- `yolo11l-cls`: cls6 rank `3`, gate rank `3`, gap `0`
- `yolo11m-cls`: cls6 rank `2`, gate rank `1`, gap `1`
- `yolo11n-cls`: cls6 rank `5`, gate rank `5`, gap `0`
- `yolo11s-cls`: cls6 rank `4`, gate rank `4`, gap `0`
- `yolo11x-cls`: cls6 rank `1`, gate rank `2`, gap `-1`
