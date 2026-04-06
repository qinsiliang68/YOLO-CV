# Stage-1 Gate Capacity Formal Report

## Protocol
- Primary task: `direct binary gate`
- Batch size: `24`
- Epochs: `200`
- Checkpoint policy: save every epoch
- Formal ranking: `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`
- Trainer `top1/acc/loss` is retained only as training-health information and does not determine the formal checkpoint.

## Main Table
See `paper_main/tables/table_stage1_gate_capacity_main.md`.

## Final Ranking
1. `yolo11m-cls`
2. `yolo11x-cls`
3. `yolo11l-cls`
4. `yolo11s-cls`
5. `yolo11n-cls`

## Top1-Best vs Gate-Best Mismatch
See `paper_main/tables/table_stage1_gate_top1_vs_gatebest.md`.
- `5` out of `5` models show `top1-best != gate-best` under the formal protocol.
- This mismatch indicates that trainer-side `top1` is not aligned with the recall-constrained gate objective and should not be used as the official checkpoint-selection rule.

## Early-Peak and Checkpoint-Dynamics Analysis
- No model reaches its gate-best checkpoint at epoch 1.
- `yolo11m-cls` peaks at epoch `78`, indicating that the formal optimum is reached in the middle stage rather than at the end of training.
- `yolo11x-cls` peaks at epoch `125`, suggesting that the largest-capacity model benefits from a longer optimization horizon, but still remains below the formal leader.
- `yolo11l-cls` peaks earlier at epoch `54`, which explains why the legacy preference for `l` is not preserved under the new gate-aware selection rule.

## Why Binary Gate Is the Official Stage-1 Selection View
- Stage-1 is defined as normal filtering under a recall constraint rather than default-threshold classification.
- The formal ranking therefore follows `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`, which directly reflects the thesis-facing gate objective.
- Under this rule, source-side classification quality remains informative but cannot replace direct binary-gate checkpoint selection.

## Key Findings
- Formal gate-capacity leader: `yolo11m-cls`
- Second control model: `yolo11x-cls`
- The current formal evidence supports `yolo11m-cls` as the primary backbone for subsequent HN, HardMix, and information-driven sampling experiments.
- The capacity scan therefore rewrites the exploratory `l/s`-centric intuition and establishes a new `m/x`-centric mainline for stage-1.
