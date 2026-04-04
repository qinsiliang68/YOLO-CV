# Stage-1 RCIS Design Note

## Goal

Stage-1 does not optimize default-threshold accuracy. Its target remains:

- hold defect recall at the high-recall operating points
- improve normal filtering under the same recall constraint

The current best baseline before RCIS is:

- `yolo11l-cls + calibration + hn02 + HardMix`
- best gate decision returns to `P0 = calibrated p_abnormal`

This means the remaining headroom is more likely to come from **sample exposure** than from more complex post-hoc trust formulas.

## First-Pass RCIS

This implementation treats sample value as an information score and maps it to class-aware sampling weights.

The first-pass linear score is:

`I = boundary + uncertainty + disagreement + rarity + hardness - redundancy - quality_penalty`

The current implementation uses:

- `boundary`
  - closeness to the current `R99.5 / R99.0` operating thresholds
- `uncertainty`
  - binary entropy of calibrated abnormal probability
- `disagreement`
  - proxy instability from the gap / decision disagreement between `P0` and a `P2`-style trust proxy
- `rarity`
  - small-cluster / off-centroid samples in embedding space
- `hardness`
  - high abnormal score for normal samples, or near-threshold support for abnormal samples
- `redundancy`
  - large-cluster / high-centroid-similarity samples
- `quality_penalty`
  - blur / exposure / contrast penalty from image heuristics

## Why This Is Only a First Pass

This version is intentionally lightweight so it can be dropped into the current stage-1 pipeline.

It does **not** yet implement:

- multi-checkpoint training-dynamics variance
- explicit forgetting-event counting across epochs
- nonlinear multiplicative gating over redundancy / quality
- cluster caps for near-duplicate video bursts

Those are second-wave upgrades only if the first RCIS pass shows real gains over the current HardMix baseline.

## Current RCIS Suite

The one-click suite now runs three experiments:

- `RCIS boundary-only`
  - isolates whether near-threshold boundary proximity is already enough
- `RCIS full linear`
  - uses the full first-pass score
- `RCIS no-redundancy`
  - ablates the redundancy penalty

All experiments are still ranked by the existing stage-1 rule:

1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0`

## Entrypoint

The default human-facing entrypoint is now:

```powershell
uv run main.py
```

The active task is:

- `stage1_gate_rcis_suite`
