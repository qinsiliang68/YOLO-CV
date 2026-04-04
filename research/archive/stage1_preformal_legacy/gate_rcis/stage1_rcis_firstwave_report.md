# Stage-1 RCIS First-Wave Report

## 1. Scope

This report records the **first-wave formal RCIS run** under the current stage-1 protocol.

The goal is not to replace the existing stage-1 evidence chain. The goal is narrower:

- keep the current best baseline fixed
- test whether **information-driven resampling** can further improve normal filtering
- rank RCIS variants using the same stage-1 operating-point rule

Current baseline before RCIS:

- `G4 current best HardMix + P0`
- full name: `yolo11l-cls + calibration + hn02 + HardMix`

The current working assumption is:

- stage-1 headroom now mainly comes from **sample exposure**
- the first-wave RCIS should therefore stay conservative
- `boundary / hardness / redundancy` should carry the main signal
- `uncertainty` stays lightweight
- `flip` and `quality_penalty` are not allowed to dominate the first formal pass

## 2. Ranking Rule

All RCIS variants are ranked by the unchanged stage-1 rule:

1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0` ascending

This report must not use default-threshold accuracy as the primary judge.

## 3. First-Wave Experiment List

The default first-wave suite is fixed to three groups only:

| Group | Label | Role |
| --- | --- | --- |
| R1 | `RCIS boundary-only sampling` | test whether operating-point proximity alone already captures most remaining value |
| R2 | `RCIS core linear information sampling` | current recommended safer first-pass RCIS |
| R3 | `RCIS full exploratory information sampling` | keep the earlier full linear version as a risk-control comparison |

The following configs exist but are **not** part of the default first-wave suite:

- `stage1_gate_l_rcis_noquality.json`
- `stage1_gate_l_rcis_noflip.json`

They are reserved for second-wave manual ablations only if first-wave RCIS shows real gains.

## 4. RCIS Signal Definition

The current first-pass linear score remains:

`I = boundary + uncertainty + disagreement + rarity + hardness - redundancy - quality_penalty`

But the first-wave default no longer treats every term equally.

### 4.1 Boundary-only

Only one signal is active:

- `boundary`

All other coefficients are set to zero.

### 4.2 RCIS Core

This is the current recommended safer first-pass setup.

| Signal | Coefficient | Status |
| --- | ---: | --- |
| `boundary` | `1.0` | primary |
| `uncertainty` | `0.2` | light auxiliary |
| `flip` | `0.0` | disabled |
| `rarity` | `0.4` | secondary positive signal |
| `hardness` | `0.9` | primary |
| `redundancy` | `0.8` | primary suppression term |
| `quality_penalty` | `0.0` | disabled |

Interpretation:

- `boundary` keeps stage-1 focused on the high-recall operating points
- `hardness` keeps dangerous hard normal / hard positive samples emphasized
- `redundancy` suppresses repeated low-gain patterns
- `uncertainty` is kept but intentionally weak
- `flip` is disabled because it is only a `P0/P2` proxy, not real training dynamics
- `quality_penalty` is disabled because sewer dark/blurred/dirty frames can still be real deployment hard cases

### 4.3 Full Exploratory

This keeps the older exploratory full linear setting as a comparison group.

| Signal | Coefficient |
| --- | ---: |
| `boundary` | `1.0` |
| `uncertainty` | `0.4` |
| `flip` | `0.8` |
| `rarity` | `0.6` |
| `hardness` | `0.8` |
| `redundancy` | `0.7` |
| `quality_penalty` | `0.4` |

This group is **not** the recommended first formal stage-1 direction. It exists to test whether the earlier, more aggressive exploratory scoring is actually worth the added instability risk.

## 5. Expected Output Artifacts

For each RCIS experiment, the pipeline should produce:

- RCIS dataset materialization summary
- trained gate weights
- exported train/val features
- PTSG-style evaluation materials
- final `ptsg_summary.csv`
- threshold sweep files

The suite-level summary should finally be written to:

- `research/results/stage1_gate_rcis_suite/stage1_rcis_suite_summary.csv`
- `research/results/stage1_gate_rcis_suite/stage1_rcis_suite_summary.json`
- `research/results/stage1_gate_rcis_suite/stage1_rcis_suite_summary.md`

## 6. Placeholder Result Table

The table below is reserved for the first-wave run results.

| Group | Label | Best Variant | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G4 | current best HardMix + P0 | `P0` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | baseline |
| R1 | RCIS boundary-only sampling | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| R2 | RCIS core linear information sampling | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| R3 | RCIS full exploratory information sampling | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## 7. Placeholder Decision Notes

To be filled after the run:

- Best row:
  - `TBD`
- Does RCIS beat current `G4 HardMix + P0`?
  - `TBD`
- If yes, which signal mix appears most useful?
  - `TBD`
- If no, is stage-1 likely saturated under current protocol?
  - `TBD`

## 8. Planned Interpretation Rule

When results come back, interpret them in this order:

1. Did any RCIS group beat baseline on `Spec@R99.5`?
2. If `Spec@R99.5` ties, did any RCIS group improve `Spec@R99.0` without worsening downstream indicators?
3. If `R2 core` beats `R3 full exploratory`, keep the safer RCIS interpretation.
4. If `R3 full exploratory` wins but `R2` does not, treat that win cautiously because the extra gain may be tied to unstable proxy signals.
5. If none wins, stage-1 should likely remain with `HardMix + P0`.
