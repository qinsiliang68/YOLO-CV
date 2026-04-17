# Goldilocks Machine A Completion Summary (2026-04-17)

This note summarizes the currently completed Machine A outputs under:

- `research/materials/stage1_formal/gate_goldilocks_campaign`
- `research/results/stage1_formal/gate_goldilocks_campaign`

## Scope Completed

- Peak scan: `R-W01` to `R-W11`, `T-W01` to `T-W11` (22 runs)
- Random controls: `Rand50-S1`, `Rand50-S2`, `Rand50-S3` (3 runs)
- Combine runs: `F-R`, `F-T`, `F-RT` (3 runs)

Total recorded in summary CSV: 28 runs.

## Key Metrics (Spec@R99.5)

Peak best:

- `R-W02`: `0.607143` (best epoch `69`)
- `T-W02`: `0.607143` (best epoch `69`)

Random controls:

- `Rand50-S1`: `0.559524` (best epoch `48`)
- `Rand50-S2`: `0.583333` (best epoch `36`)
- `Rand50-S3`: `0.535714` (best epoch `81`)

Combine:

- `F-R`: `0.488095` (best epoch `1`)
- `F-T`: `0.488095` (best epoch `1`)
- `F-RT`: `0.559524` (best epoch `65`)

## Two-Class Organization

Class A (already curated for Git):

- Top-level summary/result files:
  - `goldilocks_campaign_summary.csv`
  - `peak_results.json`
- Per-run key artifacts (`R/T peaks + Rand50 + F-*`):
  - `all_checkpoints_index.csv`
  - `best_epoch_manifest.json`
  - `epoch_gate_summary.csv`
  - `epoch_gate_summary.json`
  - `epoch_gate_summary.md`

Class B (local-only / non-essential for Git):

- Dense intermediate traces:
  - `per_epoch_gate/` (all epochs, all calibration sweep internals)
- Visualization binaries:
  - `epoch_gate_dashboard.png`
- Runtime logs:
  - `machineA_stdout_*.log`
  - `machineA_stderr_*.log`

Current local-only volume snapshot:

- `per_epoch_gate`: ~7200 files, ~0.174 GB
- `epoch_gate_dashboard.png`: 6 files, ~1.95 MB
- runtime logs: 6 files, ~0.30 MB

## Notes

- This summary reflects Machine A status only.
- Cross-machine combine targets (`F-RD`, `F-TD`, `F-RTD`, `F-RTDC`) require merged peaks from the other machine.
