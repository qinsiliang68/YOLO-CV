# SCTSR V4 Fleet Deployment Plan Status - 2026-08-21

## Current operational state

- Eight eligible `DISCOVERY_PARENT` jobs are already running and have each passed an independent real Epoch 1 artifact validation.
- No `DISCOVERY_BRANCH` job has been started because its immutable E120 parent dependency is not complete yet.
- Deployment remains serial: one eligible node is launched and proven before the next node is touched.
- All epoch transactions, checkpoints, receipts, logs, and any failed generations remain on their original nodes.

## Frozen placement plan

- Assignment seed: `20260821`
- Plan digest: `FAC5F01A7E8D8ABC5DF09C4C12DBE70C7732382B433A992CE15BA02236642536`
- Plan file SHA-256: `DF61879B9D46DDAEA7ACEF3B29721C3DC60A5829ACD1F9DFFFF8A75CD7C5ECBD`
- Seed registry SHA-256: `8C467E8A86E7E138BD7FD726098FB495C11FF4A12D1670B1FE7D6F856D2D3651`
- Jobs: 198 total, using 12 active machines and one buffer machine.
- Buffer machine: `P36`; it has zero regular placements.

The active machine input order was selected before branch deployment so the seeded plan exactly preserves the eight already-running parent placements:

| Parent job | Machine |
| --- | --- |
| `PARENT_431404666` | `P25` |
| `PARENT_1583055843` | `P24` |
| `PARENT_906427910` | `P35` |
| `PARENT_51447201` | `P34` |
| `PARENT_725590974` | `P12` |
| `PARENT_322761319` | `P14` |
| `PARENT_327787489` | `P26` |
| `PARENT_2019192314` | `P13` |

## Files

- `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_fleet_training_20260821/DEPLOYMENT_PLAN.json`
- `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_fleet_training_20260821/BUILD_DEPLOYMENT_PLAN_RECEIPT.json`
- `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_fleet_training_20260821/inputs/SEED_REGISTRY.json`

The plan fixes placement only. It does not claim that branch jobs are eligible before their own parents pass E120 validation, and it does not duplicate any currently running parent job.
