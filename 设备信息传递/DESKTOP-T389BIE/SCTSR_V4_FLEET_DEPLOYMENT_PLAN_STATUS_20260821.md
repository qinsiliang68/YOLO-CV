# SCTSR V4 Fleet Deployment Plan Status - 2026-08-21

## Current operational state

- The eight frozen `DISCOVERY_PARENT` placements have produced their immutable E120 parents; branch execution is in progress.
- On `P25`, `SCTSR_DISCOVERY_S005_T_TO_R2_AT_160` trained E121-E200 and published its E200 endpoint evidence. Its non-training completion marker remains pending while the closeout-only binding/index defects are fixed; the completed epochs and endpoint are not rerun.
- On `P25`, `SCTSR_DISCOVERY_S005_T_F` retry-v2 entered real E121 CUDA training at 2026-08-25 10:02 UTC (`epoch_0121.generation_1.inprogress`, RTX 3090 7,212 MiB, 81% utilization). It uses parent SHA `E8CE7912A4199047B2812981425F00031CDE45D070A7DD74ECBC761576792665`, a 3,000-row `T_STRESS` pool, and the frozen 80-epoch participant manifest SHA `3951AFF850A89282A6B96CFC8B7A7AE5E04A0C463296B241975DA7D7A1E9BBB6`.
- Deployment remains serial. Data/checkpoint/participant correctness is the first gate; documentation and repeated same-invocation hashing are not launch blockers.
- All epoch transactions, checkpoints, participant ledgers, receipts, logs, and failed attempts remain on their original node's `D:` output root; active source, dataset views, controls, and caches remain on `C:`.

## 2026-08-25 implementation note

- Fixed Windows extended-length traversal for deep preserved quarantine paths in `build_artifact_index`.
- Removed only adjacent duplicate byte revalidation for terminal-only E200 finalization; normal training still revalidates before endpoint publication.
- Resume keeps a legal power-of-two AMP backoff instead of resetting the saved optimizer trajectory.
- External formal asset registries are accepted only when byte-bound to the immutable formal-input snapshot.
- Focused regression: `25 passed`; compileall and `git diff --check`: PASS.
- The first `T_F` start failed before output creation/claim/GPU because a deployment shim encoded the parent recovery-pointer byte count as 1,283 instead of 1,275. The failed receipt is preserved; retry-v2 changed only that bound byte count and then entered real training.

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
