# Repository change specification

This is a construction specification, not an assertion that the modules below already
satisfy it. Until every implementation test, evidence gate, and global completion audit
passes, **formal training remains forbidden**. The implementation must wrap or extend the
current v3 runtime without mutating archived Phase-1 training sources.

## Module boundaries

| Module | Required responsibility | Fail-closed boundary |
|---|---|---|
| `stage1_dynamic_replay_v3/qrad_contract.py` | Typed Q/R/A/D fields, ordered-gate states, allowed evidence types | Reject any weighted aggregate or missing gate provenance |
| `stage1_dynamic_replay_v3/reliability_gate.py` | Q label/identity/source eligibility and frozen strata | Reject mutable identity, mixed data roles, or absent audit provenance |
| `stage1_dynamic_replay_v3/residual_learnability.py` | R candidates from completed-epoch trajectory signals | Keep confidence/loss/RHO/forgetting/AUM as named proxies, never utility |
| `stage1_dynamic_replay_v3/target_alignment.py` | A sign/stratum against the independent `val_target` local objective | Reject `val_op`/test references and current/future epoch inputs |
| `stage1_dynamic_replay_v3/set_coverage.py` | D source/video/feature/gradient set coverage with deterministic tie handling | Reject single-sample D claims and nondeterministic unlogged ties |
| `stage1_dynamic_replay_v3/selection_pipeline.py` | Q→R→A→D sequencing and arm-specific snapshots | Reject missing intermediate universes and composite ranks |
| `stage1_dynamic_replay_v3/exposure_ledger.py` | Planned/actual slots, unique IDs, repeats, cumulative optimizer-visible exposure, steps | Reject requested-count accounting and unmatched traces |
| `stage1_dynamic_replay_v3/runtime_controller.py` | One-epoch-lag decisions plus R1/R2/current-loss/no-replay policies | Source epoch must equal logical epoch minus one |
| `stage1_dynamic_replay_v3/training_runtime.py` | Auxiliary replay gradient accumulation into frozen base optimizer steps | Reject any added optimizer step or changed base draw trace |
| `stage1_dynamic_replay_v3/evaluation.py` | Fixed-epoch tie-safe FN=0..95 frontier and secondary endpoints | Reject best-checkpoint and role/identity drift |
| `stage1_dynamic_replay_v3/confirmatory_analysis.py` | Seed pairing, sign-flip tests, Holm family, stability and stopping gates | Reject missing pairs and partial-comparator success |
| `stage1_dynamic_replay_v3/recovery.py` | RNG/controller/draw-plan/global-step restore, quarantine, canonical closeout | Reject partial atomic state or generation mismatch |

Existing `matrix.py`, `seeds.py`, assignment, supersession, completion, and resource
telemetry modules must consume these contracts. They must not create an assignment or
release state merely because preregistration validation passes.

## Required CLIs

| CLI | Inputs | Outputs | Release effect |
|---|---|---|---|
| `validate_preregistration_v3.py` | this directory | `PREREGISTRATION_VALIDATION.json` | none |
| `build_qrad_snapshot.py` | frozen manifests, completed epoch receipt, Q/R/A inputs | identity-bound signal snapshot | none |
| `build_qrad_selection.py` | snapshot, arm, slot trace, seed | decision and selected occurrence plan | none |
| `validate_selection_trace.py` | snapshots and decisions | per-gate lineage audit | none |
| `validate_exposure_parity.py` | occurrence and epoch ledgers for a paired arm set | exact slot/exposure/step parity audit | none |
| `evaluate_confirmatory_seed_pairs.py` | canonical fixed-epoch predictions and closeouts | paired endpoints, Holm/stability audit | none |
| `validate_stage1_completion.py` | all expert, literature, code, run, endpoint, and mirror audits | global completion audit | none unless every registered condition passes |

Every CLI writes through a temporary sibling, fsyncs, and publishes with an atomic
replace. Existing immutable evidence is never overwritten; a new attempt uses a new
attempt ID and supersession receipt.

## Command-line contract

The future builders must require explicit values for experiment root, run/attempt/job
identity, assignment generation, training seed/scope, arm, source-epoch receipt,
manifests and their expected SHAs, fixed checkpoint, output directory, and dry-run audit
mode. Defaults may locate this registered experiment but may not infer data roles,
release work, open test, or substitute `best.pt`.

## Test contract

Implementation is test-first and must include at least:

- `test_qrad_contract.py`: rejects weights, role mixing, signal-as-utility, missing
  intermediate universes, and non-causal epochs;
- `test_selection_pipeline.py`: deterministic Q→R→A→D snapshots, R2 terminal
  randomization, quota/unique/repeat matching, and tie behavior;
- `test_exposure_ledger.py`: exact denominators, replay occurrence conservation,
  cumulative unique/repeat counts, base trace equality, and optimizer-step equality;
- `test_training_runtime.py`: replay contributes gradient inside the same optimizer step
  while base order and step count remain fixed;
- `test_evaluation.py`: tie-safe FN 0 through 95, fixed checkpoint identity,
  `TN_at_FN95`, and `FN_at_TN68253`;
- `test_confirmatory_analysis.py`: paired seed identity, exact sign flips, Holm ordering,
  12/14 wins, worst-seed and dual-end gates, missing-pair failure, and safety stopping;
- `test_recovery.py`: RNG/OOM/kill/disk/atomic-write/resume and canonical completion;
- CLI integration tests proving no job, assignment, gate, pilot, training, or blind/test
  access occurs during validation.

For every claimed repair, retain the failing-first test receipt and the subsequent green
receipt with command, exit code, bytes, and SHA-256. Markdown-only references do not
count as reproduced behavior.

## Runtime artifact layout

Within an authorized future run directory, publish only registered identities:

```text
run_identity.json
signal_snapshots/epoch_NNNN.parquet
selection_decisions/epoch_NNNN.parquet
replay_occurrences/epoch_NNNN.parquet
exposure_ledgers/epoch_NNNN.json
epoch_receipts/epoch_NNNN.json
checkpoints/epoch_0200.pt
predictions/val_op_epoch_0200.csv
predictions/val_op_epoch_0200.csv.manifest.json
closeout/run_closeout.json
telemetry/resource_samples.parquet
```

The scientific report consumes canonical closeouts only. Synthetic or canary paths are
labelled as engineering evidence and cannot enter paired endpoint tables.

## Migration

1. Freeze source-tree SHA, role manifests, prediction IDs, current v3 behavior, and the
   current all-HELD registry. Preserve v1/v2 runtime material as historical evidence.
2. Add the typed schema and validators without changing runtime behavior; run negative
   tests first and record both red and green receipts.
3. Add Q/R/A/D snapshot and selection modules behind a disabled feature flag. Compare
   their dry-run identities against frozen synthetic fixtures only.
4. Add occurrence conservation and `validate_exposure_parity`; no real replay run is
   permitted until base exposure and optimizer-step invariants pass.
5. Add auxiliary replay gradient accumulation, recovery, and closeout checks behind the
   same disabled flag. Exercise mock/canary data only and label it non-scientific.
6. Add confirmatory analysis and verify exact synthetic cases, including intentional
   harm, missing pairs, ties, multiplicity failure, and stopping.
7. Re-run expert/source, tripartite, literature, preregistration, source-tree, data-role,
   and Desktop SHA audits. Only the global completion contract may later authorize an
   engineering gate; this specification itself never does.

## Rollback

Rollback is a feature-flag and artifact-supersession operation. Disable the Q/R/A/D
entrypoint, restore the previous wrapper/config reference, and leave archived training
code and all immutable evidence untouched. Quarantine partial attempts by attempt ID;
never delete or relabel them as canonical. Restore the previous assignment generation
only through a signed supersession receipt, then re-run source-tree and registry audits.
If any identity, exposure, recovery, or analysis check fails, the all-HELD state is the
safe terminal state; no partial implementation may fall through to the old pilot or
assignment directories.

