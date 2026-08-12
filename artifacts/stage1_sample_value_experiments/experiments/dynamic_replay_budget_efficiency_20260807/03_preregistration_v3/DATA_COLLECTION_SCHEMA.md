# Data collection and identity schema

`DATA_COLLECTION_SCHEMA.json` is the machine authority. All artifacts bind `run_id`,
`attempt_id`, `assignment_generation`, arm, seed, source manifests, and checkpoint where
applicable. Publication is atomic; resumed attempts must restore the same controller,
RNG, draw-plan, global-step, and assignment-generation identities.

## Signal snapshot and one-epoch lag

`sample_signal_snapshot` records sample/source/video identity and separate Q/R/A/D
fields. There is no weighted-score field. For logical epoch `t`,
`source_epoch = logical_epoch - 1`. `selection_decision` records the ordered gate sequence,
terminal selection stage, reason, composition mode, and random seed. `evidence_type` is
one of PAPER, EXPERT_CLAIM, CODE, SYNTHETIC, STAGE1_OBSERVATION, or FUTURE_INTERVENTION.

## Draw and exposure accounting

`replay_occurrence` is one row per optimizer-visible draw and includes the physical draw
slot, occurrence index, augmentation seed, and optimizer step. The epoch ledger records:

- planned and actual base/replay slots;
- per-epoch unique identities and `unique_replay_ids_cumulative`;
- per-epoch repeat count and `cumulative_repeat_occurrences_actual`;
- `optimizer_visible_base_exposure_actual` and
  `optimizer_visible_replay_exposure_actual`;
- planned/actual optimizer steps and global step.

The replay denominator is the count of actual optimizer-visible sample occurrences.
Requested slots, selected-list length, and unique-ID count are never substitutes.
Parity checks fail on missing epochs, duplicates, excess/missing occurrences, step drift,
or any divergence of actual replay traces among matched arms.

The planned conservation target is 600 replay occurrences for each epoch 2 through 200,
119,400 cumulatively, with 40 five-epoch identity blocks (the last has four epochs),
24,000 cumulative unique identities, and 95,400 cumulative repeat occurrences. Every
actual ledger must reconcile to these totals and the canonical 187,600 optimizer steps.

## Prediction, endpoint, and closeout

Predictions bind sample/label identity, split role, fixed checkpoint epoch and SHA, plus
both validation manifest SHAs. A paired endpoint row joins treatment and comparator by
the same unseen training seed and carries the raw-frontier delta, both endpoint deltas,
seed win, and dual-end degradation flag.

`run_closeout` binds the canonical attempt, assignment generation, actual budget totals,
checkpoint policy, RNG receipt, completion receipt, and real resource telemetry. OOM,
kill, disk failure, identity mismatch, non-atomic partials, or incomplete telemetry closes
the attempt as failed/quarantined; it does not authorize a hyperparameter change.
