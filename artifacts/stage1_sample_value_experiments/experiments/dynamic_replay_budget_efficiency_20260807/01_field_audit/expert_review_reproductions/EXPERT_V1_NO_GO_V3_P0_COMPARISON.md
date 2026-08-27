# Expert v1.0.0 NO-GO Versus Current v3: P0 Evidence

## Scope And Limits

This comparison covers the seven P0 findings in the supplied independent
review. The BudgetedReplay v1.0.0 source TAR, ZIP, and Wheel are absent, so the
expert code locations and supplied outputs remain review evidence rather than a
fresh source-level rerun. Current-v3 claims are backed by local source hashes,
line references, unit tests, and `local_v3_p0_20260809/V3_P0_REPRODUCTION.json`.

No expert training code, real training job, test set, or blind holdout was run.

## Result

| Finding | Current v3 status | Conclusion |
|---|---|---|
| P0-01 OOF fold cube | `PARTIALLY_MITIGATED` | The cube error is gone, but held-out lineage is not proven. |
| P0-02 `best.pt` | `CONFIRMED_ABSENT` | v3 fixes epoch 200 and commits `self.last`; end-to-end release binding remains future work. |
| P0-03 `val_op` reuse | `PARTIALLY_MITIGATED` | The leaking method is absent, but so is the independent `val_target` needed by A. |
| P0-04 test oracle | `CONFIRMED_ABSENT` | No current runtime reference exists; future release still needs a sealed-test gate. |
| P0-05 shared threshold | `CONFIRMED_ABSENT` | The two constrained metrics are computed from separate feasible sets. |
| P0-06 conflicting CLD | `NOT_APPLICABLE` | Both conflicting implementations and the desired target-direction mechanism are absent. |
| P0-07 formal matrix | `CONFIRMED_PRESENT` | The 236-run matrix is internally valid but scientifically superseded and remains `HELD`. |

## Important Reproductions

### OOF Is Structurally Better But Not Lineage-Complete

Current v3 requires exactly one epoch-200 prediction per training identity and
checks the full sample/label set. It does not construct a sample-by-fold cube.
However, the local reproduction deliberately placed:

```text
job-a/fold_00/...csv -> oof_fold=01
job-b/fold_01/...csv -> oof_fold=00
```

and `build_oof_epoch200_reference()` still returned PASS. The remaining
requirement is not another count check; it is a cryptographic relationship among
sample fold assignment, fold-run checkpoint identity, training-member exclusion,
and prediction source.

### The Dual-Metric Bug Is Fixed In v3

For the same ten-row fixture used to separate the constraints, v3 returned:

```text
FN-limit point: FN=1, TN=4
TN-target point: FN=2 when target TN=5
```

Because the values differ, v3 is not reusing the FN-limit confusion matrix for
`FN_at_TN`. The final schema should additionally retain the target-TN threshold
and tie group, but that is an auditability enhancement rather than the reviewed
P0 bug.

### Current Matrix Must Not Run

The matrix validator correctly reports 236 HELD runs in four cycles. That only
proves internal consistency. It contains no Q reliability arm, no target
direction A, no diversity D, no defect guard, and no method-matched R2. Running
it now would spend the limited compute window on a superseded RHO-only question.

## Current Test Baseline

```text
full v3 collection: exit 2
reason: tests/stage1_dynamic_replay_v3/test_recovery.py imports missing recovery.py

excluding that unfinished contract: 62 passed
new expert-delivery audit tests: 7 passed
new P0 reproduction tests: 2 passed
```

The full suite is therefore not green and must not be described as such.

## Decision

P0-02, P0-04, and P0-05 no longer justify blocking by themselves. P0-01 and
P0-03 remain incomplete contracts; P0-06 identifies a missing research
capability; P0-07 remains a direct release blocker. Formal training,
engineering gate, pilot release, and blind holdout stay forbidden.
