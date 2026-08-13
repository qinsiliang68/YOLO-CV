# SCTSR v4 R2 specification change request

Status: `OPEN_HARD_BLOCKER`

This request does not relax the implementation and does not authorize training. It records a contradiction between the frozen taskbook claim and the currently registered assets.

## Frozen requirements in conflict

- T is the 3,000-ID `RUN_010` stress set with digest `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`.
- R2 contains 3,000 unique canonical-base identities.
- R2 has zero identity overlap with T.
- R2 exactly matches T label and `oof_group_id` quota (in addition to dynamic bucket and OOF fold).
- Matching may not relax, use nearest groups, or reuse identities.

These requirements are jointly infeasible with the registered canonical base and OOF assignments.

## Reproduction

The content audit joins:

- `artifacts/stage1_oof_folds_10fold_20260617/train_oof_assignments.csv`;
- `artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/generated/selections/RUN_010/selection_manifest.csv`.

It removes all T identities, keeps `oof_y_true=0`, and compares the remaining count in each `oof_group_id` with the T quota.

The full four-field joint audit observes **172 shortage strata** and **378 missing occurrences**. Two examples are:

| joint stratum `(label|bucket|fold|group)` | T quota | zero-overlap candidates | shortage |
| --- | ---: | ---: | ---: |
| `0|learnable_hard|0|filename_bucket_1000:382` | 8 | 3 | 5 |
| `0|learnable_hard|5|filename_bucket_1000:500` | 2 | 0 | 2 |

The machine-readable summary is `docs/stage1_sctsr_v4/tdd_receipts/commit_02/R2_INFEASIBILITY_SUMMARY.txt`. The current implementation must raise `R2_QUOTA_INFEASIBLE`; it must not silently weaken matching.

## Owner decision required

Exactly one preregistered change is required before formal R2 construction can pass:

1. replace the fixed T stress set with a pool whose zero-overlap comparator is feasible, changing the T digest; or
2. amend the R2 group constraint with a scientifically justified coarser pre-terminal stratum, explicitly acknowledging it is not exact `oof_group_id` matching; or
3. add a new frozen canonical-base asset containing at least the missing zero-overlap normal identities in the two groups, while preserving every other data-role and leakage constraint.

Repetition, replacement sampling, T overlap, defect-label substitution, nearest-group matching, and quota tolerance are rejected because they violate the current taskbook.
