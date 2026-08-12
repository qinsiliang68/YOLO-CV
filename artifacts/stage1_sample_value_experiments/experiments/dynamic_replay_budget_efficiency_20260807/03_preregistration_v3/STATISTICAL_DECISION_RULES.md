# Statistical decision rules

## Frozen seed phases

Seed derivation is deterministic and excludes all historical/v2 seeds. The sets are
pairwise disjoint: 10 discovery seeds, 10 ranking-ablation seeds, and 14 unseen
confirmation seeds. Discovery may choose one complete policy and discard mechanisms for
feasibility or safety, but cannot support a policy-effectiveness claim. The policy,
signals, thresholds, budgets, and code/source hashes freeze before confirmation outcomes
are revealed.

## Endpoint and oracle prohibition

The sole primary endpoint is the tie-safe **FN=0..95** raw safety-frontier normalized
AUC at fixed epoch 200 on `val_op`; higher is better. The trapezoidal ordinate is TN
divided by the frozen normal count at each integer FN budget. Secondary endpoints are
`TN_at_FN95`, `FN_at_TN68253`, dual-end degradation rate, seed win rate, and worst-seed
delta. All raw per-seed endpoints and deltas are retained.

The checkpoint is fixed at epoch 200. `best.pt` is forbidden. `val_op` and test may not
choose a method, checkpoint, threshold, seed, stop, or release. A single deployed
threshold, if required, freezes on `val_cal`; the raw frontier remains an evaluation
object. A test oracle is forbidden, and test stays sealed until the complete analysis is
frozen.

## Confirmation and multiplicity

The frozen `T_QRAD` policy is paired by training seed against NO_REPLAY, R1 global random,
R2 method-matched random, and current-loss. For each comparator, the primary one-sided
test is the exact paired sign-flip test of the mean seed delta. The four p-values form one
family and use Holm control at familywise alpha 0.05. A confirmatory superiority claim
requires all four Holm rejections; no weighted endpoint or post-hoc subset may replace
this family.

Stability must also hold separately for every comparator: at least **12 of 14** positive
primary seed deltas, a nonnegative worst-seed primary delta, and zero seeds with both
`TN_at_FN95` worse and `FN_at_TN68253` worse. Report the full seed win rate, worst-seed
record, dual-end degradation rate, unadjusted p-value, and Holm-adjusted decision.

These intentionally strict zero-harm gates are falsifiable rules, not evidence that they
will be met.

## Stopping and missingness

There is no efficacy early stop. Confirmation schedules all 14 paired seeds. A
predeclared safety stop may terminate computation, but the only permitted scientific
state is `NOT_EVALUATED_OR_HARM`; it cannot claim benefit. A missing, corrupted,
superseded, or noncanonical member of any pair causes that confirmatory claim to fail
closed. OOM or resource failure never changes batch size, image size, optimizer,
precision, augmentation, worker count, checkpoint, or replay schedule within the study.

