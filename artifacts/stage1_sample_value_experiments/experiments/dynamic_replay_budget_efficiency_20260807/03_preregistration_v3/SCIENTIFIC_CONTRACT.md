# Scientific contract: finite-budget dynamic replay

## Scientific state and claim boundary

This package is `PREREGISTERED_NOT_RUN`. No Stage1 replay intervention represented here
has run, so no candidate method is described as effective. Paper evidence, expert claim,
code presence, synthetic/canary output, Stage1 observation, and future intervention are
separate evidence types. Only a paired real replay intervention can establish utility.

## Fixed concepts

- **Q = reliability**: evidence that the candidate identity and label are sufficiently
  reliable for inclusion. Q is the first eligibility gate or a frozen reliability
  stratum; it is not a reward term.
- **R = residual/reducible learnability**: evidence that the current model still has
  reducible error on the candidate, conditional on Q and the completed model state.
- **A = direction toward the independent FN95 local objective**: the sign or stratum of
  a candidate's direction relative to `val_target`, never `val_op` or test.
- **D = set-conditioned coverage**: source, video, feature, and gradient coverage of the
  selected set after Q/R/A, not a context-free single-sample value.

Q/R/A/D may be combined only by **ordered gating**, frozen stratification, or factorial
ablation. No arbitrary weighted total score is allowed. Confidence, loss, RHO, gradient,
forgetting, AUM, and coverage are candidate signals and are **not utility evidence**.
No offline association, paper result, source-code feature, or synthetic check can be
promoted to a replay-benefit claim.

## Temporal and role separation

For replay at logical epoch `t`, every state-dependent Q/R/A/D input is frozen from the
atomically completed epoch `t-1`; future information is rejected. Static Q audit fields
must be frozen before the run. A may read only the independent `val_target` role.

The exact role registry is `train`, `OOF`, `val_target`, `val_model`, `val_cal`, `val_op`,
and `test`. Membership and SHA identity are frozen and mutually checked. OOF predictions
for a fold member must come from a model that did not train on that member. `val_model`
is discovery-only, `val_cal` freezes a single operational threshold when needed, and
`val_op` reports the fixed-checkpoint confirmatory frontier. Test is sealed until all
methods, checkpoints, thresholds, analyses, and stopping decisions are immutable.

## Budget and optimizer visibility

Every arm uses the identical canonical base draw trace and base optimizer settings.
Replay gradients are accumulated into the already scheduled base optimizer step; replay
does not create an extra optimizer step. Thus optimizer steps and base exposures are
identical across all arms. Every replay arm uses the same frozen per-epoch planned slots
and identity-multiplicity skeleton. Fairness is judged from actual optimizer-visible
sample occurrences, not requested counts or dataset length.

`REPLAY_BUDGET_A` is fully numeric: the frozen base has 120,000 samples, batch 128,
938 optimizer steps per epoch, 200 epochs, and 187,600 total optimizer steps. Epoch 1 has
no replay because no completed source epoch exists. Epochs 2 through 200 each expose 600
replay occurrences (0.5% of the base), for 119,400 planned optimizer-visible replay
occurrences. Selection refreshes every five replay epochs in 40 blocks. Each block starts
with 600 distinct identities and forbids reuse from earlier blocks, producing 24,000
planned cumulative unique IDs and 95,400 planned repeat occurrences. The final block has
four epochs. Any eligible pool with fewer than 600 unused IDs fails the arm; it may not
shrink the budget or silently reuse identities.

The base lock is
`configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json`, SHA-256
`7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E`.
The 0.5% budget is the lowest already registered historical pressure and is adopted only
as a finite non-outcome design constraint; its historical registration is not utility
evidence.

For each epoch and cumulatively, record planned/actual replay slots, unique IDs, repeat
occurrences, optimizer-visible base/replay exposure, and optimizer steps. R1 global
random, R2 method-matched random, current-loss, and every Q/R/A/D arm must have exact
slot and identity-multiplicity parity. `NO_REPLAY` has zero replay exposure and the same base steps; its
contrast estimates replay presence, while selection utility is identified by matched
replay controls.

R2 randomizes terminal identities within the treatment's preterminal candidate universe
and matches strata, source/video quotas, unique count, repeat multiplicity, per-epoch
slots, cumulative actual exposure, and optimizer steps. It must not inherit the terminal
signal under test.

## Falsifiable intervention logic

The minimal ladder is Q, Q→R, Q→R→A, and Q→R→A→D. Discovery estimates each incremental
factor only against its proper preterminal matched control. A frozen full policy then
faces all four controls on disjoint unseen training seeds. Utility remains
`UNKNOWN_IN_STAGE1` unless real replay pairs satisfy every confirmatory rule. A failure,
safety stop, incomplete seed pair, or endpoint degradation cannot be reworded as benefit.
