# Formal mechanism evidence cards

## EC01: Late replay exposure

**Mechanism.** Repeated exposure changes the realized optimizer path; the same identities may cease to provide residual correction after the base learner has moved.

**Direct evidence.** P002, P017-P019, P032-P033, P040-P042 and P045 establish state-dependent curricula/replay, exposure effects or optimizer-path dependence.

**Counterevidence and limits.** P031 finds several adaptive curricula worse than uniform. P045 warns that early substitutions can also propagate strongly. Most replay papers study continual learning, while Stage1 repeats already labeled same-task images.

**Minimal falsifier.** Same selection and seed, canonical learner, identical restart boundaries: continuous versus same-peak taper. No replay is a separate baseline.

**Required fields.** Intended and realized per-epoch ratio, identity exposures, optimizer steps, LR, restart lineage, difficult-normal and weak-defect raw trajectories.

**Useful negative result.** If taper does not improve safety or lowers both tails, late exposure is not the main mechanism for this frozen selection.

## EC02: Timing versus cumulative dose

**Mechanism.** A taper can appear beneficial merely because it supplies fewer repeated samples and fewer optimizer steps, rather than because late exposure is harmful.

**Direct evidence.** P017-P018, P032, P040-P041 and P045 show that pacing, repeated processing, effective learning rate, total exposure and path length are coupled.

**Counterevidence and limits.** No reviewed paper studies this exact 120k additive replay and FN-constrained endpoint. A dose-matched early concentration can itself cause early overfit.

**Minimal falsifier.** Continuous, same-peak taper and exact dose-matched taper. The latter uses the same cumulative replay exposure as continuous while stopping replay after epoch 160.

**Required fields.** Cumulative replay slots, role-specific cumulative exposure, optimizer-step count, per-identity concentration, effective identity count and all-epoch tail probes.

**Useful negative result.** If only same-peak taper helps, total dose is the stronger explanation; if both tapers help, timing remains eligible.

## EC03: Weak-defect guard

**Mechanism.** Normal replay may lower difficult-normal scores while also lowering weak-defect scores. Replacing some normal slots with learnable weak defects may protect the constrained tail.

**Direct evidence.** P004-P006 and P034 motivate separate target directions; P024-P028 motivate constrained tail treatment; P041 and P050 show that aggregate performance can hide protected-subgroup degradation.

**Counterevidence and limits.** None of these papers proves that the proposed OOF learnable-tail rule is beneficial. High loss and low confidence can include noise (P008-P010, P020-P023, P036, P044).

**Minimal falsifier.** Fixed normal IDs, ratio and schedule; replace slots with raw guard, learnable guard or matched-random defect at 10% and 20%. Total slots never increase.

**Required fields.** Guard identities, eligibility rule, OOF fold, learning trajectory, realized role exposures, difficult-normal/weak-defect losses and raw probabilities.

**Useful negative result.** If matched-random defect performs equally, class/tail balance rather than guard ranking is the mechanism. If every guard harms normal efficacy, guard allocation is too costly at that ratio.

## EC04: Cross-seed confirmation and controls

**Mechanism.** A policy is useful only if its paired effect distribution improves, not because one optimizer path happens to land at a favorable point.

**Direct evidence.** P008, P043, P049-P052 and the existing Stage1 same-selection reversals support paired seeds, estimator-versus-training variance separation and conditional claims.

**Counterevidence and limits.** Fourteen seeds cannot prove universality across architectures, data revisions or future hyperparameter searches. The blind holdout remains unbound.

**Minimal falsifier.** Fourteen unseen seeds with final policy, same-selection continuous, dynamic no-guard, global random, disjoint difficulty-matched random and no replay.

**Required fields.** Paired seed, machine/environment block, exact RNG/resume lineage, frozen manifests, lock SHA, run failures and all registered endpoints.

**Useful negative result.** A well-powered null or reversal identifies a policy that is not stable enough for deployment and still bounds the attainable replay effect under the canonical learner.

## EC05: Gradient mechanism probe

**Mechanism.** Magnitude measures leverage; signed alignment to separate target losses estimates local benefit or harm; set cancellation and finite residuals determine whether first-order geometry survives a real update.

**Direct evidence.** P004-P006, P008, P011-P016, P029-P031, P034-P036, P038 and P042.

**Counterevidence and limits.** Influence is fragile in deep networks (P013-P014); large gradients can be noise or cancel (P008, P029, P031, P036); repeated finite replay is not a one-step perturbation.

**Minimal falsifier.** At key checkpoints, repeat last-layer/projected measurements for fixed candidate and tail panels, then compare signs with same-state finite virtual updates and later paired arm effects.

**Required fields.** Norm, target-specific dot/cosine, reference-model mean/std/sign agreement, aggregate cancellation, projection identity, optimizer-aware update, finite-minus-linear residual and checkpoint hash.

**Useful negative result.** Poor repeatability or no out-of-seed relationship blocks a gradient-selected replay arm while preserving gradient fields as diagnostics.
