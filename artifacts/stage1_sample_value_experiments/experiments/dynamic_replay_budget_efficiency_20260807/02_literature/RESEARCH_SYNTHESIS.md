# Stage1 full-text research synthesis

## Evidence gate

- Registered candidates: 155
- Claim-eligible full reads: 50
- Replication-depth reads: 34
- Structured notes: 50
- Visually reviewed PDF pages: 1,050

The earlier matrix labels such as `METHOD_READ=55` and `DEEP_READ=33` are superseded. Only papers listed in `FULL_TEXT_READING_LEDGER.csv`, with a matching note under `FULL_TEXT_EVIDENCE_NOTES/`, may support the formal protocol.

## What the literature does establish

1. Loss, confidence and gradient magnitude measure forms of difficulty or leverage; none establishes positive downstream value. P008, P029, P031 and P036 provide direct mathematical or empirical counterexamples.
2. Signed contribution is target- and state-dependent. P004, P005, P006, P011, P034 and P042 motivate separate alignment or finite-intervention diagnostics against difficult-normal and weak-defect targets.
3. Sample effects interact with the learner, surrounding set, budget and stochastic path. P013-P016, P037-P039 and P045-P052 rule out treating one static `V(x)` as universally stable.
4. A score and an exposure schedule are different interventions. P017-P019, P032-P033 and P040-P042 support measuring identity-level exposure and testing timing causally, but do not supply a transferable Stage1 stop epoch or ratio.
5. Tail safety must remain a separate constraint. P024-P028 support target-specific constrained or partial-tail evaluation; they do not justify blending normal benefit and weak-defect harm into one weighted scalar.
6. Initialization and implementation choices can reverse application behavior. P043, P049 and P050 support paired seeds, machine blocking, complete configuration locks and distributional reporting.

## What the literature does not establish

- No paper proves that `0.5%`, `1.0%` or `2.5%` is optimal for Stage1.
- No paper proves that epoch 140, 150 or 160 is the correct stopping point.
- No paper proves that `GapCritical`, a learnable-hard guard or gradient alignment will improve the Stage1 raw safety frontier.
- Continual-learning replay, active learning, pruning and language-model data selection are related mechanisms, not equivalent tasks.
- Positive first-order gradient alignment is not a finite repeated-replay guarantee.

## Registered causal direction

The value object is a conditional finite treatment effect:

```text
V(S | theta_t, canonical learner, replay ratio, timing, cumulative exposure,
      seed, optimizer path, surrounding data, protected tail)
```

The first frozen selection is used as a stress-test object because it already exhibits cross-seed sign reversal. It is not declared a correct value function.

The four result cycles are:

1. Compare continuous replay, same-peak late taper and no replay at `2.5%`.
2. Add exact cumulative-dose matching and transfer the timing comparison to `0.5%` and `1.0%`.
3. Freeze the eligible normal policy and replace, rather than add, `10%` or `20%` of replay slots with weak-defect guards and matched controls.
4. Freeze one complete policy and confirm it on 14 entirely unseen seeds against component ablations, global random, disjoint matched random and no replay.

All ratios are fractions of the complete 120,000-sample base training pool. Absolute row counts are derived implementation fields and never method names.

## Non-negotiable learner lock

Every formal arm uses the exact canonical 240-run learner. The machine-readable source of truth is `configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json`; its file SHA must be present in every preregistration row, queue job, training audit, resume record and result package. Literature hyperparameters are transfer boundaries only and may not alter this lock.

## Gradient decision

Gradient collection remains a bounded mechanism diagnostic at key checkpoints. It records magnitude, separate target dots/cosines, cross-reference dispersion, cancellation and finite-minus-linear residuals. It does not create a gradient-ranked formal arm unless those diagnostics first reproduce and predict paired finite effects outside the fitting seeds.

## Primary interpretation

Primary endpoints remain separate:

- safety: `FN_at_TN68253` non-inferiority;
- efficacy: `TN_at_FN95` improvement;
- distribution: raw `FN=0..95` safety frontier;
- mechanism: difficult-normal and weak-defect trajectories;
- reliability: paired success, reversal, double degradation and worst-seed effect;
- efficiency: benefit per realized replay exposure and GPU hour.

`gap_q68_q050` is exploratory. No unvalidated weighted composite is a primary endpoint.

## Source map

The detailed paper-by-paper reasoning, equations, negative results, code audits and transfer limits are in `FULL_TEXT_EVIDENCE_SYNTHESIS.md` and `FULL_TEXT_EVIDENCE_NOTES/`.
