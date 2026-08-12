# P041 - Experience Replay for Continual Learning

## Identity

- Paper ID: P041
- Full title: Experience Replay for Continual Learning
- Authors: David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne
- Venue and year: NeurIPS 2019
- Official proceedings page: https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Main paper: `source_papers/Experience_Replay_NeurIPS_2019.pdf`, SHA256 `68E9F35D56F26E13F5F9FB7C3519C4CD5195B1FB9A0ABC7A734744F5674FED73`
- Official supplement: `source_papers/Experience_Replay_NeurIPS_2019_supplemental.zip`, SHA256 `9E28116B5D5A83617A534EBDA58FF800176C128F688FE2C912A850B780765F28`
- Official code: none linked by the paper or NeurIPS proceedings; a targeted author/title search found no author-released CLEAR implementation

## Reading Coverage

- Main paper: 11/11 pages read, including the complete method, equations, Figures 1-7, all experimental comparisons, discussion, and references.
- Supplement: 3/3 pages read, including distributed execution, buffer semantics, hyperparameters, evaluation, repeat count, and replotted Figures 8-12.
- Visual verification: all 14 main and supplement pages inspected at original detail under `audit/visual_checks/P041_ExperienceReplay_NeurIPS_2019/` and `audit/visual_checks/P041_ExperienceReplay_NeurIPS_2019_supplemental/`.
- Peer review: all five official reviews, the meta-review, and the one-page author response were checked. Reviewer concerns about task count, task balance, destructive interference, ratio choice, memory-size response, overlapping uncertainty, and missing mechanism interpretation are retained below rather than discarded.
- Reproduction boundary: no official CLEAR code or complete environment lock is supplied, so this paper is full-read evidence but not replication-depth evidence.

## Research Question

The paper asks whether replay can preserve old behavior while a reinforcement-learning agent continues to learn new tasks without explicit task-boundary labels. Its central distinction is a stability-plasticity tradeoff:

```text
stability  = retaining performance on previous tasks
plasticity = acquiring the current or newly introduced task
```

This is relevant to Stage1 because extra replay may protect one score region while impeding another. The task is nevertheless materially different: CLEAR replays old trajectories under sequential non-stationary RL tasks; Stage1 duplicates selected identities during one stationary supervised binary-classification run and evaluates an asymmetric fixed-FN tail.

## Method And Equations

CLEAR combines current on-policy experience and stored off-policy experience. V-Trace supplies truncated importance correction for the historical behavior policy. The policy-gradient, value, and entropy terms are applied to both current and replay samples. Replay samples additionally receive:

```text
L_policy_cloning = KL(historical_policy || current_policy)
L_value_cloning  = ||current_value - historical_value||^2 / 2
```

The cloning terms resist drift on stored experience. In the supplement their weights are `0.01` and `0.005`; policy-gradient, value, and entropy weights are `1`, `0.5`, and approximately `0.005`. These values describe the RL experiment only. They are not candidates for the Stage1 configuration.

Each distributed actor contributes one pair containing a new unroll and a replay unroll. The learner chooses from these pairs according to the requested new/replay ratio, which also limits any actor to one batch element. A bounded buffer is maintained by reservoir sampling, so stored unrolls are intended to be a uniform sample of all past unrolls.

## Experimental Contract

- Domains: three cyclic DMLab tasks, a novel DMLab probe task inserted at different points, and six sequential Atari games.
- Learner: asynchronous IMPALA/V-Trace, with architectures and most hyperparameters copied from the referenced IMPALA or Progress-and-Compress setup.
- Main replay mixtures: 75/25 new/replay, 50/50, and 100% replay. The paper normally uses 50/50, while the Atari comparator uses 75/25.
- Buffers: 450M, 50M, and 5M frames in a 900M-frame sequence; the smallest is approximately 0.5% of past experience at the end.
- Baseline: no replay buffer with the remaining network and training parameters held constant.
- Repeats: each experiment was independently run three times; plots show means and standard-deviation error bars.
- Evaluation: separate testing actors evaluate every task throughout training; final tables use cumulative reward, effectively area under each task's learning curve.
- Tuning: the supplement says no significant effort was made to optimize CLEAR fully.

## Main Results

For the three-task DMLab comparison, final cumulative values are:

```text
Separate                         29.24   8.79  19.91
Simultaneous                     32.35   8.81  20.56
Sequential without CLEAR         17.99   5.01  10.87
CLEAR 50/50                      31.40   8.00  18.13
CLEAR without behavioral cloning 28.66   7.79  16.63
CLEAR 75/25                      30.28   7.83  17.86
CLEAR 100% replay                31.09   7.48  13.39
CLEAR buffer 5M                  30.33   8.00  18.07
CLEAR buffer 50M                 30.82   7.99  18.21
```

Replay greatly reduces forgetting in this setting, and behavioral cloning improves aggregate stability. Pure replay is especially stable but learns a later probe task more slowly as that task occupies a smaller fraction of the growing buffer. The authors therefore interpret 50/50 as a useful tradeoff, not as a theorem or a ratio that dominates every task. Their response acknowledges that 75/25 was slightly better on Atari.

The 5M reservoir buffer remains competitive but shows slightly more forgetting. The paper's concrete explanation is disproportionate repeated training on a small identity set, which may overfit those stored examples. This is direct evidence that nominal replay ratio and unique-set size are insufficient: realized per-identity exposure concentration matters.

## Ablations, Negative Results, And Review Challenges

1. Removing behavioral cloning weakens stability but does not make replay useless. The mechanism is therefore multi-component.
2. The 75/25 mixture reduces most but not all forgetting. Pure replay protects old behavior but lowers overall and early new-task performance.
3. Ratio winners differ by domain and task; 50/50 is not shown to be universally optimal.
4. The main experiments rely heavily on three cyclic, equally exposed tasks. Reviewer 3 correctly notes that disproportionate task durations, more tasks, non-cyclic sequences, and destructive interference remain under-tested.
5. Only three independent runs are used. Several error bars overlap, and there is no paired-seed analysis, confidence interval on differences, multiplicity adjustment, worst-seed result, or success-probability estimate.
6. The authors deliberately choose tasks with little destructive interference. The study therefore cannot show that preserving old behavior is harmless when old and new objectives conflict.
7. The discussion explicitly places continual-learning methods on a Pareto frontier and concedes that protecting obsolete behavior can be harmful; selective forgetting may be required.
8. The meta-review requested sensitivity to sample complexity and V-Trace truncation coefficients. Those questions are not resolved strongly enough to import method ratios or coefficients.

## Direct Support For Stage1

1. Replay has at least two endpoints that must be reported separately. Stability on a protected tail and plasticity on another tail can move in opposite directions.
2. Replay ratio, cumulative replay dose, and per-identity repetition concentration are distinct process variables.
3. A small fixed replay set can be overexposed even if the buffer is representative. Record unique identities, occurrence counts, concentration, and effective sample size, not only the configured percentage.
4. A replay policy can sit on a Pareto frontier rather than maximizing one universal score. Stage1 must preserve difficult-normal benefit and weak-defect harm as separate constraints.
5. Same-selection comparisons need paired seeds and full trajectories. A three-run mean cannot establish cross-seed stability.
6. No-replay is a necessary baseline because replay itself, not just selection quality, may cause the observed benefit or damage.

## What The Paper Does Not Support

1. It does not support a 50/50, 75/25, 0.5%, or any other numeric Stage1 replay setting.
2. It does not support changing Stage1 batch size, optimizer, learning rate, augmentation, worker count, precision, or any canonical 240-run hyperparameter.
3. It does not test replay stopping or decay at epochs 140-160, cumulative-dose matching, a fixed-FN safety frontier, or weak-defect guard selection.
4. It does not identify valuable individual images or distinguish clean hard samples from label noise.
5. V-Trace off-policy correction and behavioral cloning do not transfer to ordinary supervised duplicate replay.
6. It does not explain Stage1's same-selection cross-seed sign reversals and supplies no gradient-direction evidence.

## Transfer Boundary And Observable Consequences

The paper strengthens a process-level hypothesis without adding a formal arm:

```text
hypothesis:
repeated exposure can stabilize the directly replayed region while reducing
plasticity or increasing overfit elsewhere; the effect depends on replay share,
identity concentration, task/model state, and training time.

Stage1 falsifiable consequence:
under identical canonical hyperparameters, initialization, base data, and
selection, changing only replay timing or cumulative dose may change the joint
difficult-normal / weak-defect trajectory and cross-seed failure rate.
```

If continuous replay, same-peak decay, and dose-matched decay have indistinguishable paired outcomes, this mechanism does not explain Stage1 reversals at the tested ratios. If decay reduces weak-defect harm while preserving normal-tail benefit, late exposure is implicated. If only the lower-dose schedule helps, cumulative dose is more plausible than timing.

## Concrete Field Requirements

Record every epoch when inexpensive:

- configured and realized replay fraction;
- integer replay slots, base examples, optimizer examples, and optimizer steps;
- identity-level occurrence counts, first/last exposure, inter-arrival gap, and cumulative exposure;
- unique replay count, maximum share, Herfindahl concentration, entropy, and effective identity count;
- schedule phase and cumulative dose at the current epoch;
- difficult-normal and weak-defect trajectories separately;
- learning rate, actual update norm, loss components, and replay/base batch composition;
- seed, initial-weight hash, base-order hash, augmentation state, sampler/RNG state, machine, resume lineage, and failure state.

At preregistered checkpoints, collect separate gradient dot products and cosines against difficult-normal and weak-defect probe gradients. Do not collapse both directions into a single alignment average.

## Decision

- Reading status: FULL_READ_COMPLETE
- New formal training arm: no
- New Stage1 hyperparameter: no
- Canonical lock change: no
- Added evidence: replay stability, plasticity, dose, identity concentration, and task/model state form a conditional tradeoff rather than a static sample-value scalar
- Remaining uncertainty: whether Stage1 late replay creates the same kind of overexposure harm under the exact 240-run canonical training contract
