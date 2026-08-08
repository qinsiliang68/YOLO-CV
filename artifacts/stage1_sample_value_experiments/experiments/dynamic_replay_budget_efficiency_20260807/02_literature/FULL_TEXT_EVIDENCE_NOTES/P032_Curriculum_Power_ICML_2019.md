# P032 - On the Power of Curriculum Learning in Training Deep Networks

## Identity

- Paper ID: P032
- Authors: Guy Hacohen and Daphna Weinshall
- Venue and year: ICML 2019, PMLR 97:2535-2544
- Published page: https://proceedings.mlr.press/v97/hacohen19a.html
- Main PDF: `source_papers/Curriculum_Power_2019.pdf`, SHA256 `78326719D8F9AB5B601615BE3F7C7A69B3B523B18258DDF13FC3D8AB4A7F2146`
- Supplement: `source_papers/Curriculum_Power_2019_supp.pdf`, SHA256 `19220A5C03990BE1845F7F006EB68776DA2B661A8EC0429362A8A307057FE8A4`
- Official code: https://github.com/GuyHacohen/curriculum_learning
- Audited code commit: `53691689c85f42f4221001edc5627771d0751908`

## Reading Coverage

- Main manuscript: 10/10 pages read, including the scoring/pacing definitions, Algorithm 1, all six image-classification cases, transfer and bootstrapped curricula, alternative pacing, gradient analysis, theoretical propositions, discussion, and references.
- Supplement: 4/4 pages read, including robustness checks, varied pacing, AUC-based model selection, repeated bootstrapping failure, learning-rate confounding, grid-search ranges, ImageNet subset details, and the proof of Proposition 2.
- Visual verification: all 14 unique pages inspected at original detail under `audit/visual_checks/P032_Curriculum_Power_ICML_2019/`.
- Code audit: the official repository HEAD was inspected and its relevant sources passed Python syntax compilation. No paper tag, environment lock, automated tests, or executable paper-wide reproduction contract was found.

## Research Question

The paper asks when curriculum learning can help deep image classifiers and separates a curriculum into two components:

```text
scoring function f(x)  -> a ranking or prior over example difficulty
pacing function g(t)   -> how much ranked data is visible at training time t
```

That separation is directly relevant to Stage1. A fixed replay selection and a replay schedule are different interventions. Their effects cannot be identified if selection, exposure, learning rate, optimizer steps, or other training settings change together.

The paper studies ordinary top-1 image classification and early easy-to-hard exposure. It does not study repeated replay on top of a fixed base stream, a weak-defect safety constraint, raw `FN=0-95` frontiers, or cross-seed reversal of one fixed image set.

## Core Formulation

At training step `t`, the scoring function orders the examples and the pacing function exposes only the first `g(t)` examples. Sampling within the visible subset is class balanced. The controls include vanilla training, anti-curriculum, and a random ranking with the same pacing.

The main scoring rules are:

```text
transfer scoring:
  pretrained teacher features -> SVM confidence -> fixed ranking

self-taught scoring:
  train a vanilla target model -> rank by its target-task confidence

self-paced scoring:
  repeatedly use the current learner's confidence
```

The principal pacing rules are fixed exponential growth, varied exponential growth, and a single-step policy that first trains on a small easy subset and then exposes all data.

The theory rewrites the curriculum-weighted utility as the ordinary utility plus a covariance term. For an induced prior `p`:

```text
U_p(theta) = U(theta) + Cov(U_theta, p)
```

For the paper's ideal prior, the covariance is proportional to covariance with the utility under an unknown optimal hypothesis. The same optimum and a steeper utility landscape follow only after assuming that the original optimum also maximizes this covariance. The constant-utility-variance corollary is another strong condition. These are explanatory sufficient conditions, not a general guarantee that easy-first training or any Stage1 replay schedule improves the desired tail objective.

## Experimental Protocol

- Six image-classification cases cover CIFAR-10, CIFAR-100, selected CIFAR-100 superclasses, and a seven-class ImageNet cat subset.
- The moderate network uses eight convolutional layers, a 512-unit fully connected layer, dropout, SGD, cross-entropy, and batch size 100. Cases using public VGG-style networks follow different architecture-specific settings.
- Reported repetition counts vary substantially: 50 for the main small-mammals case, 25 for some CIFAR settings, five for some transfer-scoring variants, and three for the ImageNet subset.
- Curriculum subsets are class balanced. Anti-curriculum and random-order controls share the pacing family but their hyperparameters are separately tuned.
- The main model-selection target is final accuracy in the original plots; the supplement repeats the comparison with area under the learning curve and changes which curriculum variant appears best while retaining the broad qualitative result.
- For VGG cases 4 and 5, the paper reports no data augmentation.
- The first-mini-batch gradient analysis uses one arbitrary 10% visible subset, corresponding to 250 examples in case 1. It compares distances between mean gradients and total gradient variance; it is not a repeated per-sample influence analysis.
- All architecture, optimizer, learning-rate, batch, epoch, pacing, and search-range values are literature context only. None may change the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. Separating scoring from pacing is a strong design contribution. A ranking can remain fixed while its presentation schedule changes, which matches the Stage1 timing question.
2. Single-step pacing performs competitively in the main case, and the authors infer that much of the curriculum effect occurs early. This motivates a timing intervention but does not identify epoch 140, replay decay, or late-replay harm.
3. Transfer and self-taught target-hypothesis scores help in several settings, while self-paced current-hypothesis scoring reduces accuracy throughout the reported comparison. A dynamic score is not automatically a useful score.
4. Repeated self-taught bootstrapping accumulates ranking errors and eventually harms performance. Iterating a proxy can reinforce its mistakes.
5. Curriculum gains are larger on harder reported tasks, but the evidence spans different datasets, architectures, repeat counts, and tuning procedures. This is not a universal monotonic difficulty law.
6. The gradient means under three transfer rankings differ from the all-data and random-subset gradients and have lower total variance in one first-batch analysis. The subset is arbitrary, the sample count is small, and there is no paired downstream attribution from these statistics.
7. The paper explicitly recognizes that changing visible dataset size changes the effective learning-rate behavior. Its varied pacing adjusts stage lengths to compensate, and the supplement states that learning-rate tuning is needed for a fair comparison.
8. Different conditions are separately tuned. This may optimize each method, but it prevents a clean causal interpretation of scoring or pacing alone when learning-rate schedules differ.
9. Selecting by final accuracy and selecting by learning-curve AUC produce similar broad conclusions but a different best curriculum. The endpoint definition matters and must be preregistered.
10. Theoretical propositions depend on an unknown ideal curriculum and covariance assumptions tied to the optimum. They do not establish that a practical confidence ranking satisfies those assumptions.
11. Repetition counts as low as three or five, standard-error bars without paired seed trajectories in several settings, and no worst-seed analysis are insufficient evidence for Stage1 cross-seed reliability.
12. The paper has no no-replay analogue because its task is curriculum ordering within ordinary training. It does not separate replay from no replay on a fixed base stream.

## Official Code Audit

The audited repository HEAD is a clean post-paper commit dated 2019-08-19. There is no tag identifying an exact paper snapshot. The README says the public code reproduces only basic transfer-scoring and fixed-exponential-pacing results.

Important implementation findings:

1. `main_reproduce_paper.py` gives curriculum, vanilla, anti-curriculum, and random conditions different initial learning rates and decay schedules. The executable comparison is therefore not a pure ranking-or-pacing intervention.
2. A cross-validation update method exists in `datasets/Dataset.py`, but no runner calls it. The published repository does not implement the cross-validation selection process described in the paper.
3. NumPy sampling, shuffling, and SVM probability calibration are not tied to a persisted random seed. No complete Python, NumPy, framework, CUDA, or data-order RNG state is recorded.
4. `requirements.txt` contains only lower bounds and omits a TensorFlow version, so the legacy Keras execution environment is not locked.
5. No automated tests were found. Relevant sources pass syntax compilation, which does not establish numerical reproduction.
6. The runner evaluates the test set every 50 batches and reports final test behavior. There is no blind-holdout role contract.
7. Output consists mainly of an averaged combined-history pickle. Per-repeat histories, model checkpoints, optimizer state, RNG state, resume state, atomic writes, and integrity sidecars are absent.
8. Downloaded and cached assets are not content-hashed.
9. The public runner exposes some arguments that are ignored or overridden, including hard-coded optimizer behavior in the training path. SGD momentum falls back to the compilation helper's default when the caller omits it.
10. A mini-batch samples without replacement from the current visible subset, but there is no epoch-level coverage guarantee; identities can be repeatedly selected while others remain unseen.
11. The repository covers only part of the paper's conditions and cases. It does not reproduce the varied pacing, single-step, bootstrapping, theory, all architectures, or all robustness analyses as one audited workflow.

## Direct Support For Stage1

1. Keep sample selection and replay timing as separate manifest fields and separate causal factors.
2. For every epoch, record the intended replay ratio, realized replay slots, unique replay identities, cumulative presentations, cumulative unique coverage, and exposure concentration.
3. Record the effective number of optimizer updates and examples processed, because equal epochs with different visible/replayed data do not imply equal optimization exposure.
4. Compare continuous replay, same-peak decay, and cumulative-dose-matched decay on the same frozen selection, seed, initialization, base stream, optimizer, and canonical hyperparameters.
5. Use no replay to determine whether replay itself helps, then use random and matched-random controls to determine whether the selected identities matter.
6. Keep all arm hyperparameters identical. Do not retune learning rate or other canonical settings per replay schedule.
7. Record early, middle, and late tail outcomes and last-head gradient summaries so a timing effect can be distinguished from an endpoint-only fluctuation.
8. Preserve the schedule state, sampler state, RNG state, current visible/replay pool identity, and cumulative exposure in checkpoints so resume reproduces the same path.
9. Preregister the primary endpoint and report both endpoint and trajectory summaries; do not choose after seeing which one favors the method.
10. Track score age and ranking source. A score learned from a previous model and a score from the current model are different interventions.

## What It Does Not Support

1. Claiming that early replay or stopping replay at epoch 160 is already proven.
2. Importing the paper's batch size, optimizer, learning-rate values, decay intervals, architecture, augmentation choices, pacing percentages, or epoch counts.
3. Treating easy-first, hard-first, transfer confidence, self-taught confidence, or self-paced confidence as a universal sample-value rule.
4. Inferring weak-defect protection, `TN_at_FN95` improvement, or raw-frontier dominance from top-1 accuracy.
5. Calling one arbitrary first-batch gradient mean or variance a per-sample causal value estimate.
6. Retuning canonical training hyperparameters separately for Stage1 arms.
7. Treating a fixed selection's effect as seed-stable without paired unseen-seed confirmation.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain:

- immutable scoring-rule ID, ranking artifact SHA256, score-producing model/checkpoint hash, and score age;
- immutable pacing/schedule ID, schedule parameters, current intended ratio, and schedule-state hash;
- per-epoch intended and realized replay slots, unique identities, duplicate fraction, cumulative presentations, cumulative unique coverage, exposure entropy, and concentration;
- base examples, replay examples, optimizer steps, skipped steps, and effective examples processed;
- all-epoch normal-tail and weak-defect probe loss, scores, membership, and trajectory changes;
- key-checkpoint last-head gradient norm, normal-tail alignment, weak-defect alignment, variance, and cross-checkpoint sign consistency;
- per-arm canonical-lock hash, initialization hash, dataset/split hash, sampler/RNG hash, checkpoint/resume lineage, and exact role identities;
- endpoint and trajectory definitions fixed before result access;
- machine, wall time, train compute, DataLoader wait, evaluation, checkpoint write, and true idle time kept separate from scientific outcomes.

None of these fields changes `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate schedule, augmentation, or any other canonical setting.

## Concrete Experiment Consequence

P032 adds no new formal arm and does not select a replay percentage. It strengthens the interpretation of the existing minimal timing block:

```text
continuous vs same-peak decay
    -> timing and cumulative dose both differ

continuous vs cumulative-dose-matched decay
    -> cumulative dose is held fixed; exposure timing differs

same-peak decay vs cumulative-dose-matched decay
    -> timing window is shared; cumulative dose differs
```

All three comparisons require the same selection, seed, base stream, optimizer, canonical lock, total 200-epoch training horizon, and primary endpoint. If arm-specific learning-rate tuning is introduced, the timing claim becomes confounded and the block must fail preflight.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for scoring/pacing separation, learning-rate and exposure confounding, early curriculum effects, proxy error accumulation, and mixed scoring results
- Replication-depth eligibility: no; the official code is partial, post-paper, untagged, environment-unlocked, and lacks tests and a complete reproduction workflow
- Direct support for static replay ranking: no
- Direct support for the existing timing/dose causal decomposition: yes, as a design requirement rather than an efficacy result
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-08
