# P022 - A Closer Look at Memorization in Deep Networks

## Identity

- Paper ID: P022
- Authors: Devansh Arpit, Stanislaw Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio and Simon Lacoste-Julien
- Venue and year: ICML 2017, PMLR 70
- Official page: https://proceedings.mlr.press/v70/arpit17a.html
- Main paper: `source_papers/A_Closer_Look_at_Memorization_ICML_2017.pdf`, SHA256 `1146F2F3D140ABA0270300EF6954EADE2ABFB399C1E461C955A810F2CBE9A68B`
- Public experiment code: none linked by the official proceedings or paper

## Reading Coverage

- Main paper: 10/10 pages read, including all methods, experiments, figures, limitations expressed in the discussion and references.
- Supplement: none published on the official PMLR entry.
- Peer review: not publicly linked by the official ICML/PMLR entry.
- Code: the paper states that experiments used Theano and Keras but provides no experiment repository or environment lock.
- Visual verification: all 10 pages and three contact sheets under `audit/visual_checks/P022_Memorization_ICML_2017/`.

## Research Question

When an over-parameterized network can fit both real and random data, does gradient-based training use the same brute-force strategy, or does it learn shared simple patterns before memorizing idiosyncratic noise?

The Stage1 relevance is mechanistic. Replayed difficult normals might first provide reusable residual corrections and later create increasingly local decision structure or weak-defect interference. The paper does not study replay, operational tails or the same selected IDs across seeds.

## Core Definitions

The authors define the effective capacity of a complete learning algorithm, not only an architecture:

```text
EC(A) = {h | there exists D such that h is reachable by A(D)}.
```

Here `A` includes model, optimizer and training procedure. This is important for Stage1: changing batch size, schedule or regularization changes the scientific object even if the architecture remains `yolo11l`.

The paper's loss-sensitivity proxy differentiates a future average loss through an unrolled sequence of SGD updates with respect to an earlier training example:

```text
g_x^t = partial L_t / partial x
```

and averages its norm over later steps. This is not the ordinary per-sample parameter-gradient norm. It includes the indirect effect of the example on subsequent updates, but was demonstrated only on a two-layer 16-unit network and 1,000 downscaled MNIST examples.

To estimate learned decision-surface complexity, a sample `x` is critical when a nearby point within an `L_infinity` box changes the predicted class. The Critical Sample Ratio is

```text
CSR = number of critical samples / number of evaluated samples.
```

The nearby point is searched using Langevin Adversarial Sample Search, a noisy FGSM-like iterative procedure with clipping to radius `r`. CSR is therefore a search-dependent proxy, not an exact count of decision regions.

## Experimental Evidence

- The main datasets are MNIST and CIFAR-10. MNIST uses two-layer ReLU MLPs, often with 4,096 units per layer and 1,000 epochs of SGD at learning rate `0.01`. CIFAR-10 uses a small AlexNet-style CNN, momentum `0.9`, learning rate `0.01` and a half-rate drop every 15 epochs.
- Random-input and random-label datasets are synthetic counterfactuals. They are useful for identifying qualitative differences, but do not establish that naturally hard examples are noise.
- In the one-epoch difficulty experiment, each condition is trained from 100 random initializations and data shuffles. Real examples show a broad, identity-specific correctness frequency; random-input examples look close to binomial variation. This supports a cross-seed learnability field rather than a one-run difficulty score.
- Loss-sensitivity is concentrated on a subset for real data and broadly high for random data. Its Gini coefficient separates those aggregate patterns in the small unrolled experiment.
- Time to reach 100% training accuracy grows much faster with dataset size for random data than for real data. Additional real examples provide shared clues; random examples require more identity-specific fitting.
- CSR rises with training and is higher for random inputs or labels. In mixed random-label experiments, validation accuracy peaks before high training accuracy is reached, while CSR rises as the model fits the random labels. This is the paper's strongest evidence for learning shared patterns before memorization.
- Noise examples can interfere with real-data validation, but larger capacity can sometimes fit noise in a way that interferes less. Low model capacity is therefore not a universally safer response.
- Dropout, input/hidden noise, weight decay and adversarial training affect random-label fitting and clean validation differently. Dropout, especially with adversarial training, appears strongest in the authors' sweep at slowing memorization while preserving clean validation.

## Evidence Limitations

1. Most conclusions are qualitative curve comparisons. The paper does not report per-seed outcome tables, confidence intervals or formal tests for the central CSR and mixed-noise experiments.
2. Only the one-epoch ease experiment explicitly states 100 initializations. Replicate counts for the remaining figures are not specified.
3. Random inputs and random labels are controlled extremes, not audited natural label errors, ambiguous video frames or targeted replay samples.
4. Validation maxima are used repeatedly. The regularization plot chooses the best clean-validation result across parameter settings, so it demonstrates attainable tuning behavior rather than one frozen causal configuration.
5. CSR depends on adversarial search radius, iteration budget, step size, injected noise and search success. The paper says several radii were qualitatively similar but gives no estimator uncertainty or failure-rate analysis.
6. The loss-sensitivity unroll is computationally expensive and demonstrated on a tiny network. It cannot justify 120,000-sample full-model all-epoch unrolling.
7. No official experiment code, exact seeds, dependency lock, raw results or appendix are available from the proceedings entry.
8. The work predates modern large vision classifiers and does not study momentum-state attribution, mixed mini-batch replay, augmentation identity or finite repeated exposure.

## Direct Support For Stage1

1. The effective unit of comparison is the full learning algorithm. Exact canonical hyperparameter locking is scientifically necessary, not just an engineering preference.
2. Sample ease should be estimated across states or seeds because real examples exhibit identity-specific learnability beyond random initialization noise.
3. Generalizable pattern learning and local memorization can occur at different training stages, so all-epoch trajectories are more informative than endpoint scores.
4. Aggregate complexity or influence concentration can increase while validation degrades, motivating process fields that track both difficult-normal benefit and weak-defect harm.
5. A future-loss sensitivity through updates is conceptually closer to value than immediate gradient norm, but requires low-cost approximations and finite-intervention calibration.
6. Data content, model capacity, optimizer and regularization interact. A sample's effect cannot be represented as a context-free scalar.

## What It Does Not Support

1. A universal epoch at which memorization starts, including epochs 140, 150 or 160.
2. Treating every persistently difficult Stage1 sample as noise or every easy sample as valuable.
3. A positive replay ranking based on CSR, loss-sensitivity, gradient norm or one-epoch correctness frequency.
4. Changing model capacity, dropout, adversarial training, batch size, optimizer or augmentation in the formal campaign.
5. Any replay percentage, cumulative dose, decay duration or weak-defect guard ratio.
6. Assuming that stopping late replay is beneficial without a continuous-versus-decayed, dose-controlled Stage1 intervention.
7. Direct inference about the raw `FN=0-95` safety frontier from average validation accuracy on MNIST or CIFAR-10.

## Stage1 Field Contract

Persist inexpensive fields for all 200 epochs:

- per-identity correctness, assigned-label margin, loss and confidence;
- first-learned epoch, learned-to-forgotten and forgotten-to-relearned transitions;
- cross-seed or cross-fold correctness frequency and transition-time variability;
- base, replay-normal, hard-normal and weak-defect group trajectories;
- training and validation tail behavior rather than only average loss;
- realized replay occurrence, cumulative exposure, repeat lag and augmentation-view identity;
- model/optimizer configuration hash, learning rate, optimizer step and schedule phase.

At selected checkpoints, add low-cost decision-complexity and sensitivity probes:

```text
local_margin_to_boundary proxy
input-gradient or adversarial-radius proxy on a fixed probe set
replay-gradient concentration and cancellation
same-state continuation versus finite replay delta
```

These proxies require estimator parameters and failure counts. They must not modify the canonical training stream.

## Concrete Experiment Consequence

P022 supports a transition hypothesis, not a cutoff. For the same selected IDs and seed under the exact canonical 240-run lock, compare no replay, continuous replay, same-peak decay and cumulative-dose-matched relocation. Test whether any late harm is accompanied by:

```text
base patterns already learned
replay-normal influence becoming more concentrated
local boundary complexity rising around replay identities
difficult-normal scores improving
weak-defect margins declining
```

If decay wins without this sequence, the memorization explanation is not established. If the sequence occurs without outcome improvement, it is descriptive rather than decision-sufficient. Regularization changes remain out of scope because they would violate canonical comparability.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for data-dependent memorization, cross-seed learnability, stage-dependent complexity and canonical algorithm identity
- Replication-depth eligibility: no, because no official code, detailed appendix or complete stochastic protocol is available
- Direct support for static replay ranking: no
- Direct support for numeric Stage1 schedule boundaries or percentages: no
- Reviewed at: 2026-08-07
