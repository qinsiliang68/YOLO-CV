# P008 - Deep Learning on a Data Diet: Finding Important Examples Early in Training

## Identity

- Paper ID: P008
- Authors: Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite
- Venue and year: NeurIPS 2021, volume 34
- Official landing page: https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html
- Main PDF: https://proceedings.neurips.cc/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf
- Supplement: https://proceedings.neurips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Supplemental.pdf
- Local main PDF: `source_papers/Data_Diet_2021.pdf`
- Main SHA256: `F5CABCA1BC3C837B924888BBAA949EBBC754B771FF91FCDA13E7B91D5C4DC9AD`
- Local supplement: `source_papers/Data_Diet_2021_supp.pdf`
- Supplement SHA256: `C09663005884829867847CCC701EFFFA3C253F41F27D03EBD839A6600373938B`
- Page count: main 12; supplement 10
- Code: https://github.com/mansheej/data_diet

## Reading Coverage

- Main PDF: 12/12 pages read, including theory, all main experiments, discussion and references.
- Supplement: 10/10 pages read, including Appendices A-G.
- Method: SGD setup, GraNd definition, one-step bound, EL2N approximation, relation to forgetting.
- Experiments: pruning curves, early score timing, architecture transfer, hyperparameter transfer, noise windows, NTK velocity and linear mode connectivity.
- Implementation: datasets, architectures, augmentation, optimizer, schedules, score/evaluation seeds and resource cost.
- Ablations: number of score runs, label-independent variants, architecture depth, cross-architecture scores, hyperparameter-search scores, noise level, pruning regime, memorization comparison.
- Limitations: loose upper bound, cancellation, score averaging requirement, coverage collapse, noise tail, validation-tuned window, accuracy-centric evaluation.
- Visual verification: main pages 4, 5, 6 and 8 plus supplement pages 2, 5 and 9 under `audit/visual_checks/P008_Data_Diet/`.

## Research Question

Can examples that matter for generalization be identified at initialization or early in training, without waiting for a full forgetting trajectory? What do gradient magnitude and output error actually measure, and when do their highest-score tails stop being useful?

## Method And Equations

For sample `(x, y)` at training time `t`, the paper defines GraNd as the expected per-sample loss-gradient norm over training randomness:

```text
GraNd_t(x, y) = E_{w_t} ||grad_{w_t} L(p(w_t, x), y)||_2
```

Lemma 2.2 shows that, for a fixed target example and one infinitesimal/one-step update, removing sample `j` changes the target example's instantaneous loss derivative by at most a constant times `||g_t(j)||`. The constant does not depend on the removed training sample, so small gradient norm implies a small upper bound on single-step influence.

The paper explicitly warns in footnote 4 that the converse is false: examples with large scores can have gradients that cancel and contribute little. Therefore:

```text
small expected gradient norm -> low one-step leverage bound
large gradient norm          -/-> positive or even large realized value
```

For cross-entropy classification, GraNd can be written using label-error components times logit Jacobians. Under the approximation that logit gradients are roughly orthogonal and similarly sized across classes/examples, the paper defines:

```text
EL2N_t(x, y) = E ||p(w_t, x) - y||_2
```

EL2N is an error-vector magnitude. It approximates gradient norm after a few epochs under stated assumptions; it contains no explicit target direction and no set-interaction term.

The expectation is important. The practical scores average over 10 independent initializations/training trajectories, not a single checkpoint.

## Experimental Contract

The main studies use CIFAR-10, CIFAR-100 and CINIC-10 with ResNet-18 and ResNet-50 variants. Data are normalized, padded by four pixels, randomly cropped to 32x32 and horizontally flipped with probability 0.5.

Training uses SGD, learning rate 0.1, Nesterov momentum 0.9, weight decay 0.0005, batch 128 for CIFAR and 256 for CINIC. Learning rate is divided by 5 after epochs 60, 120 and 160; all runs use 200 epochs. Crucially, pruned-dataset runs keep the same iteration count and schedule as full-data runs. Consequently, retained samples receive more repeated exposure, but omitted base samples disappear entirely.

All GraNd, EL2N and forgetting scores are averaged over 10 independent runs. Evaluation retrains from different random seeds, with four independent runs per plotted quantity; means and 16th-84th percentiles are shown. The supplement states that 10-20 score runs suffice empirically and more offer little additional benefit on these benchmarks.

The study consumed about 15,000 V100 GPU hours across exploration and final experiments. Code uses JAX/Flax.

## Main Results

- GraNd at initialization selects subsets better than random, but EL2N becomes more effective after a few epochs.
- EL2N at epoch 20 can prune about 50% of CIFAR-10 without degrading final test accuracy and about 25% of CIFAR-100 with little or no loss in the reported setting.
- Scores transfer across ResNet/VGG architectures and across a grid of learning rates and weight decays.
- Scores from one run are materially less effective than scores averaged over 10 runs. A single-run GraNd ranking has about 0.75 Spearman correlation with the 10-run average, yet its selected subset trains worse.
- High-EL2N subsets drive larger NTK-submatrix velocity and retain rougher loss-landscape barriers for longer, indicating prolonged feature learning.
- EL2N, GraNd and forgetting scores correlate, but supplied memorization values do not correlate with EL2N on the tested CIFAR-100 subset.

## Ablations And Failure Cases

### Large gradient is not sufficient

The theoretical result is an upper bound. The paper explicitly states that large gradients may cancel. No direction-to-validation-target test is present.

### Extreme score tails can be harmful

At high pruning, keeping only the highest EL2N/GraNd scores causes a sharp collapse from poor distribution coverage. A sliding-window experiment on CIFAR-10 performs best after excluding approximately 500 of the very highest EL2N examples in the stated setting.

### Noise reverses the meaning of high GraNd

With 10% randomized labels, corrupted samples move to the high EL2N tail. Selecting high GraNd at initialization can perform worse than a random subset. The highest-score tail contains blur, unusual angles/backgrounds, outliers and label noise as well as clean hard samples.

### Budget changes whether the high tail hurts

At a lower pruning level where 50% rather than 60% is removed, keeping high-score samples no longer produces the same clean-data penalty. Value is therefore conditional on retained budget and coverage, not just rank.

### Averaging is essential

Scores from one trajectory are noisier and select worse subsets. Averaging 10 independent runs removes model-specific variability. This directly limits single-OOF-model interpretations.

### Label and theory discrepancy

The supplement finds label-dependent GraNd outperforming a label-independent variant even at initialization, despite the theoretical initialization argument. The authors flag this as an unresolved discrepancy.

## What It Supports For Stage1

1. Gradient norm is a leverage screen, not a positive-value criterion.
2. Large-gradient candidates must be separated by target alignment, noise/learnability and interaction.
3. The very highest hardness/gradient tail should not automatically receive the largest replay dose.
4. Candidate ranking reliability depends on averaging across independent model trajectories.
5. Budget and coverage change the value of a ranked subset.
6. Output-error and parameter/feature velocity evolve over training; all-epoch trajectories are scientifically useful.
7. Repeated exposure of a retained subset can change optimization even when total step count is fixed.
8. Diversity/coverage is necessary at aggressive selection levels.

## What It Does Not Support

1. It studies pruning and retraining, not adding replay while preserving the full base pool.
2. It optimizes average test accuracy, not the Stage1 `FN <= 95` raw safety frontier.
3. It does not distinguish normal-tail benefit from weak-defect harm.
4. It does not show that any single gradient threshold is a necessary or sufficient condition for value.
5. It does not use gradient alignment to a validation or operational target.
6. It does not test the same fixed replay set across many training seeds with paired controls.
7. Its 50%/25% pruning findings are dataset-specific and cannot set Stage1 replay percentages.
8. Its score averaging uses 10 independent predictions per sample; one 10-fold OOF prediction per sample is not equivalent.

## Transfer Boundary

Stage1's existing 10-fold OOF trajectory gives each training sample one out-of-fold model trajectory, because each sample belongs to one held-out fold. It does not provide the paper's 10 independently initialized scores for the same sample. Thus the existing OOF table can reveal temporal behavior but cannot by itself reproduce the variance-reduced GraNd/EL2N ranking.

A feasible transfer is to compute multi-seed error/gradient statistics on a preregistered candidate and probe pool, rather than rerun all 120,000 samples through full gradients at every epoch.

The paper's retained-subset repetition is closer to concentrated exposure than ordinary one-pass pruning, but it still removes the rest of the base set. Any replay claim must be tested in our additive-replay contract.

## Concrete Field Requirements

At every epoch for replay samples and fixed probes:

- probability vector or binary raw logit and label error;
- EL2N/error magnitude;
- current loss and correctness transition;
- prediction/logit velocity and sign consistency;
- cumulative exposure and replay schedule;
- seed-specific value plus cross-seed mean, variance and rank stability;
- cluster/video identity and coverage counts;
- extreme-tail/noise-review flags.

At key checkpoints for a stratified candidate set:

- final-layer gradient norm;
- gradient dot product/cosine to normal-tail and weak-defect target gradients;
- cross-seed gradient-direction agreement;
- within-selected-set cancellation/conflict summaries.

## Concrete Experiment Consequence

- Do not use gradient magnitude alone as an arm claimed to be high value; use it as one prespecified candidate screen.
- Include a `Grad-Magnitude` negative/diagnostic control and a `Grad-Alignment` arm under the same budget if literature synthesis continues to support it.
- Estimate ranking stability across multiple unseen seeds on the candidate pool before freezing a selection.
- Avoid the extreme top tail or isolate it as a noise-risk stratum rather than silently mixing it with clean hard samples.
- Enforce coverage/diversity and video limits when selection is concentrated.
- In weekly reports, show score rank stability, gradient cancellation and tail trajectories, not only final TN/FN.

## Reproduction Notes And Missing Information

- Main and supplement provide the exact score definitions, architectures, datasets, augmentation, optimizer, schedule, score-run count, evaluation-run count and code repository.
- Score calculation averages 10 independent runs; evaluation uses distinct seeds.
- The paper preserves total training iterations for pruned datasets, an important exposure detail.
- Exact seed integers and every code-level preprocessing choice require repository inspection.
- The study does not report paired Stage1-like success probabilities or per-selection cross-seed sign reversals.
- `REPLICATION_DEPTH` denotes a complete reproduction plan, not an independent benchmark rerun.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, within the transfer boundaries above
- Direct support for gradient-magnitude-only replay: no
- Direct support for multi-seed candidate screening and noise/diversity controls: yes
- Reviewed at: 2026-08-07
