# Stage1 Full-Text Evidence Synthesis

This file remains provisional until the preregistered full-text corpus has been completed and audited.

## Evidence Checkpoint 1

P002 (DoCL, AISTATS 2021) provides primary evidence that a useful training sample cannot be reduced to loss or gradient magnitude. Its score combines label residual with the direction and speed of output change induced by the current data distribution; its NTK form also depends on relations to other residual-bearing samples. The score is time-varying and contextual.

For Stage1 this supports all-epoch trajectory capture, dynamic/contextual value hypotheses, exploration, and a causal comparison of replay timing. It does not support any specific Stage1 replay percentage, a fixed stop epoch, weak-defect guard efficacy, or a claim that its subset curriculum transfers directly to duplicate replay. See `notes/P002_DoCL_2021.md`.

The expert schedule proposal remains a candidate design. No numeric arm or weekly run count is frozen from this single paper.

## Evidence Checkpoint 2

P009 (RHO-LOSS, ICML 2022) separates current underlearning from irreducible difficulty using `current loss - frozen reference loss`. Controlled noise experiments show why high loss and high gradient norm can select corrupted points. The paper also shows that dynamic updating of a reference model on biased selected data can deteriorate late in training.

For Stage1 this supports a frozen cross-fitted learnability/reference field and reinforces the need to capture current loss every epoch. It also strengthens weak-defect protection: the paper's own ethics analysis notes that rare groups may be deprioritized when they contribute less to average holdout loss. Its 5-20% prebatch selection ablation is not a replay-budget result and supplies no numeric Stage1 ratio.

Combined P002/P009 evidence now supports three design statements:

1. static hardness is insufficient;
2. value must be conditioned on current model state and learnability/reference context;
3. timing/dose and tail protection require our own causal arms because neither paper studies duplicate replay under an FN constraint.

## Evidence Checkpoint 3

P008 (Data Diet, NeurIPS 2021) gives the precise limitation needed for the gradient route. Small expected gradient norm bounds one-step leverage, but the paper explicitly says the converse is false because large gradients may cancel. Its experiments also show that extreme high-score tails can contain noise/outliers and that high-GraNd selection can fall below random after label corruption.

The paper averages scores across 10 independent runs and shows single-run subsets are worse. Stage1's current OOF trajectory supplies one held-out trajectory per training sample, not ten independently initialized trajectories for the same sample. Therefore a multi-seed candidate/probe measurement is a real field gap, while all-120k all-model full-gradient capture remains unnecessary.

The first three papers jointly reject a scalar based only on one-shot confidence, loss, EL2N or gradient magnitude. They support a staged decomposition into leverage, learnability/noise, target direction, temporal exposure and set coverage.

## Evidence Checkpoint 4

P004 (Learning to Reweight Examples, ICML 2018) supplies the direct first-order direction test. Differentiating a one-step virtual training update gives an example weight proportional to the positive part of `g_validation^T g_i`. A large sample gradient pointing against the current target receives zero weight. The paper therefore supports target-gradient alignment as a mechanism diagnostic, while also proving that value is conditional on the current model, current validation objective and current mini-batch.

Its limits are equally important. The method optimizes ordinary clean validation loss, costs about three times normal training, normalizes weights within each mini-batch, and provides only a local smooth-objective result under bounded gradients and small steps. It does not study additive replay, raw `FN <= 95` metrics, fixed-set cross-seed reversals or final-layer approximations. The paper also reports a strong random-weight baseline and a clean-data penalty caused by validation-subset bias.

Combined P002/P004/P008/P009 evidence now supports an observational gradient-probe channel, not an immediate replacement of the frozen Stage1 optimizer. The probe objective must be non-test, cross-fitted and decomposed into normal-tail benefit and weak-defect protection. Alignment sign, temporal/seed stability and set-level cancellation must remain separate fields until causal evidence justifies any aggregation.

## Evidence Checkpoint 5

P005 (GradMatch, ICML 2021) moves from independent sample alignment to a weighted-set gradient residual. OMP repeatedly selects the atom that explains the current residual and refits all selected weights, so complementarity, cancellation and redundancy are part of the objective. The paper also makes timing explicit: the subset is refreshed every `R` epochs, warm start changes quality, low-budget variance is higher, and weight regularization prevents excessive concentration.

For Stage1, the important transfer is an inference rather than a copied algorithm. Because the complete 120k base pool remains present, replay should be evaluated as a correction to the base gradient:

```text
residual_before = ||g_target - a g_base||
residual_after  = ||g_target - (a g_base + b g_replay)||
```

This explains why a sample can have positive individual target alignment yet add no new value, and why repeated exposure can eventually overshoot. The supplement's finite-step condition also shows that positive cosine alone is insufficient: target-gradient magnitude, aggregate update norm, learning rate and curvature determine whether the actual step lowers the target loss.

The paper does not justify a numeric Stage1 budget, reselection interval or stop epoch. It trains on a replacing subset, uses convex theory, and does not test fixed selections across seeds. Its supplementary SGD proof also contains a residual-sign inconsistency in equations 64-69, while the main theorem uses the expected positive penalty. The usable result is therefore a measurable mechanism hypothesis, not an imported guarantee.

The current five-paper synthesis expands the field contract to include base/replay/combined gradient norms, separate normal-tail and weak-defect residual correction, set cancellation, concentration, finite-step probe change and last-layer versus full-network agreement.

## Evidence Checkpoint 6

P006 (GLISTER, AAAI 2021) adds a validation-targeted, state-dependent set marginal. Its online approximation rebuilds a subset every `L` epochs and recomputes the validation gradient as the greedy set grows. The candidate score is therefore conditional on the current model and on samples already selected. This independently supports the rejection of a permanent per-image scalar.

Its descent theorem reinforces the finite-step limitation from P005: positive validation-gradient alignment is insufficient unless the learning rate and aggregate selected-gradient norm also satisfy a smoothness bound. The appendix further reports that low Taylor recomputation count `r` is unstable, while stale selection through larger `L` can reduce accuracy. This supports measuring target staleness, update norm and refresh age. It does not establish any Stage1 stop epoch or replay percentage.

The paper's use of one held-out likelihood is also a warning. The authors add diversity/random regularization to reduce validation overfitting, and their robustness experiments assume a clean or balanced validation set. Stage1 must not collapse difficult-normal benefit and weak-defect preservation into one unvalidated average target. The two target axes remain separate until a preregistered non-inferiority rule admits candidates.

The public code audit materially narrows transfer. Its deep script repeatedly fixes seed 42, locally sets `num_runs=1`, and does not consume the launcher-provided run-count/warm arguments. The PDFs report no seed uncertainty or paired same-selection replications. GLISTER therefore supports dynamic target-aligned mechanism probes, but supplies no evidence that one selected set is stable across initialization seeds.

The current six-paper synthesis now requires checkpoint-specific set marginal gain, refresh age, finite-step descent margin, separate tail targets and cross-seed sign stability. Canonical Stage1 optimizer, augmentation, batch and model settings must remain locked so replay timing and dose are the only causal interventions.

## Evidence Checkpoint 7

P007 (Example Forgetting, ICLR 2019) supplies the first direct multi-seed training-dynamics result. A forgetting event is a correct-to-incorrect transition between consecutive mini-batch presentations of the same training sample. Counts are reasonably stable across seeds, especially after aggregation, and low-forgetting examples can be removed much more safely than random examples.

The hard end is not a value ranking. Synthetic label and pixel corruption both increase forgetting, and the paper's own removal curve turns upward at the extreme tail, which the authors attribute to outliers or mislabeled samples. The usable decomposition is therefore stable easy versus unstable candidate-hard versus never-learned/noise-risk. High forgetting alone cannot define Treatment.

The measurement contract is also narrower than common summaries imply. The estimator observes stochastic augmented training presentations and is a lower bound on all parameter-step transitions. Stage1's existing 10-fold OOF checkpoint predictions are held-out fixed-view trajectories. Their correct-to-incorrect transitions must be called OOF epoch correctness reversals, not training forgetting. Agreement between the two is a new empirical field, not an identity.

The paper's strongest stability evidence uses ten to one hundred seeds, while the central removal curves use five. This supports multi-seed trajectory aggregation and warns that one OOF trajectory per sample is insufficient for a stable hard-tail rank. The final collection schema must preserve presentation occurrence, replay identity, cumulative exposure, fixed-view checkpoint reversal and stochastic augmentation context as distinct dimensions.

## Evidence Checkpoint 8

P010 (Dataset Cartography, EMNLP 2020) formalizes per-sample mean gold-label confidence, temporal variability and correctness as learner-dependent coordinates. Its appendix and BERT/RoBERTa comparison show that the global map can look similar while individual samples move between regions. This independently rejects treating a trajectory label as an intrinsic, permanent value attached to an image.

The most important causal clue for Stage1 is the subset-composition failure. Small pure-ambiguous subsets fail to optimize, while replacing only a small part with easy-to-learn examples restores learning; replacing too much reduces performance again. An individually easy, low-gradient sample can therefore have positive set value by supplying coverage or optimization support. This is why gradient magnitude, ambiguity and static top-k scores cannot be sufficient rankings, and why replay composition must be studied at fixed dose and schedule.

The paper's hard-to-learn region also mixes difficulty, ambiguity and label problems. Synthetic flips shift toward low confidence, but real-data human audit leaves imperfect separation and variability-only detection is weak. This supports noise-risk stratification and audit of extreme tails, not wholesale replay of low-confidence samples.

The code audit sharpens the measurement contract. The released implementation records train-mode logits from each sample's own mini-batch before that batch's optimizer step. Samples within an epoch are observed under different parameter states and active dropout. Stage1's existing OOF checkpoint trajectory is a held-out, fixed-checkpoint observable and must not be called the same training dynamic. The final schema now requires three explicit namespaces: presentation dynamics, fixed-view train probes and held-out OOF checkpoint dynamics.

P010 provides no numeric Stage1 replay fraction, stop epoch or weak-defect guard rule. Its experiments replace the training set rather than add duplicate exposure, use only three seeds for central subset results and mix mean-versus-best reporting across tables. It supports all-epoch low-cost capture, conditional coordinates, set-composition pilots and cross-seed stability checks; it does not freeze a large GPU arm.

## Evidence Checkpoint 9

P011 (TracIn, NeurIPS 2020) separates three quantities that are often conflated. Ideal influence is exact target-loss change along one realized update path. First-order TracIn attributes a small SGD step through a target-gradient dot product. TracInCP instead evaluates gradients at saved checkpoints and has an explicit counterfactual interpretation: the sample is scored as if visited at those states. Only the first quantity is exact historical accounting; the practical score is state-conditioned diagnostic evidence.

This directly refines the Stage1 gradient plan. Self-influence is a sum of squared gradient norms and therefore measures leverage, memorization or outlier status, not positive business value. Directional influence must be computed separately against difficult-normal and weak-defect probe losses. Even then, positive raw-gradient alignment is local and does not include curvature, cumulative replay exposure or the optimizer's momentum/adaptive preconditioning.

The optimizer boundary is material. TracIn's derivation is for plain SGD and the paper says other update rules need rederivation. Stage1's canonical configuration must therefore capture both raw-gradient alignment and a controlled actual-update alignment `-g_target dot delta_theta`. The latter is closer to realized effect, while attribution inside a mixed mini-batch remains approximate. Hyperparameter and initial-weight locks are mandatory because these values are path-dependent.

Replay also violates the practical paper assumption that each sample is visited exactly once between epoch-boundary checkpoints. Occurrence count, cumulative exposure and checkpoint interval must enter any exposure-weighted influence summary. A static TracInCP score computed without these fields cannot explain continuous versus decayed replay.

The paper supports checkpoint diversity: different stages identify different data, middle/high-loss-reduction checkpoints outperform a final converged checkpoint, and one final model can be noisy. It does not support a universal checkpoint list or stop epoch. Its 0.978 first-order correlation is a specific MNIST result under small-step conditions, not a guarantee for yolo11l.

The public-code audit narrows claims further. Only ImageNet application notebooks are released, not the central CIFAR/MNIST comparative code. The notebooks use equal weights at checkpoints 30/60/90, last-layer approximations and inconsistent bias handling, with no environment or artifact locks. P011 therefore supports low-cost checkpoint gradient probes and optimizer/exposure diagnostics, but not a gradient-selected large replay arm or any numeric campaign setting.

## Evidence Checkpoint 10

P012 (Influence Functions, ICML 2017) adds local curvature to the direction test. Its core target effect is `-g_target^T H^-1 g_i`: the inverse empirical Hessian measures how much the surrounding dataset resists a candidate update direction. This provides a principled contextual signal beyond loss, norm and raw cosine, and suggests direct diagnostics for redundancy or unsupported directions.

The same paper sharply limits the claim. Influence is an infinitesimal derivative around an empirical-risk minimizer. The authors explicitly identify larger subpopulation changes as open because the model may move too far. Stage1's repeated 0.5-2.5% replay over many epochs is exactly such a finite, path-changing intervention. Curvature-aware influence can explain candidates locally, but only retraining arms can establish replay value.

The deep-model evidence is modest. The non-convex demonstration uses a 2,616-parameter tanh CNN on 10% MNIST with damping 0.01 and reports correlation 0.86 after nearby retraining. Damping changes the local quadratic; LiSSA additionally depends on scale, recursion depth, mini-batch sampling and repeat count. Every influence artifact therefore needs approximation parameters, linear-system residuals, checkpoint/data/config hashes and a small finite-difference calibration.

P012 also reinforces tail separation. The released code averages target gradients before applying the inverse Hessian, but difficult-normal and weak-defect gradients can cancel in that average. Stage1 must compute separate target axes and distributional/worst-case probe summaries. High self-influence remains an audit signal for leverage, noise or ambiguity, not a positive replay score.

The code audit found additional provenance risks: hard-coded model seed zero, cache keys that omit checkpoint/data/approximation identities, direct non-atomic NPZ writes, and a 2020 correction from `sigma` to `sigma squared` in the logistic LOO formula. These findings directly inform the final implementation's cache/manifest tests. P012 supports calibrated curvature-aware probes, not a new large Treatment arm or numeric schedule.

## Evidence Checkpoint 11

P013 (Influence Functions in Deep Learning Are Fragile, ICLR 2021) is the required counterweight to P012. Across Iris, MNIST, CIFAR and ImageNet, influence fidelity changes with depth, width, architecture, target point, weight decay, inverse-HVP approximation and the subset used for evaluation. On ImageNet, ordinary continued training changes target loss enough to swamp the leave-one-out signal; even the nominal retraining ground truth is noisy.

The closest result to Stage1 replay is the group study. ResNet-18 group-influence correlations are only 0.01-0.21 on MNIST and 0.01-0.18 on CIFAR-100. A replay set with repeated finite exposure therefore cannot be treated as a sum of fixed single-image influence scores. The full causal intervention remains `selection x model state x timing x cumulative dose x surrounding data`.

P013 also narrows the proposed gradient field contract. Raw dot products, actual optimizer-update alignment and curvature-adjusted influence must be computed at the same checkpoint and against separate difficult-normal and weak-defect targets. Every curvature result needs an explicit upweight/removal sign, Hessian scope, damping, scale, recursion identity, residual, target-ID manifest and ordinary-training drift control. Full-pool sign agreement and unseen-seed stability matter more than correlation measured only on a top influence-selected tail.

Two manuscript inconsistencies strengthen the provenance requirements: the removal statement after Equation 5 omits the sign implied by `epsilon=-1/n`, and an ImageNet target is identified as both 13,923 and 13,293. The final implementation must test intervention signs and sample identities instead of transcribing them informally.

This paper does not prove that influence is universally useless, and it does not justify any Stage1 replay percentage or stop epoch. It supports a cheap, calibrated mechanism-probe channel and argues against allocating the ten-machine campaign to unvalidated all-network inverse-Hessian ranking. The canonical 240-run hyperparameters must remain locked because model training choices change the meaning and fidelity of these state-dependent measurements.

## Evidence Checkpoint 12

P014 (If Influence Functions are the Answer, Then What is the Question?, NeurIPS 2022) resolves part of the apparent conflict between P012 and P013 by separating five quantities: cold-versus-warm initialization, proximity regularization, non-convergence, local linearization and numerical solver error. In its experiments the first three usually dominate. Practical influence can therefore be a good approximation to a PBRF-like local, prediction-preserving intervention while being a poor predictor of leave-one-out retraining from initialization.

This distinction matters more than choosing a better inverse-Hessian solver. Stage1's scientific estimand is the end-to-end effect of finite replay applied throughout training under a fixed seed, schedule and canonical configuration. A local PBRF score is a mechanism probe, not that causal estimand. Its damping value changes the proximity-constrained question rather than merely stabilizing a calculation.

P014 supplies a useful calibration control: ordinary continued training can dominate a tiny deletion effect, so compare a same-checkpoint continuation branch with an otherwise identical finite intervention branch. Its two-stage LOO subtraction improves agreement with influence. For Stage1, the corresponding fields are baseline drift, finite replay delta and separate difficult-normal/weak-defect target changes.

The paper's evidence still has boundaries. It deletes only 20 random points per setting, reports variability largely across deleted identities rather than initialization seeds, uses average output distance, and does not study replay or an FN-constrained tail objective. The published checklist says central experiment code was not attached; the later PyTorch library implements generic exact/CG/LiSSA influence, not the five-gap suite. Its CG path discards solver status and its LiSSA path lacks an explicit persisted solver RNG. Our implementation needs stronger residual, seed, identity and cleanup contracts.

The synthesis after twelve papers is now sharper: do not ask whether a sample has one influence value. Ask which intervention, at which state, against which target, under which optimizer and cumulative exposure. Use gradient/curvature measurements to explain or predict finite intervention signs, while the preregistered replay arms determine actual value.

## Evidence Checkpoint 13

P015 (SOURCE, NeurIPS 2024) extends local attribution across a segmented training path. Exact unrolling expresses a sample's infinitesimal effect as its gradient at every occurrence, transported through all later update Jacobians. SOURCE replaces the intractable path with segment-average gradients, curvature and learning rates, using a finite-time spectral filter. This supports the Stage1 premise that value depends on model state, training stage, optimizer and cumulative exposure rather than on one final confidence score.

The paper also supplies three direct warnings. First, neural-network attribution accuracy collapses near one-point leave-one-out because training stochasticity overwhelms the effect. Second, averaging ten trained models improves several attribution methods, so one OOF trajectory cannot be treated as a stable per-image ground truth. Third, SOURCE assumes stationarity within segments and independence across segment Jacobians, explicitly neglecting optimization autocorrelation. Repeated replay is likely to create exactly the correlated exposure that violates this approximation.

Its evaluation target remains removal or subset downweighting. LDS additionally predicts a set by summing individual scores, while Stage1 keeps all base data and repeatedly replays a selected set under an asymmetric tail constraint. SOURCE therefore remains a mechanism probe. A tiny same-seed, same-checkpoint finite replay branch, minus ordinary continuation drift, is required to calibrate sign and magnitude before any attribution-derived ranking can enter a formal arm.

The code audit strengthens the engineering boundary. The paper-linked `kronfluence` repository provides the EK-FAC backend but not the central SOURCE experiment pipeline. A later 2026 `simple-influence` implementation supports only Linear/Conv2d SGD-style SOURCE, requires manual momentum scaling, omits the Adam/preconditioner extension, allocates a full query-by-train matrix and lacks Stage1 provenance/resume contracts. The canonical resolved optimizer must therefore be extracted from prior artifacts rather than inferred from `optimizer=auto`, and no off-the-shelf SOURCE run may be applied blindly to yolo11l.

After thirteen papers, the highest-value mechanism fields are now: exact replay occurrence and cumulative dose; per-epoch state; resolved optimizer and effective update; separate normal-tail and weak-defect target gradients; within-segment drift and lag autocorrelation; seed-conditioned attribution and sign stability; and finite intervention calibration. These are explanatory fields. The causal campaign still needs continuous, decayed, dose-matched, weak-defect-guard and no-replay controls under one machine-verified canonical hyperparameter lock.

## Evidence Checkpoint 14

P016 (TRAK, ICML 2023) shows that contextual gradient geometry is more informative than raw gradient similarity. Its inverse after-kernel reweighting and confidence term are critical in ablation, while removing the reweighting collapses LDS close to TracIn-level performance. This independently supports measuring how replay corrects the base/target gradient residual rather than ranking by norm or cosine alone.

The strongest TRAK estimator is explicitly an ensemble quantity. More independently trained models improve LDS, and the paper states that behavior created only by training randomness cannot be explained by data identity. Multiple checkpoints from fewer trajectories can partially substitute for independent models on one CIFAR-10 experiment, but they do not erase initialization heterogeneity. Stage1 must retain seed-conditional scores, sign reversals and worst cases before reporting any mean.

Two ablations prevent an easy implementation shortcut. Full-model projected gradients substantially outperform penultimate/last-layer features, and projection dimension has a non-monotone regularization effect. A cheap last-layer mechanism probe is justified only after agreement calibration on a small stratified pool; no paper projection size is a transferable constant.

TRAK still predicts subset effects additively and uses removal/retraining counterfactuals. Its authors identify early feature-learning dynamics, mini-batches, momentum and weight decay as unmodeled. These are central to repeated Stage1 replay. The method can explain state-specific associations but cannot replace the continuous/decayed/dose-matched/guard/no-replay campaign.

The public-code audit adds concrete safeguards for our implementation. The released persistence layer does not bind sample IDs, checkpoint/model/task identity, projection seed or output function; target scoring lacks a completion mask; score inversion records no condition diagnostics; and the paper's soft-thresholding step is absent from the core API. Stage1's projected-gradient artifacts need immutable row manifests, complete hyperparameter/checkpoint hashes, atomic partitions, per-row completion, numerical diagnostics and explicit estimator variants.

After fourteen papers, the attribution tranche is sufficient to freeze one negative decision: do not spend the first seven-day GPU cycle on another static gradient/influence-ranked Treatment. The next evidence tranche must address the actual causal variables still open: replay timing, cumulative exposure, optimizer path, memorization/noise transitions and weak-tail constraints.

## Evidence Checkpoint 15

P017 (Data Echoing, arXiv 2020) directly studies repeated training observations. Its most transferable result is not a numeric echo factor: repeated examples are useful but generally less useful than fresh examples, and their utility changes with where repetition is inserted, whether augmentation is resampled, batch size, shuffle-buffer capacity and the distance between copies. Cumulative replay count alone is therefore an incomplete exposure description.

The paper also exposes a critical experimental confound. Every baseline and echo condition independently tunes learning rate, momentum and schedule parameters, then selects the best trial reaching a broad validation target with the fewest fresh examples. This establishes an attainable systems tradeoff after tuning, not the causal effect of replay under one fixed optimizer path. Stage1 must do the opposite: extract the exact canonical configuration from the 240-run artifacts, hash-lock it, pair seeds and reject any arm whose non-replay hyperparameters drift.

Data Echoing tests constant repetition of the full training distribution. It does not stop replay late, relocate a fixed cumulative dose, target hard normals, protect weak defects or report seed-level sign reversals. It therefore cannot justify 140/160 epoch boundaries or any replay percentage. It does justify the minimal causal decomposition of continuous replay, same-peak decay and cumulative-dose-matched decay, provided the numeric schedule is chosen independently and discrete slot integration is verified exactly.

P017 expands the mandatory collection schema: planned and realized replay slots per epoch; cumulative per-sample occurrences; base versus total optimizer examples and steps; replay concentration within batches; lag between repeated identities; augmentation-view identity; shuffle/order digest; and separate training, loader-wait, evaluation, write and idle time. These fields let us distinguish statistical harm from mere systems behavior and determine whether nominally identical percentages produced different realized exposure.

After fifteen papers, the first-cycle negative decision remains unchanged: no new static gradient/influence Treatment. The leading causal question is now sharper: with the same selected IDs, seed, base stream, optimizer and cumulative exposure, does moving replay away from late training reduce weak-defect harm? A no-replay arm and separate normal-tail/weak-defect trajectories remain necessary because average validation quality can hide the exact failure mode observed in the 240 runs.

## Evidence Checkpoint 16

P018 (Stochastic Optimization with Laggard Data Pipelines, NeurIPS 2020) provides the first formal decomposition of repeated-data training into two resources: `B*T` fresh independent examples and `K*T` optimization steps. Under smooth convex assumptions, echoed GD can improve the curvature term from order `1/T` to `1/(K*T)` while the optimal statistical term remains order `1/sqrt(B*T)`. Repetition can finish optimization on existing information; it cannot manufacture independent information.

The same derivation gives the harm mechanism. Uniform stability for `K` GD steps on one batch scales as `2*eta*rho^2*K/B`. More repeated steps improve the potential term but increase sensitivity to one stale batch. The tuned step size balances these terms; proximal regularization can make the step-size choice less sensitive to `K`. This supplies a principled interpretation of late replay saturation, but not a Stage1 stopping rule.

The transfer boundary is severe and useful. Theory assumes convex smooth losses, deterministic GD, i.i.d. replacement-sampled batches and a tuned learning rate. The experiments use tiny logistic regressions, zero initialization, training-loss convergence and a separate learning-rate search for every batch/echo pair. The authors explicitly identify deep-network learning-rate, BatchNorm and re-augmentation interactions as unresolved confounds. Stage1 must hold the 240-run configuration fixed and estimate the replay effect under that path instead of importing an optimized echo factor.

P018 strengthens the timing-dose design. An equal-peak decay changes cumulative `sum_t K_t`; it cannot identify timing alone. A cumulative-dose-matched relocation arm is necessary. The collector must preserve fresh identities, base/replay occurrences, optimizer steps, integer schedule area, batch concentration, lagged replay-gradient correlation, momentum-adjusted update alignment and marginal normal-tail/weak-defect change per exposure.

After sixteen papers, the highest-value first-cycle question remains causal and state-dependent: does moving a fixed replay dose away from a saturated late phase reduce weak-defect harm without sacrificing hard-normal benefit? Saturation diagnostics can explain the result, but they must not adapt the first confirmatory schedule after outcomes are visible.

## Evidence Checkpoint 17

P019 (Learn the Time to Learn, TMLR 2023) supplies direct empirical evidence that replay timing is not interchangeable. In its motivating experiment, the same ten Task-1 examples are replayed once at different later tasks and final five-seed ACC ranges from 89.66% to 94.49%. This is the cleanest current support for treating time as part of replay value rather than as an implementation detail.

The transfer boundary is decisive. The paper schedules proportions of old tasks in a continual-learning stream, whereas Stage1 adds selected duplicate observations to one stationary classification run. Its MCTS retrains a model 100 times and chooses the maximum final validation reward. The UCT implementation also exploits the maximum reward seen below a node rather than the mean. This is legitimate within the paper's stated algorithm, but it is an adaptive discovery procedure with substantial optimizer's-curse risk, not a confirmatory design.

The complete appendix prevents a broad success claim. Many five-seed comparisons are not significant, some MCTS schedules lose to random or heuristic policies, transferred schedules differ by source seed, and DQN/A2C policies fail to dominate on new FashionMNIST environments. Fixed baselines are copied across RL seeds in some Welch tests, producing zero variance and infinite statistics. Schedule value is therefore conditional on dataset, state, history, replay method and seed, exactly as Stage1's same-selection reversals suggest.

The code audit adds operational constraints. Candidate test metrics are computed during every rollout even though validation accuracy is the programmatic reward; checkpoint caches do not bind full config/data/code identity; writes are non-atomic; there is no scientific test suite; and the all-schedule history accumulator is corrupted by variable reuse. The final Stage1 system must use immutable manifests, role-based split access, content hashes, atomic completion and paired per-seed reporting rather than inheriting this runner.

After seventeen papers, the first causal block should remain fixed and small: no replay, continuous replay, same-peak decay and cumulative-dose-matched relocation on one frozen selection. P019 strengthens the timing hypothesis but supplies no epoch boundary or percentage. Numeric schedules must be preregistered from Stage1 trajectories and operational feasibility, then confirmed on unseen seeds under the exact 240-run hyperparameter lock. Adaptive scheduling can be studied only after this fixed block establishes that timing itself causally changes difficult-normal benefit and weak-defect harm.

## Evidence Checkpoint 18

P020 (Early-Learning Regularization, NeurIPS 2020) adds a specific dynamic mechanism. In cross-entropy training, confidently learned examples have shrinking `p-y` factors. A smaller conflicting group can therefore become relatively dominant later even if its absolute gradient does not grow. The paper's linear model proves this only in a narrow high-dimensional Gaussian/noisy-label regime, but deep-network figures exhibit the same qualitative transition. For Stage1 the falsifiable analogue is not “replay samples are noisy”; it is that replay-normal contribution may rise relative to the learned base stream and eventually conflict with the weak-defect target.

The method and negative ablations show why trajectory magnitude alone is insufficient. Temporal targets without the directional ELR term eventually follow noisy labels, while a KL consistency penalty can merely delay memorization and then overfit its early targets. Useful process fields must therefore keep temporal stability, gradient magnitude, target direction and realized exposure separate. A smoothed confidence curve cannot be promoted to sample value.

The transfer boundary is unusually important. The theorem assumes binary linear softmax regression, `p/n` near one, random label replacement and sufficiently small Gaussian variance; the NeurIPS meta-review questions the precise variance scaling. Stage1 uses labeled replay in a nonlinear binary classifier and optimizes a raw FN-constrained tail metric. P020 provides no replay percentage, no stop epoch, no weak-defect guard ratio and no evidence that every late phase is harmful.

The empirical and code audit further limit stability claims. Main tables use five noise realizations rather than paired initialization-seed blocks; some baseline columns use different selection conventions. The public code leaves NumPy unseeded even though it controls data splitting, label corruption and mixup, constructs the two ELR+ loaders independently, evaluates the official test set every epoch and cannot faithfully resume its per-sample target histories. These gaps strengthen Stage1's requirements for immutable data/seed identities, full replay state, role-based split access and atomic resumable artifacts.

After eighteen papers, the first-cycle decision remains unchanged, but the mechanism collector becomes sharper. Persist all-epoch base/replay loss and `|p-y|` summaries, replay-to-base gradient contribution ratios, learned/forgotten/never-learned counts and separate hard-normal versus weak-defect gradient alignment. At key checkpoints compare raw gradient alignment with the actual optimizer update and a same-state finite intervention. Decay is supported only if late replay dominance and weak-defect conflict appear in Stage1 itself; the first confirmatory schedule must remain fixed under the canonical 240-run hyperparameter lock.

## Evidence Checkpoint 19

P021 (AUM, NeurIPS 2020) adds a practical sample-trajectory statistic: average the assigned-label logit margin over training rather than trusting one epoch. In one CIFAR-10 40% uniform-noise experiment, AUM rankings exceed 98% cross-architecture Spearman correlation, while a single margin or training loss is around 75% and validation loss around 40%. This supports full-epoch low-cost trajectories and explicit cross-seed rank agreement instead of single-checkpoint confidence.

The negative evidence is equally important. AUM degrades slightly on ImageNet, fails badly under 40% pairwise asymmetric noise and produces non-bimodal distributions on real datasets. Its low tail therefore indicates persistent assigned-label conflict, not automatically label error and certainly not positive replay value. Reviewers also questioned class imbalance, threshold construction, removal controls and systematic noise.

The measurement protocol is a confound if copied carelessly. The paper deliberately reduces batch size from 256 to 64 and stops at the first learning-rate drop to inhibit memorization. Stage1 may not import either change: all formal arms must retain the exact 240-run canonical hyperparameters. AUM-like fields must be observed inside those runs, with no extra artificial class or altered optimizer path.

The code audit reveals a replay-specific issue absent from the paper definition. The official calculator averages over every presentation. Replaying an identity more often therefore changes its AUM weighting even when its epoch-level trajectory is unchanged. Stage1 must store both epoch-weighted and presentation-weighted margins, plus per-epoch and cumulative occurrence counts. Their difference measures exposure, not intrinsic sample quality.

The released tests pass but cover only basic arithmetic. Batch size one crashes, AUM state cannot resume, sample IDs are subset positions, outputs are non-atomic and large-dataset scripts select checkpoints after evaluating test data every epoch. These issues reinforce content-bound identities, complete collector state, atomic sidecars and role-based split access.

After nineteen papers, AUM becomes a diagnostic channel, not another Treatment ranking. The causal campaign remains no replay versus continuous, same-peak decay and cumulative-dose-matched replay under one hash-locked canonical configuration. AUM-like trajectories can explain persistent disagreement and transition timing; only paired finite replay outcomes can establish value for the raw `FN=0-95` safety frontier.

## Evidence Checkpoint 20

P022 (A Closer Look at Memorization, ICML 2017) supplies foundational evidence that an over-parameterized network does not fit real and random data by the same path. Real examples have stable identity-specific ease across 100 one-epoch initializations, random-input examples look closer to binomial variation, and mixed random-label runs reach peak validation quality before fitting the random labels. Decision-boundary complexity, measured by a search-dependent critical-sample ratio, rises as memorization proceeds.

This supports transition-aware collection but not a transition rule. Most central figures are qualitative, replicate counts outside the one-epoch experiment are unspecified, and random labels or Gaussian inputs are synthetic extremes. The paper provides no universal epoch, no Stage1-like replay, no weak-defect target and no raw safety-frontier evaluation.

The paper's definition of effective capacity is especially relevant: a learning algorithm includes architecture and the full training procedure. Changing batch size, regularization, schedule or optimizer changes the object under study. Formal Stage1 arms must therefore inherit the exact canonical 240-run hyperparameters. The paper's dropout and adversarial-training sweeps are literature context, not allowed campaign changes.

Its unrolled loss-sensitivity differentiates future loss through SGD updates with respect to an earlier training example. That is conceptually closer to path-dependent value than immediate parameter-gradient norm, but the demonstration uses a two-layer 16-unit model and 1,000 downscaled MNIST examples. Full-model, all-sample unrolling is not justified. Low-cost checkpoint gradients and same-state finite interventions remain the appropriate calibration route.

P022 also complicates a simplistic capacity story: higher capacity can sometimes fit noise with less interference to real patterns. Likewise, regularizers differ in how much they slow random-label memorization versus clean learning. Harm is an interaction among data, state, model and procedure, not a property of one image.

After twenty papers, the mechanism fields now include cross-seed ease, first-learned and forgetting transitions, exposure-normalized margins, replay/base contribution concentration, local boundary-complexity proxies and weak-defect harm. None is promoted to a weighted value formula. The first confirmatory intervention remains no replay, continuous replay, same-peak decay and cumulative-dose-matched relocation under the canonical lock.

## Evidence Checkpoint 21

P023 (Progressive Early Stopping, NeurIPS 2021) adds a module axis to the timing hypothesis. Its clean-retraining probe reports that later network parts peak earlier and deteriorate more sharply under several synthetic noise processes. This supports measuring head and backbone dynamics separately: one whole-network loss curve can hide antagonistic module behavior, just as one average validation score can hide difficult-normal gain and weak-defect harm.

The method does not provide a Stage1 cutoff. PES reinitializes later modules, freezes earlier modules, trains the replacements with Adam for separately tuned durations and then adds confident-example weighting or MixMatch. Its reported `T1/T2/T3` vary by dataset and architecture, and the authors identify the additional timing hyperparameters as their main limitation. These values cannot be imported into a canonical yolo11l replay experiment.

The code audit materially narrows literal replication. The public CIFAR validation path applies random training augmentation; semi-supervised scripts report the maximum test accuracy; there is no deterministic or resumable experiment contract; and metric weighting uses channel count instead of batch count. More seriously, renewed layer4/classifier parameters are absent from the original SGD optimizer. A live identity probe found zero of 17 replacement parameters in its parameter groups, so later apparent full-model training leaves those replacement parameters implicitly fixed. Current Clothing1M defaults also move refinement and LR drops about ten paper-defined epochs later than the paper reports.

These discrepancies do not erase the qualitative stage/module hypothesis, but they prevent treating the released algorithm or numeric settings as authoritative. Stage1 should add head-versus-backbone raw gradient, actual update, replay/base contribution and target-alignment fields at key checkpoints while preserving all 200 epochs of low-cost trajectories. Formal arms must not reset or freeze modules.

After twenty-one papers, the first causal intervention remains no replay versus continuous, same-peak decay and cumulative-dose-matched relocation under the exact canonical 240-run hyperparameter lock. P023 provides a falsifiable explanatory sequence: late replay contribution may become relatively dominant in the head while weak-defect alignment turns adverse. Only paired unseen-seed finite outcomes can establish that this sequence predicts safety-frontier value.

## Evidence Checkpoint 22

P024 (two-dataset constrained learning, ICML 2019) adds a validation architecture rather than a new replay selector. Its model player minimizes an objective and differentiable proxy constraints on training data, while its constraint player enforces the original constraints on an independent validation set. The paper's Neyman-Pearson example explicitly separates false-positive minimization from a false-negative constraint. For Stage1, the transferable principle is to keep difficult-normal benefit and weak-defect safety as distinct quantities and to estimate safety on an identity-frozen stream that was not used to derive the replay ranking.

The result does not say that validation constraints automatically generalize. The authors retain a validation generalization term, warn about hyperparameter overfitting and prove their strongest bounds only for oracle/discretized or strongly convex algorithms. Their practical neural algorithms have no formal guarantee. Real-data means over 100 random splits usually favor two datasets for constraint generalization but can pay objective error; Business Entity Resolution is a visible case with no clear constraint gain and worse error.

The code audit blocks literal transfer. The linked notebook covers only Communities and Crime with one fixed split and a 10-unit network, not the paper's four datasets, 100 random splits and linear Communities model. It computes label and protected-group thresholds before train/test splitting, downloads data without a hash and exposes test predictions at every candidate snapshot. More seriously, it encodes labels as 0/1 but uses `label < 0` for protected-group negative slices, making those slices empty. The split-rate-context rewrite was committed after ICML 2019, so the current tutorial is not an identified paper snapshot.

P024 therefore strengthens role-based data lineage, original-versus-proxy metrics and constraint-generalization fields, not constrained-training changes. The formal Stage1 arms must retain the exact canonical `yolo11l`, batch 128, image size 224, workers 4, optimizer, schedule and augmentation. OOF/train may discover candidates, a fixed `val_op` stream may calibrate normal-tail benefit and weak-defect non-inferiority, and blind holdout remains closed until the replay rule is frozen.

After twenty-two papers, the minimum replication-depth target is met but the 50-full-read evidence gate is not. The first causal block remains no replay, continuous, same-peak decay and cumulative-dose-matched relocation on one frozen selection. P024 adds a mandatory reporting condition: a replay policy is not safe because its surrogate improves; the original raw FN constraint and its train-to-probe generalization gap must be reported separately.

## Evidence Checkpoint 23

P025 (Neyman-Pearson umbrella classification, Science Advances 2018) converts the asymmetric-error discussion into an exact calibration feasibility statement. For a scorer fixed independently of a held-out prioritized-class sample, selecting an order-statistic threshold controls the probability that population prioritized error exceeds `alpha`. The two quantities must remain separate: `alpha` is the tolerated error rate and `delta` is the tolerated probability of violating that rate.

For Stage1's approximate `alpha=0.005` recall requirement and `delta=0.05`, the minimum independent calibration size is 598. Ninety-five is an allowed outcome count, not enough calibration evidence: with only 95 prioritized identities, even the most conservative finite threshold has violation probability about 62.1%. Stage1 may report observed `FN<=95` without this gate, but it may not call that a distribution-free 95% population guarantee.

The class orientation matters. The paper controls class-0 error for classifiers whose scores increase toward class 1. To protect Stage1 false negatives, defect must be treated as prioritized class 0 and the rule applied to a normal-oriented score, equivalently `-p_defect`. Orientation must be fixed semantically before calibration; the package's data-dependent sign check uses held-out class-0 scores and does not cleanly match the theorem's independence condition.

The full code audit prevents a literal package import. Official reproduction code names `nproc` 2.0.9 and includes data and precomputed results, but two major simulations are not seed-complete when recomputed. The paper swaps two printed minimum-size examples and disagrees between 100 and 1,000 real-data repetitions; official code confirms 1,000. Version 2.0.9 has an invalid insufficient-sample path, 2.1.1 repairs it, and the 2.1.4 efficiency rewrite retained by 2.1.5 reintroduces an unguarded empty-rank path. `split=0` violates calibration independence, while majority-vote multi-split control is empirical rather than covered by Proposition 1.

P025 therefore adds no training arm and no training hyperparameter. It adds role-separated manifests, exact `alpha/delta/n_min/rank/bound` fields, a fail-fast calibration-size check, a fixed score orientation, tie/dependence diagnostics and a post-freeze conservative operating point. Raw `FN=0-95` frontiers remain the utility outcome; the NP point is a separate safety statement.

After twenty-three papers, the causal replay question remains no replay versus continuous, same-peak decay and dose-matched relocation on identical selected IDs under the canonical 240-run lock. The evaluation architecture is now stricter: OOF/train for discovery, a fixed policy-calibration stream for schedule decisions, an independent defect stream for one-split threshold calibration and blind holdout only after all rules are frozen. Correlated video frames must be audited at group level before any i.i.d. claim.

## Evidence Checkpoint 24

P026 (partial-AUC via DRO, ICML 2022) changes the unit of analysis from an isolated image to a tail pair. With defect as positive and normal as negative, one-way pAUC compares defects against top-scored normals; two-way pAUC additionally restricts the positive side by score rank. A replay normal's relevance is therefore conditional on which weak defects it outranks, how concentrated those violations are and the current model state. This provides a principled explanation for why a fixed static `V(x)` can reverse across seeds.

The paper offers two complementary tail summaries. CVaR gives an exact hard top-fraction surrogate through a per-positive auxiliary threshold, while KL-DRO gives a smooth log-sum-exp approximation that moves between worst-pair and full-AUC behavior as its temperature changes. Their disagreement is useful as an extreme-outlier sensitivity diagnostic. It does not yield a calibrated Stage1 value score: the mapping from KL temperature to FPR is tuned, the tested FPR/TPR regions are moderate, and the paper does not address `FN<=95` or raw-frontier dominance.

The experiments also block a simplistic success claim. Exact SOPA often beats smooth SOPA-s on images, a naive mini-batch baseline wins one moltox21 OPAUC setting, rare molecular TPAUC results have large dispersion, and the exact-versus-smooth approximation check uses only one dataset with 100 random parameter vectors. The exact number of train/split/seed repetitions behind reported means is unspecified. No paper percentage, gamma, lambda, batch size, optimizer or epoch count may enter Stage1.

The official LibAUC audit exposes two implementation requirements relevant to our collector. The method maintains per-positive moving state keyed by stable sample index, but v1.4.0 does not register that state in the loss `state_dict`, so ordinary resume changes the path. Its bundled SOPAs example also uses constructor and forward keyword names incompatible with the tagged loss implementation, and no relevant automated tests were found. Stage1 must make collector state, completion masks, identity manifests and resume lineage explicit rather than copying this implementation.

After twenty-four papers, P026 adds no formal training arm and leaves the canonical 240-run configuration untouched. It adds a nested mechanism test: at key checkpoints, measure sparse defect-normal pair violations, hard and smooth tail risk, per-identity partner concentration, and separate alignment to difficult-normal correction and weak-defect protection. Continuous, same-peak decay, dose-matched relocation and no-replay remain the causal arms. Pair diagnostics support the timing mechanism only if their paired trajectory change accompanies raw-frontier improvement on unseen seeds.

## Evidence Checkpoint 25

P027 (structural SVM for pAUC, ICML 2013) makes the state dependence of the operational tail explicit. The negative examples inside an FPR interval are selected by the current scoring order. When the model changes, the identities defining the pAUC loss can change. Before a restricted ordering theorem is applied, the loss-augmented problem is non-decomposable because each candidate ordering induces a different negative subset. This is a direct theoretical reason not to treat the current normal high tail as a permanent high-value list.

The finite-sample definition also matters for Stage1. Empirical pAUC uses integer order statistics plus fractional boundary terms and assumes enough negatives to resolve the interval. Narrower tails contain fewer effective independent identities, are more sensitive to ties and video duplication, and require explicit endpoint conventions. The runtime experiment shows a complementary operational effect: one most-violated-constraint call costs about the same across tail widths, but the number of calls rises sharply as `beta` shrinks.

The paper's experimental record is mixed enough to prevent over-transfer. Targeted pAUC training improves several datasets, but loses to full-AUC SVM on TREC10 at pAUC `[0,0.1]` and at full AUC; several other gains are not significant. All models are linear, uncertainty reporting is limited, PPI uses transductive preprocessing and assumed-negative labels, and patient-group separation is not reported for random breast-ROI splits. The legacy code link is not a verifiable snapshot.

After twenty-five papers, add dynamic tail membership, entry/exit, adjacent-epoch Jaccard turnover, finite endpoint weights, tie policy and effective video/group count to the mechanism schema. These fields remain descriptive until paired interventions establish outcome effects. The formal causal block and exact canonical 240-run hyperparameter lock remain unchanged.

## Evidence Checkpoint 26

P028 (generic TPAUC optimization, ICML 2021) joins the two operational tails into one model-dependent object. Its empirical target selects bottom-scored positives and top-scored negatives under the current scoring function, then evaluates their pairwise ordering. This makes a weak defect's relevance depend on which difficult normals it is paired with, and makes both sets change as the model changes. It supports dynamic two-tail membership and pair diagnostics, not a static image value.

The paper also supplies direction-agnostic evidence that training stage matters. It warns that emphasizing hard examples at the beginning can overfit and inserts an ordinary-AUC warm-up before activating TPAUC weighting. Delay sensitivity is strong for its exponential weighting on several CIFAR subsets but weaker and more variable for polynomial weighting. This does not prove Stage1's late-replay decay: delayed onset and early stopping are different interventions. It only strengthens the need for same-selection, same-seed timing contrasts.

The empirical record is too incomplete for numeric transfer. Tails cover 30%-50% of each class, far from Stage1's extreme operating region. Table 1 has no seed count or uncertainty, sensitivity boxes combine hyperparameter settings rather than stochastic replicates, and a “significantly” claim has no reported statistical test. Total training epochs are omitted, the delay grid disagrees with plotted values, polynomial parameter notation conflicts with its stated domain, and no official code snapshot was found.

The theory is conditional rather than universal. Its upper bound depends on a data- and model-dependent sufficient condition; its generalization result additionally assumes no ties, iid class draws, bounded surrogate loss and sufficient positive and negative counts. Stage1 must therefore record tail effective sample size, video/cluster concentration, tie policy and exact finite boundary ranks instead of treating nominal percentages as equally informative.

After twenty-six papers, P028 adds no formal arm and changes no canonical hyperparameter. It adds all-epoch joint tail membership, pair violations, pair-weight concentration and activation/exposure state to the collector. Continuous, same-peak decay, cumulative-dose-matched decay and no replay remain the causal timing comparison under one canonical-lock hash. Any later delayed-onset arm would require a separate preregistration based on Stage1 evidence, not retrospective use of this paper.

## Evidence Checkpoint 27

P029 (importance sampling, ICML 2018) separates three quantities that were previously easy to conflate. Per-sample gradient norm measures current update leverage. The dispersion of those norms determines how much an unbiased importance sampler can reduce stochastic-gradient variance. Neither quantity says whether the update points toward difficult-normal correction, weak-defect protection or raw safety-frontier improvement. Target alignment and finite paired outcomes are separate evidence channels.

The method is dynamic by construction. It presamples a fresh candidate batch, scores it with the current model, resamples with replacement and applies inverse-probability correction. The paper explicitly rejects a one-time importance ranking because sample importance changes with model state. Its conditional `tau` statistic turns importance sampling on only when predicted variance reduction justifies the extra forward work. This supports collecting score age, state identity, sampling probability, actual weight and realized exposure; it does not support global static top-k replay.

The negative results are unusually useful for Stage1. Loss is poorly correlated with true gradient norm except near the small-gradient regime. High-loss sampling increases variance early, helps the easier CIFAR-10 setting only with fresh scores and warmup, fails on CIFAR-100 and hurts pixel-permuted MNIST. Therefore neither loss nor confidence can stand in for gradient leverage, and gradient leverage still cannot stand in for target value.

The empirical evidence is about fixed-wall-clock convergence and averages only three independent runs for the main image and fine-tuning curves. It provides no confidence intervals, FN constraint, replay duplication, weak-defect guard or cross-seed stability test. The theoretical derivation uses inverse-probability correction and a distance-to-one-optimum convergence expression. Stage1 replay intentionally changes exposure without that correction, so the guarantee does not transfer.

The official `v0.7` audit confirms fresh candidate scoring and conditional activation but exposes a path-reproduction gap. Checkpoints save weights without NumPy/TensorFlow RNG, condition EMA, sampler cache, iteration and logger state. An interrupted run therefore cannot reconstruct the same activation or sampling path. Dependencies also leave TensorFlow/Keras unpinned, and no condition/resume/atomic-output tests were found. The Stage1 collector and runner must persist every adaptive state and bind it to the canonical config and data manifests.

After twenty-seven papers, add last-head gradient magnitude, separate alignment to difficult-normal and weak-defect targets, gradient-dispersion or `tau`-like diagnostics, score staleness, intended probability/weight, realized presentation count and gradient-to-actual-update alignment. Keep them as mechanism fields, not a blended value score. P029 adds no formal arm and changes no hyperparameter: the first causal block remains no replay, continuous replay, same-peak decay and cumulative-dose-matched relocation under the exact canonical 240-run lock.

## Evidence Checkpoint 28

P030 (Meta-Weight-Net, NeurIPS 2019) provides direct mathematical support for target direction but also exposes a sharp information limit. Its bilevel update learns a shared function from scalar training loss to sample weight. Equation 6 contains the dot product between each training gradient and the average clean-meta gradient, so agreement can raise the learned curve near that loss and disagreement can suppress it. This supports collecting target-gradient dot products and cosines rather than treating gradient norm or loss as value.

The learned output is not an identity-level value. At a fixed model state, two samples with equal loss receive the same weight even if one improves difficult-normal correction and the other damages the weak-defect tail. The directional signal updates the global loss-to-weight curve; sample identity, video context, class-tail role and pair interactions are absent from the input. The Stage1 falsifier is therefore explicit: measure within narrow loss bins how often normal-tail and weak-defect alignments differ in sign.

The empirical curves also argue against one universal hardness rule. The learned weighting function increases with loss under class imbalance, decreases under synthetic label noise and is non-monotone on Clothing1M. Yet the method is not best in every table cell, Table 1 has no uncertainty, and the main noisy-label comparisons use short schedules that a reviewer worried may not let baselines converge. Five repetitions in the noise tables do not establish same-selection cross-seed stability or raw-frontier safety.

The theoretical record must be treated cautiously. Reviewer 3 identified missing classifier-state dependence, an unproved smoothness step and non-telescoping proof transitions. The authors acknowledged missing dependence and introduced additional assumptions and revised derivations. The meta-review accepted the work while recording unresolved reviewer concern. These smoothness, bounded-gradient, Hessian and step-size results do not imply Stage1 safety-frontier improvement.

The official code audit strengthens split and resume requirements. The repository has no paper tag; its post-publication stable script differs from the paper and appendix in architecture, epochs, schedule, VNet optimizer and weight normalization. More importantly, clean-meta and ordinary training datasets independently shuffle and select class indices with unseeded NumPy, so they are not complementary. An exact-logic probe produced 972-987 overlaps out of the intended 1,000 CIFAR-10 meta identities. The scripts also lack real resume and checkpoint state, evaluate test data every epoch and report best test accuracy.

After twenty-eight papers, P030 adds strict identity-disjoint role manifests, probe-overlap preflight, target-gradient composition, same-loss directional dispersion and separate alignment to difficult-normal and weak-defect objectives. It adds no MW-Net arm and changes no Stage1 hyperparameter. The causal campaign remains no replay, continuous, same-peak decay and cumulative-dose-matched replay under the exact canonical 240-run lock; bilevel quantities are mechanism diagnostics whose value must be calibrated against same-state finite interventions and paired raw-frontier outcomes.

## Evidence Checkpoint 29

P031 (Automated Curriculum Learning, ICML 2017) supplies the cleanest mathematical warning so far against using gradient magnitude as value. Under a local first-order SGD model, true expected progress is proportional to the squared norm of the mean gradient. Same-sample prediction gain is proportional to the mean squared gradient norm, so its expectation equals true progress plus gradient variance. Gradient prediction gain uses the same squared-norm signal with an additional Taylor approximation. Large self gain can therefore identify a high-variance task rather than a direction that generalizes.

The proposed alternative is dynamic and role-aware. A nonstationary bandit changes its task distribution as the student changes. Self-prediction gain evaluates the update on a fresh sample from the same task and is unbiased under the paper's derivation, while target prediction gain evaluates the desired target distribution. Both require extra samples and have higher variance; target gain can also be nearly useless early when the target is not yet learnable. This supports separate replay-sample, same-stratum, difficult-normal and weak-defect probes rather than one blended reward.

The negative results matter as much as the positive ones. Uniform task sampling is a surprisingly strong baseline. On Repeat Copy, GPG, GL2G and L2G are much worse than uniform; on bAbI, several adaptive signals are equal or worse, GVCG starts faster and becomes slightly worse later, and VCG fails for reasons the authors cannot explain. The N-gram curriculum is explicitly superfluous because direct target training is already optimal. Adaptive sampling is not beneficial merely because it is dynamic.

The empirical scope is narrow. Ten initialization replicates and standard-deviation bands are useful, but the tasks are synthetic sequence curricula with known task labels. bAbI is regenerated at one million stories per task and reports training performance because train and evaluation curves are said to be indistinguishable. There is no identity-level image replay, label-noise audit, weak-defect constraint, paired selection-versus-seed decomposition, raw safety frontier, public code, or public review record.

The controller also has state beyond model weights: task probabilities, importance-corrected rewards, exploration, a reward reservoir, and clipping quantiles. Any future adaptive Stage1 policy would need to checkpoint all of it. Its progress-per-processing-time reward combines scientific effect with compute efficiency, so Stage1 must keep raw tail utility and utility per GPU-second separate.

After twenty-nine papers, add same-sample versus independent-probe pre/post loss, per-stratum mean squared gradient norm versus squared mean-gradient norm, progress-signal bias and variance, transfer to unpresented related groups, adaptive-state identity and progress per GPU-second. P031 adds no bandit arm and changes no canonical hyperparameter. The formal first-cycle intervention remains no replay, continuous, same-peak decay and cumulative-dose-matched replay; process fields test whether the same selection's independent tail progress changes sign with training stage.

## Evidence Checkpoint 30

P032 (On the Power of Curriculum Learning in Training Deep Networks, ICML 2019) makes the cleanest conceptual separation so far between a scoring function and a pacing function. The former ranks examples; the latter controls how much ranked data is visible over time. For Stage1, a frozen replay selection and its replay schedule are therefore separate interventions. A result cannot be attributed to sample identities when schedule, cumulative exposure, optimizer steps or other training settings also change.

The paper also documents the central confound. Restricting the visible dataset changes repeated exposure and effective learning-rate behavior. The supplement introduces varied pacing partly to compensate for this effect and separately tunes conditions. That is defensible for finding each method's best result, but it prevents a pure causal comparison of ranking or pacing. Stage1 must do the opposite: preserve the exact canonical 240-run hyperparameter lock for every arm and use same-peak decay plus cumulative-dose-matched decay to distinguish late timing from lower total replay dose.

The empirical record argues against replacing one static score with another. Transfer and self-taught target-model rankings help in several cases, while self-paced current-model ranking hurts throughout its main comparison. Repeated bootstrapping accumulates scoring errors and eventually degrades performance. Single-step pacing can be competitive, suggesting an early effect, but the tasks, endpoints and interventions differ too much to select a Stage1 stop epoch. The AUC robustness analysis even changes which curriculum variant appears best, showing why endpoint and trajectory summaries must be preregistered.

The gradient evidence is narrow. One arbitrary first-batch 10% subset shows transfer-ranked examples with a different mean gradient and lower total variance than random or all-data controls. This is group-level descriptive evidence, not per-sample value, target-tail alignment or downstream causal attribution. The theoretical utility-covariance result is likewise conditional on an unknown ideal curriculum and a strong covariance-maximization assumption. Neither establishes that practical confidence scores or early replay improve the Stage1 raw safety frontier.

The official-code audit adds several engineering falsifiers. The untagged post-paper runner covers only basic transfer ordering and fixed exponential pacing, assigns different learning-rate schedules to different arms, does not call its cross-validation update method, leaves NumPy/SVM randomness unpersisted, evaluates the test set during training, and saves neither complete checkpoints nor resume state. Dependencies are not locked and no automated tests exist. These are precisely the failure modes the Stage1 manifest, canonical lock, role separation and atomic checkpoint contracts must prevent.

After thirty papers, P032 adds no formal arm and changes no canonical hyperparameter. It strengthens the existing four-condition causal core: no replay, continuous replay, same-peak decay, and cumulative-dose-matched decay on one frozen selection and seed. Add all-epoch intended and realized ratio, effective examples, optimizer steps, unique coverage, duplicate fraction and cumulative exposure; keep timing, dose and selection identities independently hashed. Any arm-specific learning-rate or augmentation retuning must fail preflight rather than be interpreted as a replay-policy result.

## Evidence Checkpoint 31

P033 (AutoSampling, ICML 2021) directly optimizes an ordered identity schedule rather than assigning one static score. Its learned frequency distributions differ at epochs 80, 160 and 240; a schedule sampled from the final aggregate distribution improves over uniform but remains worse than the online dynamic path. This supports a conditional path object and all-epoch exposure measurement, not a permanent `V(x)`.

The strongest result for Stage1 is also the largest confound. Every short interval trains many child models on different sampled batches, chooses the best held-out-validation result and copies that winner's weights to all workers. Random Exploration still samples uniformly, yet already gains substantially over one-worker uniform training. The nonuniform mixture provides inconsistent additional gains and is worse than random for DenseNet-121 and slightly worse on CIFAR-10. The reported effect therefore mixes sample order, stochastic optimizer-path search, maximum-of-many winner selection, model cloning and repeated validation adaptation.

This distinction changes the collector contract. A short-window reward is not a property of the batch unless parent state, branch RNG, candidate schedule, probe identities, competing branches, selection event and horizon are known. Selecting the maximum among 20 or 80 noisy branches is optimistically biased. Any future adaptive Stage1 policy would require an identity-disjoint selection probe, cumulative query counts and a blind final endpoint. The current fixed-arm campaign should avoid population branching so timing remains the only manipulated factor.

The paper also documents a concentration failure. Raw winning frequencies become so skewed that most data receive zero probability and training destabilizes, requiring logarithmic smoothing and explicit uniform mixtures. Frequency and loss are not visibly correlated in a 500-image sample, and low-frequency examples include both easy clean images and malformed low-quality examples. High or low exposure cannot be interpreted as difficulty, cleanliness or value without separate fields.

Empirical reliability is underreported. Most CIFAR tables show a plus/minus quantity without defining its meaning or repetition count, ImageNet has no uncertainty, the exact split and seeds are absent, and comparisons to prior methods use different architectures and only roughly aligned protocols. No official code or schedule artifact was found. These gaps prevent a cross-seed or causal efficacy claim.

After thirty-one papers, P033 adds per-epoch distribution drift, top-k churn, unique coverage, zero-exposure count, concentration, short-versus-endpoint reward and adaptive branch lineage. It adds no AutoSampling arm and changes no canonical hyperparameter. The formal test remains no replay, continuous, same-peak decay and dose-matched decay on one frozen selection and seed; if a simple fixed timing contrast fails, descriptive schedule drift alone is not evidence for a costly adaptive controller.

## P034 update: gradient conflict is local and role-specific

P034 (GEM, NeurIPS 2017) supplies the clean first-order interference test missing from magnitude-only proposals. For an SGD step `theta_new = theta - eta * g_i`, protected loss changes locally by `-eta * dot(g_protected, g_i)`. A negative dot product therefore identifies a locally harmful direction even when the candidate norm is large. This directly supports separate difficult-normal and weak-defect dot products, cosines and violation flags.

The transfer boundary is equally important. GEM constrains aggregate memories from previous tasks, assumes a local linear step and representative memory, and reports a one-seed multi-task stream with method-specific grid search. Aggregate gradients can hide identity-level tail harm, and repeated updates can depart from the first-order sign. Its official HEAD also post-dates publication with a QP PSD fix and contains a task-evaluation endpoint bug. It therefore adds a bounded gradient diagnostic plus finite-intervention validation, not a GEM arm, a static value formula, or permission to change the canonical Stage1 hyperparameters.

## P035 update: gradient geometry is not signed replay value

P035 (BADGE, ICLR 2020) separates uncertainty from diversity in a last-layer gradient embedding. For a hallucinated predicted label, each class block is `(p_i - I[y_hat=i]) z(x)`, and Proposition 1 shows that its norm lower-bounds the last-layer gradient norm under any possible true label. This makes it a useful low-cost representation of model uncertainty and candidate geometry, but it does not reveal whether replay lowers difficult-normal loss without harming weak defects.

The negative evidence is decisive for transfer. BADGE acquires unlabeled examples and retrains from scratch after every query; Stage1 replays already labeled identities inside one optimizer path. The paper reports that pure uncertainty can select redundant batches, but also that diversity can be harmful when the representation is poor and Coreset can lose to random. The binary-logistic argument is restricted to a low-margin linear region. None of these results establishes a static replay-value ranking or a Stage1 stop epoch.

The implementation audit exposes additional underspecification. Paper Algorithm 2 chooses the first `k-means++` center uniformly, while both the first public 2020 implementation and current code choose the maximum-norm point. The runner has no seed argument, does not preserve the external five-repeat job matrix, reads an epoch count without using it as a training cap, and provides no checkpoint, resume, environment lock, or artifact identity. Current HEAD uses a 2023 factorized distance speedup and is not the publication snapshot.

P035 therefore adds checkpoint-conditioned penultimate and last-layer embeddings, true-versus-hallucinated embedding disagreement, nearest-neighbor/center distances, effective rank, video concentration, center order, and RNG identity to the bounded diagnostic collector. These geometry fields remain separate from P034's signed candidate-to-normal-tail and candidate-to-weak-defect alignment fields. P035 adds no formal arm and changes no canonical Stage1 hyperparameter. A diversity transfer test is eligible only after the timing/dose mechanism is established and must hold selection rule, replay exposure, seed, and the canonical lock fixed.

## P036 update: gradient outliers are risk flags, not value signs

P036 (Outlier Gradient Analysis, ICML 2025) connects detrimental-sample influence to gradient-space outlier detection, but the connection is a hypothesis rather than an equivalence. Target-specific influence contains an evaluation gradient and an inverse-Hessian transformation in addition to the candidate gradient. Dropping those terms destroys the signed relation to Stage1's difficult-normal and weak-defect objectives. Raw gradient outlyingness can therefore describe leverage or rarity without identifying beneficial replay direction.

The paper's own synthetic table supplies the strongest negative evidence. L1/L2 outlier rules identify 98% of mislabeled points but retraining after trimming reaches 87% accuracy, below the 90% untrimmed model; iForest detects 96% and reaches 96%. Detector accuracy is not sufficient for downstream model value, and different detectors and trimming budgets win under different CIFAR noise regimes. Removal-and-retraining also differs fundamentally from repeated replay inside one optimizer trajectory.

The code audit further limits direct transfer. Vision scripts incompletely seed Python, NumPy, cuDNN, iForest and random projection, evaluate test data every epoch and report the maximum test accuracy. CIFAR-100 allocates an approximately 20.48 GB float64 last-layer gradient matrix before projection. The paper describes ImageNet-pretrained ResNet-34 fine-tuning, while the public source uses a custom randomly initialized CIFAR ResNet. The Llama benchmark uses known category blocks and broadcasts class-level scores to every identity in a class, so it does not establish individual-sample attribution.

P036 therefore adds `gradient_outlier_score`, detector identity, contamination, projection hash, local density and cross-checkpoint/seed stability to the bounded gradient collector. These remain risk/geometry fields and must be crossed with P034's separate signed alignment to difficult-normal and weak-defect probes and calibrated against finite replay interventions. P036 adds no formal arm, supplies no replay percentage or stop epoch, and changes no canonical Stage1 hyperparameter.

## P037 update: value is conditional on the selected set and representation

P037 (D2 Pruning, ICLR 2024) makes subset interaction explicit. The method diffuses difficulty over an embedding graph and then sequentially suppresses neighbors of each selected point. A candidate's score therefore changes with the representation, graph, already-selected prefix, suppression order and total budget. This supports modeling replay evidence as set- and state-conditioned rather than attaching a permanent scalar to one image.

The negative evidence prevents direct method transfer. D2 is a one-shot pruning method followed by retraining, not repeated replay inside one optimizer trajectory. It is not best at every pruning rate: at 90% pruning it trails CCS on CIFAR-100 and ImageNet, and at 95%-99.9% pruning the paper reports no consistent trend. Two propagation rounds mostly hurt. A qualitative example suppresses dolphin images because water-background similarity connects them to a selected landscape, showing that embedding coverage can discard label-relevant distinctions.

The code audit adds a material reproduction discrepancy. The paper equations use squared distance inside the forward and reverse exponentials, while the CIFAR implementation uses unsquared distance; a three-point numerical probe confirms different values. The untagged runner also lacks complete seed and dependency locks, repeatedly uses test performance for checkpoint selection, saves incomplete resume state, writes dynamics non-atomically, and exposes unseeded/approximate DataComp paths. These facts make paper-table parameters unsuitable for Stage1.

P037 therefore adds ordered selection position, local density, video/role neighborhood composition, suppression lineage, coverage radius, set effective rank, protected-role conflicts, and graph/set churn to bounded key-checkpoint diagnostics. These geometry fields must remain separate from P034's signed normal-tail and weak-defect alignments and from finite replay outcomes. P037 adds no D2 arm and changes no canonical Stage1 hyperparameter. A later same-budget diversity replacement is eligible only after the timing/dose mechanism is established and must be separately preregistered.

## P038 update: vector cancellation is real, but local influence is not a tail guarantee

P038 (Dataset Pruning, ICLR 2023) provides direct mathematical support for set-conditioned effects. It minimizes the norm of a vector sum rather than sorting individual influence norms, so two individually large effects can cancel and many modest aligned effects can accumulate. This supports collecting norm-of-sum, sum-of-norms, cancellation ratio, pairwise direction, and role-specific aggregate alignment instead of inventing another scalar sample score.

The guarantee does not transfer directly. It approximates finite subset removal by a sum of infinitesimal leave-out influences around an empirical minimizer, assumes a positive-definite Hessian, and bounds average expected loss rather than an FN-constrained frontier. Its epsilon notation is inconsistent: the algorithm constrains the sum after `1/n` scaling, while Theorem 1 constrains the unscaled sum and then derives an `epsilon/n` term. The proof also leaves the supremum domain and expected-test-gradient bound implicit. These issues require explicit units and finite-intervention residuals in Stage1.

The empirical protocol does not resolve the uncertainty. Figures and the NAS table report no seed count, standard deviations, confidence intervals, or error bars; simulated-annealing schedule, seed, budget, and convergence are unspecified; no independent validation protocol or final code package is available. Removal plus retraining from scratch is also a different intervention from repeated replay in one optimizer path.

P038 therefore adds vector-sum versus sum-of-norms, cancellation ratio, pairwise cosine/sign distribution, separate normal-tail and weak-defect aggregates, influence-scaling identity, and finite one-step/short-horizon approximation residuals. It adds no pruning arm, no numeric threshold, and no training hyperparameter. The causal campaign remains no replay, continuous, same-peak decay, and cumulative-dose-matched decay under one canonical lock.
