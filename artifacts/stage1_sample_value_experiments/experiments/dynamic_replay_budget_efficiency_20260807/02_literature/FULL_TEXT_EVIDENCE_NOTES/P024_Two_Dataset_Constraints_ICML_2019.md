# P024 - Training Well-Generalizing Classifiers for Fairness Metrics and Other Data-Dependent Constraints

## Identity

- Paper ID: P024
- Authors: Andrew Cotter, Maya Gupta, Heinrich Jiang, Nathan Srebro, Karthik Sridharan, Serena Wang, Blake Woodworth and Seungil You
- Venue and year: ICML 2019, PMLR 97
- Official page: https://proceedings.mlr.press/v97/cotter19b.html
- Main paper: `source_papers/Training_Well_Generalizing_Classifiers_ICML_2019.pdf`, SHA256 `1D08A44C7CF1CA67F8B91C49C62F0D4BFECEB751A996014D5871966E869C10DD`
- Supplement: `source_papers/Training_Well_Generalizing_Classifiers_ICML_2019_Supplemental.pdf`, SHA256 `28DF0B03DAF51380A3737719D0CF227FFE48EF5EF2991522944FF38CDB720CA4`
- Official code: https://github.com/google-research/tensorflow_constrained_optimization at commit `723d63f8567aaa988c4ce4761152beee2b462e1d`

## Reading Coverage

- Main paper: 9/9 pages read, including definitions, both theoretical algorithms, both practical algorithms, bounds, all datasets, results and limitations.
- Supplement: 15/15 pages read, including proofs, Neyman-Pearson and rate-constraint examples, shrinking, simulation, Lagrangian baseline and additional derivations.
- Code: current library, setup, README, candidate shrinking, split-rate context, relevant unit tests, the full Communities notebook and its history inspected.
- Executable checks: all 50 Python files syntax-compiled; the TensorFlow-free candidate solver reproduced the official expected distribution and indices. Full tests were not run because TensorFlow is absent and the repository does not provide a reproducible dependency lock.
- Visual verification: all 24 PDF pages and seven contact sheets under `audit/visual_checks/P024_Two_Dataset_Constraints_ICML_2019/` were inspected.

## Research Question

The paper studies population problems of the form:

```text
minimize_theta  E[l0(x; theta)]
subject to      E[li(x; theta)] <= 0, i=1,...,m.
```

When constraints are estimated on the same data used to fit a high-capacity model, satisfying empirical constraints need not imply population feasibility. The proposed remedy is to let the model player learn parameters on one dataset and let the constraint player choose penalties on an independent validation dataset.

For Stage1, the transferable question is whether the difficult-normal objective and weak-defect safety constraint should be estimated on separate, identity-frozen streams so that a replay policy cannot overfit the same tail used to declare it safe.

## Core Method

With `lambda` in the `(m+1)`-simplex, the two players use different empirical functions:

```text
L_theta(theta, lambda)
  = lambda_1 * mean_train(l0)
  + sum_i lambda_(i+1) * mean_train(proxy_li)

L_lambda(theta, lambda)
  = sum_i lambda_(i+1) * mean_validation(original_li).
```

The theta player minimizes the objective and differentiable proxy constraints on `S_train`. The lambda player maximizes original constraint violations on independent `S_validation`. The resulting game is non-zero-sum.

The theoretical algorithms either discretize the multiplier space and call an optimization oracle or assume strong convexity for nested gradient updates. They return a stochastic classifier over iterates. A linear-program shrinking step can reduce support to at most `m+1` candidates.

The practical algorithms replace the idealized updates with TensorFlow proxy-Lagrangian or Lagrangian-style optimization. The paper explicitly states that these practical variants do not inherit the formal guarantees.

The supplement's Neyman-Pearson example minimizes false-positive rate subject to false-negative rate at most 0.1. Indicator rates define the actual objective and constraint; hinge upper bounds provide differentiable optimization proxies. This matches the structure, but not the numeric level, of Stage1's asymmetric objective.

## Experimental Evidence

- Real-data experiments cover Communities and Crime, proprietary Business Entity Resolution, Adult and COMPAS.
- One-dataset and two-dataset conditions receive the same total data. The two-dataset condition splits its training allocation between theta and lambda roles; the baseline uses their union for both roles.
- Models include linear, a calibrated random-tiled lattice model, and one-hidden-layer networks with 10 or 100 units. One hundred evenly spaced iterates are retained and shrunk by linear programming.
- Main real-data results average 100 random train/validation/test splits. The table does not report standard deviations or confidence intervals alongside these means.
- Constraint generalization generally improves under the two-dataset design, often at a cost in objective error. Business Entity Resolution is a visible failure/cost case: constraint behavior is comparable while error is worse.
- The simulation averages ten random splits, constrains recall to at least 97%, varies kernel width and compares linear with 5-, 10- and 100-unit networks. The benefit becomes more apparent as overfitting capacity rises, but testing accuracy is slightly worse.
- The paper warns that independent validation does not eliminate hyperparameter overfitting and that its strongest guarantees require assumptions not met by the practical neural experiments.

## Code Reproduction Audit

1. The official repository is now archived. The local head is from July 2021, not an immutable ICML-2019 release.
2. The linked Communities notebook is a tutorial, not the paper's four-dataset and 100-split pipeline. It runs one fixed split and one dataset.
3. The tutorial uses a 10-unit neural network for Communities, while the paper reports a linear model for that dataset.
4. Binary-label thresholding and protected-group quantiles are computed before train/test splitting. Test-distribution information therefore affects preprocessing.
5. Labels are encoded as `0/1`. Overall negatives use `label <= 0`, but protected-group negatives use `label < 0`. The latter is empty for this encoding, so the tutorial's group constraint calculation is not trustworthy. History shows this predicate in the initial tutorial and in the later split-context conversion.
6. The split-rate-context conversion was committed in October 2019, after the conference. The exact experiment code underlying the paper is not identified.
7. The tutorial fixes seeds for the outer split, inner role split, model and minibatch order, but shuffles the minibatch permutation only once. Wrapped epochs repeat the same order.
8. Candidate snapshots are evaluated on both train and test every loop. The linear-program selection uses train metrics, but repeated test visibility prevents a blind workflow.
9. Data are fetched over HTTP without a hash. Results remain in notebook memory, with no checkpoint, resume, atomic artifact, completion sidecar or configuration provenance.
10. The package has 19 unit-test files. They were not executed because TensorFlow is absent and `setup.py` provides only broad lower bounds. The TensorFlow-independent shrinking implementation did pass a direct numerical reproduction of its known test case.

## Evidence Limitations

1. Fairness and generic rate constraints are not selected-sample replay.
2. The paper asks whether constraints generalize, not whether replay timing or cumulative exposure changes a raw `FN=0-95` safety frontier.
3. Theoretical guarantees apply to oracle/discretized or strongly convex algorithms, not the practical deep-network routine used in the experiments.
4. The final predictor is generally a stochastic mixture over iterates; Stage1 deploys one checkpoint and threshold.
5. Independent constraint data reduce effective model-fitting data and can increase objective error.
6. Results do not establish any Stage1 replay percentage, epoch boundary, guard ratio or optimizer setting.
7. The linked tutorial contains a label-predicate defect and is not sufficient to reproduce the paper tables.
8. Reported real-data means omit paired uncertainty in the main table despite 100 random splits.

## Direct Support For Stage1

1. Keep difficult-normal benefit and weak-defect safety as separate objective and constraint quantities.
2. Derive replay candidates from OOF/training information, tune constraint behavior on an identity-frozen `val_op` probe, and keep blind holdout closed until the policy is frozen.
3. Record both differentiable proxy quantities and original business rates. A surrogate improving does not prove the raw FN constraint generalizes.
4. Measure constraint generalization gaps separately from objective improvements at every saved checkpoint.
5. Ensure the weak-defect probe is large enough for stable tail estimates; tiny tail subsets can make constraint updates or diagnostics noisy.
6. Treat stochastic checkpoint mixtures as a diagnostic frontier tool only unless deployment semantics explicitly permit randomized inference.

## What It Does Not Support

1. Replacing the canonical Stage1 optimizer with a proxy-Lagrangian optimizer.
2. Importing the paper's ADAM settings, batch sizes, network widths, split fractions or constraint levels.
3. Claiming that an OOF or validation constraint is automatically satisfied on blind data.
4. Selecting a schedule or checkpoint after repeatedly inspecting blind test predictions.
5. Treating the current official notebook as a faithful reproduction of the paper.
6. Treating one blended score as a substitute for separate normal-tail benefit and weak-defect harm.

## Stage1 Field Contract

Under the exact hash-locked 240-run hyperparameters, collect for every epoch:

- difficult-normal proxy loss and original raw-score tail outcome;
- weak-defect proxy loss, raw FN count/rate and constraint violation;
- theta-role and constraint-probe sample identities, sizes and overlap hashes;
- train-to-probe constraint generalization gap;
- realized replay exposure, optimizer steps and schedule area;
- objective and constraint gradient direction on fixed probes when sampled;
- candidate-checkpoint identity and whether any result was used for discovery, calibration or blind confirmation.

At 120, 140, 150, 160, 180 and 200, save the existing heavy artifacts and evaluate original non-differentiable rates on the fixed operational probe. Do not alter `yolo11l`, batch 128, image size 224, workers 4, optimizer resolution, learning schedule, augmentation or any other canonical parameter.

## Concrete Experiment Consequence

P024 does not add a new formal training arm. It changes the validation architecture of the existing timing-and-dose experiment:

```text
selection discovery: OOF/train only
policy calibration: fixed val_op objective and weak-defect constraint streams
confirmatory outcome: paired unseen seeds
blind holdout: opened only after the rule is frozen
```

For each no-replay, continuous, same-peak-decay and cumulative-dose-matched arm, report both normal-tail benefit and weak-defect constraint generalization. A policy is not safe merely because its proxy loss or calibration-set FN improves.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for independent constraint estimation, proxy-versus-original separation and constraint-generalization auditing
- Replication-depth eligibility: yes, because main, supplement, code, history, tests, notebook behavior and a numerical candidate-solver path were audited
- Direct support for static replay ranking: no
- Direct support for a new optimizer or constrained-training arm: no
- Direct support for numeric Stage1 schedules or percentages: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
