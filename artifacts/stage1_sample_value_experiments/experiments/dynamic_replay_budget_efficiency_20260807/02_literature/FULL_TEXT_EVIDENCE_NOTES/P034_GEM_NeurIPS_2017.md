# P034 - Gradient Episodic Memory for Continual Learning

## Identity

- Paper ID: P034
- Authors: David Lopez-Paz and Marc'Aurelio Ranzato
- Venue and year: NeurIPS 2017
- Published page: https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
- Main PDF: `source_papers/Gradient_Episodic_Memory_2017.pdf`, SHA256 `9AA6A5F73220449CBE8B2C9BD4BDC3F5407AC49295C4CA1F242A36E79E34B763`
- Supplement PDF: `source_papers/Gradient_Episodic_Memory_2017_appendix.pdf`, SHA256 `2A87C971E370FE003E8E854AED158F34CE2FB8EA06C690AD531EE12AF10AA7BE`
- Official code: https://github.com/facebookresearch/GradientEpisodicMemory
- Audited code commit: `34c6b8e9a0607db7567301c48b727430d20bee7e`, an untagged post-paper HEAD dated 2018-10-22

## Reading Coverage

- Main manuscript: 10/10 pages read, including Equations 1-11, Algorithm 1, Tables 1-3, Figure 1, experiments, stated limitations, and references.
- Supplement: 7/7 pages read, including the complete hyperparameter grids and all reported task-by-task evaluation matrices.
- Visual verification: all 17 unique pages inspected at original detail under `audit/visual_checks/P034_GEM_NeurIPS_2017/`.
- Code audit: README, runner, main loop, GEM implementation, metric implementation, complete commit history, and paper-to-HEAD changes inspected. Syntax compilation passed; a paper training run was not attempted because the legacy PyTorch/quadprog environment is unpinned.
- Public reviews: the official NeurIPS review page was checked as scope context; reviewer opinion is not used as efficacy evidence.

## Research Question

GEM asks how a learner can update on a current task without increasing loss on episodic memories from earlier tasks. Its unit is a task-level memory constraint in sequential multi-task learning, not a static value assigned to one image.

This is relevant to Stage1 because replaying difficult normal samples may reduce their loss while increasing loss on weak defects. GEM provides a local gradient test for that interference. The task, stream, and objective are nevertheless different enough that GEM cannot be imported as a Stage1 method without a separate causal experiment.

## Core Mathematics

For a current gradient `g` and a previous-memory gradient `g_k`, an SGD step is:

```text
theta_new = theta - eta * g
```

The first-order change in previous-memory loss is:

```text
L_k(theta_new) - L_k(theta) ~= -eta * dot(g_k, g)
```

Therefore:

```text
dot(g_k, g) >= 0  => locally non-increasing previous-memory loss
dot(g_k, g) <  0  => locally interfering update
```

When a constraint is violated, GEM projects `g` onto the closest vector satisfying all previous-task half-space constraints:

```text
minimize_g_tilde  0.5 * ||g - g_tilde||^2
subject_to        dot(g_tilde, g_k) >= 0 for every previous task k
```

The dual reduces the QP dimension from the number of network parameters to the number of previous tasks. This is a direction constraint, not a gradient-magnitude score and not a proof of endpoint value.

## Experimental Protocol

- Twenty sequential tasks are constructed from MNIST permutations, MNIST rotations, and class-incremental CIFAR-100.
- Each example is observed once in the main protocol; Table 3 separately tests one, two, and five passes per task.
- MNIST uses two hidden layers of 100 ReLU units. CIFAR-100 uses a reduced-width ResNet-18 with task-specific output restrictions.
- Plain SGD and batch size 10 are used. Learning rate and method-specific parameters are selected from broad grids and differ by method and dataset.
- The public runner fixes `seed=0`. The paper and supplement report point estimates without cross-seed uncertainty.
- The public GEM preset uses 256 memories per task for 20 tasks, corresponding to the paper's total memory budget 5,120, and margin 0.5.
- These settings are literature context only. None may change the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. Equations 6-11 give a clear local distinction between a large beneficial gradient and a large interfering gradient. Magnitude alone cannot provide this distinction.
2. The constraint is evaluated against task-memory aggregate gradients. Opposing sample gradients can cancel inside an aggregate, so a satisfied task constraint does not guarantee that every weak-defect identity is protected.
3. The first-order sign guarantee assumes a sufficiently local step and a representative memory. It can fail after finite multi-step replay, momentum, changing batch context, augmentation, optimizer-state evolution, or a nonrepresentative tail probe.
4. Table 3 shows that additional passes strongly worsen backward transfer for memoryless sequential models. This supports cumulative-exposure measurement, but the multi-task stream is not Stage1's iid base training plus replay overlay.
5. GEM maintains substantially better backward transfer in the reported settings, but the evidence is based on one published trajectory per configuration with no seed distribution.
6. CIFAR accuracy rises with memory size in Table 2, but this is one dataset, one selected configuration, and no uncertainty. It does not establish monotonic replay benefit in Stage1.
7. Hyperparameters are selected separately for every method. The comparison does not isolate gradient constraints under one canonical training recipe.
8. The tuning target and a separate validation role are not documented. Test tasks are evaluated repeatedly throughout training, so the reported best-grid protocol lacks Stage1's blind-holdout separation.
9. Official code stores a ring buffer per task and computes one full-model backward pass per previous task at every update. The cost does not scale to 120,000 samples times 200 epochs.
10. The code constrains only earlier tasks, not competing strata within the same current task. Stage1 normal-tail and weak-defect roles therefore require explicitly separate probe gradients.
11. The current repository HEAD post-dates the paper and adds a `1e-3` diagonal term to fix a non-positive-semidefinite QP failure. There is no paper tag or locked environment.
12. The public evaluation loop slices to `x.size(0)-1` with an exclusive endpoint, omitting the final sample of a multi-sample task evaluation. Published code is therefore not a flawless executable ground truth.
13. The runner provides only seed 0, no resume contract, no RNG-state artifacts, no atomic completion sidecars, and no automated tests.

## Direct Support For Stage1

1. Split gradient influence into magnitude and direction. A sample with large norm and negative weak-defect dot product is high-impact but harmful for the protected tail.
2. At key checkpoints, compute separate last-layer probe gradients for difficult-normal loss and weak-defect loss. Never collapse them into one weighted vector before recording the two signs.
3. Record raw dot product, cosine, candidate norm, each probe norm, sign, violation flag, and the angle between the two probe gradients.
4. Record both aggregate-tail and identity-stratified probe gradients because aggregate cancellation can hide concentrated harm.
5. Measure whether the local sign predicts an independent finite intervention and the final paired safety frontier. A local gradient sign that does not survive this test is only a diagnostic.
6. Measure sign persistence across checkpoint, seed, augmentation draw, and batch context. Conditional instability is part of the result, not noise to discard.
7. Preserve cumulative replay exposure and optimizer steps because repeated locally acceptable updates can still yield a harmful nonlinear endpoint.
8. Keep a no-replay arm and same-selection timing/dose controls. GEM does not identify replay benefit relative to ordinary base training.

## What It Does Not Support

1. Defining high-value samples by gradient norm.
2. Treating one positive dot product as sufficient evidence of endpoint value.
3. Treating a small-gradient sample as necessarily useless.
4. Assuming an aggregate memory gradient protects the most dangerous weak-defect identities.
5. Adding a full GEM projection arm to the first Stage1 campaign.
6. Claiming the continual-learning result transfers directly to one-task binary tail optimization.
7. Importing GEM's learning rates, batch size, memory budget, margin, optimizer, architecture, epoch protocol, or seed.
8. Changing any Stage1 canonical hyperparameter.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, collect at epochs 120, 140, 150, and 160 for the bounded gradient pilot:

- candidate last-layer gradient norm and squared norm;
- difficult-normal aggregate probe gradient norm;
- weak-defect core and weak-defect buffer probe gradient norms separately;
- candidate-to-normal and candidate-to-defect raw dot products and cosines;
- normal-defect probe dot product and cosine;
- violation flags for each protected role;
- per-identity or small-stratum alignment dispersion, not only aggregate alignment;
- checkpoint, model-state, optimizer-state, batch-context, augmentation, and RNG hashes;
- finite-intervention delta on identity-disjoint OOF/val_op probes;
- cross-checkpoint and cross-seed sign persistence;
- canonical lock SHA256 on every gradient artifact.

## Concrete Experiment Consequence

P034 adds no formal training arm. It strengthens the bounded gradient pilot attached to the frozen causal schedule experiment:

```text
same selection + same seed + canonical hyperparameters
continuous vs same-peak decay vs dose-matched decay vs no replay
```

The pilot first tests whether role-separated gradient conflicts predict finite and endpoint tail harm. Only if that measurement is reproducible and adds information beyond exposure and trajectory fields should a later evidence card propose a constrained guard intervention.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for local gradient interference, projection semantics, memory representativeness, and exposure/forgetting cautions
- Replication-depth eligibility: yes, with explicit caveats for the post-paper code revision, unpinned environment, one-seed runner, and evaluation bug
- Direct support for static replay ranking: no
- Direct support for role-separated gradient diagnostics: yes
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-08
