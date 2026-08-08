# P037 - D2 Pruning

## Identity

- Paper ID: P037
- Full title: D2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data Pruning
- Authors: Adyasha Maharana, Prateek Yadav, and Mohit Bansal
- Venue and year: ICLR 2024
- Official proceedings page: https://proceedings.iclr.cc/paper_files/paper/2024/hash/f646a4c970799f789e031b63161d018d-Abstract-Conference.html
- Local PDF: `source_papers/D2_Pruning_ICLR_2024.pdf`, SHA256 `A27D3E6D5D9F62AD0D895054B6889ED185E7B0A290EF700773AEED31DC12CFF3`
- Official code: https://github.com/adymaharana/d2pruning
- Audited code commit: `bf1ea46b22cc79549df421fcd92c9cd2d3408008`, dated 2023-10-13

## Reading Coverage

- Main paper and appendices: 20/20 pages read, including Algorithm 1, the message-passing equations, Tables 1-13, Figures 1-5, difficulty-score comparisons, pruning-rate studies, graph-parameter studies, embedding-source studies, ablations, DataComp experiments, limitations, and qualitative failure examples.
- Visual verification: all 20 pages inspected at original detail under `audit/visual_checks/P037_D2Pruning_ICLR_2024/`; no blank, missing, clipped, or malformed page was found.
- Code audit: complete four-commit public history, vision training and selection paths, graph sampler, DataComp scoring/selection path, NLP runner, README, and dependency declarations inspected.
- Static execution check: all 53 Python files parsed successfully with the current Python AST parser without importing dependencies or creating bytecode.
- Reviewer evidence limitation: the OpenReview landing page and API were blocked by an access challenge during this audit. No reviewer claim is used; the official ICLR proceedings paper and public code are the evidence sources.
- Training execution limitation: no benchmark was rerun because the public repository lacks the publication environment, deterministic run manifests, complete seeds, and exact paper-table command set.

## Research Question

The paper asks how to choose a one-shot coreset that balances example difficulty with coverage of the embedding distribution. Stage1 asks whether replaying a fixed labeled set at a particular model state, dose, and phase improves difficult-normal filtering while protecting weak defects.

The direct lesson is not that D2-selected examples should be replayed. It is that the marginal meaning of one sample depends on which samples have already been selected, the representation, graph neighborhood, budget, and suppression order. This supports set-conditioned diagnostics and redundancy accounting. It does not establish a static scalar `V(x)`, a replay schedule, or a Stage1 training arm.

## Core Method

The method initializes each graph node with a difficulty score and connects examples through embedding-space nearest neighbors. A forward message-passing step raises a node score when nearby examples are difficult. The paper writes a distance-weighted contribution of the form:

```text
message(j -> i) = exp(-gamma_f * distance(i,j)^2) * difficulty(j)
```

After selecting a high-scoring node, reverse message passing suppresses nearby nodes:

```text
score(j) <- score(j) - exp(-gamma_r * distance(selected,j)^2) * score(selected)
```

Selection is therefore sequential. A candidate's score after step `m` is conditional on the ordered prefix selected in steps `1..m-1`. The scientific object is closer to:

```text
V(x | representation, graph, selected_prefix, budget, difficulty_score)
```

than to an image-only scalar. The Stage1 transfer is a set-level diagnostic: record local density, neighborhood role composition, selection order, suppression lineage, coverage radius, effective rank, and protected-tail conflicts. These fields remain descriptive until a same-selection intervention changes the raw safety frontier.

## Experimental Protocol

- CIFAR-10 and CIFAR-100 use ResNet-18 and compare several difficulty scores and coreset baselines over multiple pruning rates.
- ImageNet experiments use ResNet-50 and pretrained embeddings; the chosen graph parameters vary by dataset and pruning ratio.
- The paper reports low-to-medium pruning as the strongest regime for D2. At 90% pruning, D2 is below CCS on CIFAR-100 (`56.9` versus `57.3`) and ImageNet (`55.6` versus `57.3`).
- The authors state that at extreme 95%-99.9% pruning rates there is no consistent trend across methods.
- Two message-passing rounds generally hurt, consistent with oversmoothing or over-diffusion.
- Representation choice materially changes results; graph geometry is not an invariant property of the image.
- A qualitative failure shows dolphin images suppressed because their water backgrounds make them similar to a selected landscape image. Similarity can erase label-relevant diversity.
- On DataComp, D2 averages `17.0`, below the originally reported CLIP-score baseline `17.3` but above the authors' reproduced CLIP score `16.0`; T-MARS plus D2 reaches `18.8`.
- The main significance calculation bootstraps many examples rather than reporting independent training-seed replications for the complete selection-and-training procedure.
- Algorithm 1 proposes selecting parameters on validation data and evaluating the test set once. The released vision runners do not preserve this split discipline.
- All numeric settings are literature context only and may not alter the Stage1 canonical `yolo11l` hyperparameter lock.

## Direct Support

1. Sample value can be set- and prefix-conditioned. Reverse suppression means adding one identity changes the scores of its neighbors.
2. Difficulty alone and diversity alone are incomplete. Their interaction depends on representation, neighborhood scale, and budget.
3. Redundancy should be measured in more than one space. Background similarity can suppress semantically distinct examples.
4. Extreme budgets can change method ordering. A rule successful at one replay percentage cannot be treated as budget-invariant.
5. More message passing is not automatically better; iterative aggregation can oversmooth and lose useful local distinctions.

## Non-Support And Negative Evidence

1. This is one-shot pruning followed by retraining from scratch, not repeated labeled replay within a shared optimizer trajectory.
2. The paper does not optimize FN-constrained safety frontiers, difficult-normal correction, or weak-defect protection.
3. D2 is not best at every dataset or pruning rate and has no consistent extreme-pruning advantage.
4. The published significance analysis does not resolve same-selection cross-seed reversal of the complete pipeline.
5. The graph is representation-dependent; a poor embedding can turn nuisance similarity into harmful suppression.
6. The method does not provide a signed target-gradient test. High graph score can still conflict with the protected defect tail.
7. No evidence supports importing its `k`, `gamma_f`, `gamma_r`, pruning rates, optimizer, or augmentation into Stage1.

## Code Audit

- Repository HEAD is untagged and has four commits. No publication release tag was found.
- `requirements.txt` pins only Transformers and scikit-learn versions; Python, PyTorch, torchvision, SciPy, FAISS, CUDA, and several other dependencies are not locked.
- `GraphDensitySampler` accepts a seed argument but does not use it, and vision train scripts expose no complete seed-control path.
- The paper equations use squared distance inside the exponential. CIFAR code uses `exp(-d)` and `exp(-gamma*d)`. A three-point numerical probe confirmed the mismatch:

```text
code_forward              = 0.270670566473
paper_equation_forward    = 0.036631277777
code_reverse_factor       = 0.367879441171
paper_equation_reverse    = 0.135335283237
```

- Vision runners repeatedly evaluate the test set and save a best-test checkpoint instead of enforcing an independent validation-selection gate.
- Checkpoints contain only model and epoch; optimizer, scheduler, scaler, RNG, sampler, selected-prefix, graph, and full resume state are absent.
- Training dynamics are retained in memory and written once at the end through a non-atomic pickle path.
- The non-coreset README path reaches a datetime-versus-float subtraction in `train.py` after checkpoint writing.
- README commands do not encode the per-budget parameters needed to reproduce the reported tables; NLP commands are listed as forthcoming.
- DataComp code uses unseeded `random.sample`, unordered multiprocessing, approximate FAISS without persisted index state, an unexplained `offset=0.115`, and direct non-atomic `numpy.save` outputs.
- All 53 Python files pass syntax parsing, but no automated tests or complete experiment manifest were found.

## Stage1 Transfer Boundary

P037 does not justify a D2 replay arm. It justifies measuring whether an already-defined replay set is redundant, poorly covering relevant states, or suppressing protected-role neighbors. The minimum bounded fields are:

```text
embedding_model_hash
embedding_checkpoint_hash
graph_builder_and_rng_hash
candidate_local_density
same_video_neighbor_fraction
same_role_neighbor_fraction
cross_role_neighbor_fraction
ordered_selection_position
suppression_parent_id
suppression_strength
distance_to_selected_prefix
coverage_radius
set_effective_rank
tail_probe_coverage
protected_role_conflict_count
graph_churn_across_checkpoint
set_churn_across_seed
```

These must be accompanied by signed difficult-normal and weak-defect gradient alignment from P034, realized exposure, and finite intervention outcomes. Geometry alone remains a risk/coverage channel, not a value label.

## Current Experiment Consequence

- Add graph/set interaction fields to the field inventory and bounded key-checkpoint collector.
- Keep selection identity, order, embedding source, graph parameters, RNG, and budget hashes explicit.
- Do not blend graph score into an arbitrary weighted value formula.
- Do not add a D2 training arm to the first timing/dose cycle.
- Do not change any canonical training hyperparameter.
- The formal causal core remains no replay, continuous replay, same-peak decay, and cumulative-dose-matched decay on one frozen selection under one canonical lock.
- A later diversity transfer test becomes eligible only if timing/dose first shows stable benefit and the new test uses a preregistered same-budget, same-exposure, same-seed set replacement.

## Bottom Line

D2 gives strong evidence that sample utility is conditional on the selected set, representation, order, and budget. It also supplies important failures: extreme pruning has no consistent winner, extra propagation can hurt, and nuisance-background similarity can suppress relevant identities. For Stage1, the correct transfer is a graph/set diagnostic and a future falsifiable diversity replacement test, not a new score, replay arm, or hyperparameter.
