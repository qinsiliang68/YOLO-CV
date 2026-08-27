# Research Questions

Each counted paper must directly support, refute, or complicate at least one question.
Keyword overlap alone is not relevance.

| ID | Question |
|---|---|
| RQ1 | Which training-dynamics signals distinguish already learned, learnable, slow-learning, unstable, and persistently unlearnable samples? |
| RQ2 | Does reducible or residual learnability add information beyond current loss, confidence, margin, gradient norm, forgetting, or uncertainty? |
| RQ3 | Can sample-gradient direction or validation-target alignment distinguish beneficial influence from large but harmful influence? |
| RQ4 | Which reliability signals separate hard-clean samples from mislabeled, ambiguous, corrupted, out-of-distribution, or irreducible samples? |
| RQ5 | How do diversity, source coverage, video/near-duplicate constraints, and set-conditional marginal value affect finite-budget selection? |
| RQ6 | How do replay timing, refresh frequency, unique count, repeat intensity, cumulative exposure, and optimizer steps change intervention utility? |
| RQ7 | Which global-random, method-matched-random, current-loss, no-replay, seed, checkpoint, and multiple-comparison controls are needed to claim stable improvement? |
| RQ8 | How should Neyman-Pearson, partial-AUC, FN95-local tails, initialization, order, and training-stage dependence define target utility and explain value reversal? |

The mapping to the Stage1 concept contract is sequential rather than weighted:

1. `Q`: reliability gate.
2. `R`: residual/reducible learnability stratum.
3. `A`: direction relative to an independent FN95-local target.
4. `D`: set-level coverage and redundancy constraint.
5. Real paired replay intervention: the only utility evidence.
