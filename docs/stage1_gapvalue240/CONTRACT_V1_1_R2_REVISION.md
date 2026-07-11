# R2 contract revision: v1 → v1.1

The archived v1 required R2 to be completely disjoint from treatment while simultaneously remaining in the same eligible pool, fold, and dynamic bucket and achieving max absolute SMD ≤ 0.10 on three hardness variables. The supplied real table proves this is mathematically impossible for multiple top-tail methods, including the primary B3000 condition.

The runnable v1.1 makes the smallest explicit correction:

1. R1 remains fully disjoint.
2. R2 first constructs the largest possible same-fold, non-treatment nearest-neighbor control.
3. If the SMD gate remains impossible, pairs are replaced by their corresponding treatment samples until the gate is satisfied.
4. Every forced overlap is recorded, including overlap count, rate, Jaccard, unique contrast, fallback level, SMD, fold balance, and dynamic-bucket balance.
5. Conclusions must disclose effective unique contrast. High-overlap R2 is a very strong but lower-power mechanism control, not a substitute for R1.

The original v1 YAML and feasibility audits are retained unchanged for provenance.
