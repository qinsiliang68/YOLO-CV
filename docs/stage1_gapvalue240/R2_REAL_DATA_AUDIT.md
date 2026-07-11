# R2 v1.1 real-data audit

The executable matcher was run against the supplied 120,000-row value table for the primary `GapCritical-Strict` treatment.

| Budget | Runtime in packaging audit | Forced overlap | Effective unique contrast | max abs SMD |
|---:|---:|---:|---:|---:|
| 600 | 1.66 s | 537 / 600 = 89.5% | 10.5% | 0.09836 |
| 3000 | 10.15 s | 2739 / 3000 = 91.3% | 8.7% | 0.09923 |

This confirms two facts:

1. v1.1 is computationally executable and satisfies the frozen SMD gate;
2. the primary treatment lies so far in the hardness tail that a near-disjoint, tightly matched R2 does not exist.

Therefore R2 must be interpreted as a low-power, near-treatment mechanism control. R1 remains the fully disjoint random baseline. Reports must include effective unique contrast and may not describe R2 as an independent sample set.

Machine-readable evidence is stored in `precomputed_direct_assets/R2_V1_1_REAL_A02_AUDIT.json`.
