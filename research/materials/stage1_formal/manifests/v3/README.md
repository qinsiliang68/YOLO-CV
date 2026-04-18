# v3 Sampling Output (Two-stage Balanced)

Generated: 2026-04-18T16:12:35.973862+00:00
Seed: 20260606

## Design Philosophy

**Train-side intervention, eval-side observation.**

### Stage 1 (essay3, binary gate) — 12,000 train

- 6,000 defect (1000/class × 6) + 6,000 Normal
- 1:1 pos:neg balance
- Each main class guaranteed 1,000 (vs v1 natural's PF=208/DE=402)

### Stage 2 (essay4, object detection, future) — 8,000 train

- 6,000 defect (same pool as Stage 1) + 2,000 Normal
- 1:3 pos:neg (Normal as background)
- normal_stage2 is STRICT SUBSET of normal_stage1 (first 2,000 after shuffle)

## Val / Test (copied from v1)

Natural distribution preserved:
- **test**: mirrors deployment; only interpretation for real-world performance
- **val**: when train is balanced, natural val is the only unbiased ruler
  that exposes training-induced bias during epoch selection

## Integrity

- v3 defects + normal drawn from pool EXCLUDING v1 val/test ids
- 7 assertions pass: size, subset, disjoint, uniqueness, disk, seed

## WARNING — test_ids.csv

**test_ids.csv must NOT be read during development.**
See repo-root LEAKAGE_AUDIT.md L1-8.
