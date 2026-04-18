# v2 Sampling Output (Sub-class Balanced Train, 1:6)

Generated: 2026-04-18T07:05:35.442959+00:00
Seed: 20260606

## Strategy

- train (24,500): 7 equal buckets, 3500 each
  - Normal (y=0): 3,500
  - PF/DE/FS/RB/AF/OB (y=1): 3,500 each under rarity priority
  - Normal:defect_total = 1:6
- val_cal / val_op / test: copied from v1 (natural distribution)

## Purpose

- essay3 (binary gate): head-to-head comparison against v1 natural training
- essay4 (object detection): primary balanced training data for 6-class localization

## Integrity

- v2 train drawn from pool excluding v1 val/test ids
- v2 train disjoint from v1 val_cal/val_op/test (asserted)
- Same seed as v1 (20260606); v1 val/test IDs identical via copy

## WARNING - test_ids.csv

**test_ids.csv must NOT be read during development.**
See repo-root LEAKAGE_AUDIT.md L1-8.
