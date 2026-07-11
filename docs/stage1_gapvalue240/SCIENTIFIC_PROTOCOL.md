# Frozen scientific protocol

The project tests whether sample rankings derived from 200-epoch OOF dynamics identify replay samples with causal training value. GapCritical is a candidate proxy, not assumed truth. Each treatment is tested against an independent global random control and a hardness-matched random control with identical training seed, replay budget, optimizer steps, and replay exposure.

The matrix contains 19 Phase-A conditions, 6 Phase-B guard conditions, and 5 new confirmation seeds for the preregistered `GapCritical-Strict B3000` primary condition. This produces 80 triads and exactly 240 validated runs.

Phase A uses additive replay. Phase B keeps 3,000 replay slots fixed and replaces a nested suffix of GapCritical normal replay with 5%, 10%, or 20% defect guard samples. Base training samples are never removed.

Method selection uses val_cal and val_op. The historical 120k test is treated as `development_benchmark_120k`, not a blind test. Final claims require a new blind holdout or external test.

Primary metrics are tie-safe `TN_at_FN95` and `FN_at_TN68253`. Gap metrics explain mechanism but cannot establish replay value by themselves.
