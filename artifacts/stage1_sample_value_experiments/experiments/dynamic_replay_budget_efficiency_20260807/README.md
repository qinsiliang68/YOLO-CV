# dynamic_replay_budget_efficiency_20260807

This experiment is parallel to the prior 40-run, 120-run, and 240-run/OOF work.
It does not replace or mutate those evidence packages.

Primary question: can replay schedule control and weak-defect protection make a fixed
sample budget produce stable benefit across unseen training seeds?

The current phase is `RESEARCH_AUDIT_HELD`. Earlier v2 assets remain preserved
as historical candidate assets, but they are not authorized for owner canary,
engineering gate, pilot release, or formal training while the evidence audit is
open. The preserved assets are:

- `03_preregistration_v2/`: 30 registered seeds, four staged cycles, 80 frozen
  Cycle-1/2 logical runs, and 296 physical segment jobs;
- `04_run_queue_v2/`: validated runtime queue consumed by every formal worker;
- `09_aiops/local_canary_20260808_v2/`: real-image Windows/GPU telemetry parity canary;
- `09_aiops/local_failure_canary_20260808_v6/`: real-image OOM, process-kill,
  checkpoint-resume, atomic-write, corrupt-sidecar, and hot-spare recovery canary.

The unversioned `03_preregistration/` and `04_run_queue/` directories are immutable
v1 evidence and must never be passed to a new release or worker.

The sole formal training process entrypoint is
`scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py`. It executes exactly
one released physical `--job-id`. The controller is optional scheduling convenience.

Formal training remains blocked until the real shared coordination-root canary,
ten-machine one-job real-data canary, machine preflights, engineering gate v2,
pilot release v2, and active assignment v2 all pass. No blind holdout has been opened.

Deadline for the ten-machine campaign: `2026-09-10`.

## Active Evidence Goal

The current research audit is governed by
`02_literature/GOAL_EXECUTION_CONTRACT_500_300_100_v2.md`. Version 1 remains an
immutable historical contract. On 2026-08-10 the user explicitly declared the
currently assembled literature sufficient and stopped further discovery and
reading. That scope change is registered in
`02_literature/review_500_300_100_v2/USER_LITERATURE_SUFFICIENCY_DECISION_20260810.md`.
The earlier exact 500/300/100 machine audit remains visibly `INCOMPLETE`; it is
superseded for the current user-approved scope and is not relabelled `PASS`.

The owner has restarted the evidence audit only. Formal training, owner canary,
an engineering gate, pilot release, and the blind holdout remain blocked while
this evidence goal is open.

## Expert Delivery Audit

The current authoritative delivery inventory is
`01_field_audit/expert_delivery_audit_v3/`. Its archive and output integrity
checks pass, but the overall state is `INCOMPLETE_SOURCE_MISSING` because the
BudgetedReplay v1.0.0 source TAR, source ZIP, and Wheel named by the expert SHA
ledger are not present. See `01_field_audit/EXPERT_DELIVERY_AUDIT_INDEX.md` for
the immutable v1-v3 audit history. Source-level comparison cannot be declared
complete until at least one ledger-matching source carrier is available.

## Global Completion Authority

`08_reports/COMPLETION_AUDIT.json` is the sole machine-readable authority for
the full research goal. Audit schema v2 currently reports
`INCOMPLETE_SOURCE_MISSING`: every locally controllable check passes, but the
BudgetedReplay source TAR, source ZIP, and Wheel are still absent. The audit
separately records the user literature scope decision, the legacy literature
validator state, the preserved historical gate/pilot assets, and the inactive
current v3 runtime. It never turns a historical or scoped local `PASS` into
training authorization.

The authoritative strict three-way comparison is the pair of v2 matrices under
`01_field_audit/expert_review_reproductions/`. All 46 rows have valid line
references, actual exit-zero read-only reproductions, and hash-bound result
artifacts. The 31 BudgetedReplay rows remain
`NOT_TESTABLE_SOURCE_MISSING` on the expert-source side.

## Q/R/A/D Preregistration v3

`03_preregistration_v3/` is the current scientific and construction contract for
finite-budget replay. Its local machine validation may pass while the global goal
remains incomplete: the package is explicitly `PREREGISTERED_NOT_RUN`, records
candidate effectiveness as `NOT_EVALUATED`, and authorizes no training, gate,
assignment, pilot, or blind/test access. It defines ordered Q/R/A/D factors without
an arbitrary weighted score, strict R1/R2/current-loss/no-replay controls, actual
optimizer-visible exposure accounting, disjoint unseen-seed confirmation, fixed
FN=0..95 endpoints, statistical stopping rules, and a migration/rollback-safe
repository construction specification.
