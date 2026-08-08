# dynamic_replay_budget_efficiency_20260807

This experiment is parallel to the prior 40-run, 120-run, and 240-run/OOF work.
It does not replace or mutate those evidence packages.

Primary question: can replay schedule control and weak-defect protection make a fixed
sample budget produce stable benefit across unseen training seeds?

The current phase is `CODE_READY_FOR_OWNER_CANARY`. The active frozen assets are:

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
