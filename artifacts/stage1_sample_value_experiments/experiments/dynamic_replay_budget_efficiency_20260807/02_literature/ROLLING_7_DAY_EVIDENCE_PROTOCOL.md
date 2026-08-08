# Rolling Seven-Day Evidence Protocol (Draft)

## Purpose

Use the remaining multi-GPU window to produce one complete, interpretable result package roughly every seven days. Seven days is a reporting and decision cadence, not a requirement to inflate or delay an experiment.

This document fixes the operational shape only. Exact replay ratios, arms and seed counts remain provisional until the full-text evidence tranche is complete and the final protocol is preregistered.

## Four Weekly Questions

### Week 1 - Does Replay Timing Matter?

Use one frozen selection and a minimal paired set of schedule controls to test whether reducing late replay changes the normal-tail/weak-defect-tail trajectory. Produce a complete mechanism report, not a final efficacy claim.

### Week 2 - Timing Or Total Dose?

Add the cumulative-dose-matched schedule required to distinguish when replay occurs from how much replay occurs. Expand to enough unseen seeds to estimate reversal and double-harm rates.

### Week 3 - Does Weak-Defect Protection Prevent Harm?

Freeze the best eligible normal schedule from Weeks 1-2 and vary only the defect-guard component. Keep total replay slots fixed so guard benefit is not confused with more training data.

### Week 4 - Frozen Combined Confirmation

Freeze one complete policy and compare it against its component ablations, matched random control, global random control and no replay on new seeds. Do not alter the policy after opening this confirmation block.

## Rolling Output Cadence Inside Each Week

- Every epoch: append lightweight process telemetry with atomic writes.
- Every completed run: validate row counts, sample identities, exposure accounting and checkpoint provenance.
- Every completed paired block: update a provisional dashboard on `val_op`; do not open blind holdout.
- Every 24 hours: emit machine throughput, GPU train-window utilization, dataloader-wait fraction, failures, retries and estimated completion time.
- At the end of the weekly block: issue a closed result package with the hypothesis, arms, exact manifests, paired statistics, raw safety frontiers, trajectory plots, failures and the preregistered release decision for the next block.

Looking at intermediate operational results is allowed. Changing scientific hypotheses or arms mid-block is not allowed unless a preregistered safety/futility gate fires.

## Queue Release Rules

1. Build all manifests and hashes before a block starts.
2. Run a local real-data smoke test and one multi-machine canary block.
3. Release complete paired blocks rather than isolated treatment arms.
4. Keep the next validated block staged so a healthy machine does not idle.
5. Release the next scientific stage only after the previous stage's registered integrity gate passes.
6. If a block finishes early, analyze and advance immediately; do not wait for the calendar.
7. If the minimal valid block needs less than seven days, keep it small. Do not add controls merely to occupy time.

## Resource And Failure Contract

- Default data-loader target: four workers per training process, pinned memory, persistent workers, bounded prefetch and non-blocking host-to-device copies when supported.
- Measure GPU utilization only over train-step windows; evaluation, checkpointing and report generation are recorded separately.
- Record GPU utilization, memory allocated/reserved, dataloader wait, batch compute time, host RSS, disk free space and write latency.
- Use atomic checkpoint and telemetry writes. A checkpoint is complete only when its sidecar validation passes.
- On process failure or OOM, clear resources and resume from the latest validated checkpoint with the identical scientific configuration.
- Never silently reduce batch size, workers, input size, precision mode, replay ratio or optimizer settings.
- A repeated OOM invalidates the attempt and moves the entire paired block to the hot-spare queue. Any alternative memory policy must be preregistered and applied to every corresponding arm.
- Machine 11 performs heavy checkpoint inference; machine 12 is the hot spare. Training machines should not block on report rendering.

## Weekly Result Package Minimum

- immutable run/selection/schedule manifests and hashes;
- complete-at-epoch status and failure ledger;
- all-epoch lightweight trajectory table;
- key-checkpoint raw predictions and safety frontiers;
- paired treatment-control effects by seed;
- reversal rate, double-harm rate and uncertainty interval;
- weak-defect and hard-normal trajectory plots;
- resource-utilization and throughput report;
- explicit conclusion: supported, unsupported, inconclusive, or operationally invalid;
- next-stage release decision tied to a preregistered rule.

## Blindness Rule

The blind holdout stays sealed through Weeks 1-3. It is opened only after the Week-4 policy, code commit, manifests, seeds, evaluation code and decision thresholds are frozen.
