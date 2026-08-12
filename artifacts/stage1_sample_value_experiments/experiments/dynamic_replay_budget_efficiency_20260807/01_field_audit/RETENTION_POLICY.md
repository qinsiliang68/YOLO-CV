# Stage1 dynamic replay retention policy

## Keep for every run

- Identity, environment, resolved configuration, input hashes, arm schedule, and sampler seed.
- Lightweight per-epoch metrics for all 200 epochs, including role-separated losses and exposure counts.
- Key checkpoints at epochs 120, 140, 150, 160, 180, 200.
- Raw val_op predictions at epochs 120, 140, 150, 160, 180, 200.
- Tail-probe trajectories, optimizer-step summaries, and compact batch/augmentation digests.
- Completion sidecars, row counts, SHA-256 manifests, and failure/retry lineage.

## Shared immutable assets

Store each frozen base split, probe manifest, and selection manifest once in a content-addressed shared immutable area.
Run folders contain only hashes and relative references. Never copy the same 120k-row manifest into every run.

## Pilot-gated heavy fields

True per-sample gradients are retained only for the preregistered candidate/probe subset and key checkpoints after
a pilot measures runtime, vector dimension, compression ratio, and predictive utility. Full-model gradients and
200-epoch checkpoints are not default campaign outputs.

## Recomputable outputs

Threshold sweeps, plots, HTML, and aggregate tables are recomputed from raw predictions and are not duplicated per run.
Text logs are compressed after successful postflight validation.

## Upload and deletion gate

A run is deletable from a worker only after: postflight passes, the artifact manifest is complete, remote upload succeeds,
remote SHA-256 verification succeeds, and the central index records the remote location. Failed or partial runs remain quarantined.

Largest current lower-bound scenario: 180 runs, 101.51 GiB,
excluding the gradient payload pending pilot measurement.
