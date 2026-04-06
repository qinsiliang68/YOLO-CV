# Stage-1 Formal Protocol

## Scope

This document defines the only thesis-facing protocol for the new **stage-1 formal capacity scan**.

It applies to:

- `stage1_formal_gate_capacity`
- `stage1_formal_cls6_capacity`

It also governs the fixed dual-machine launchers:

- `main_A.py`
- `main_B.py`

## Formal Objective

### Binary Gate

Stage-1 direct binary gate is a **high-recall normal-filtering task**, not a default-threshold accuracy task.

The official ranking rule is:

1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0` ascending

### Six-Class Source

The six-class source scan is an auxiliary source-capacity reference, not the final gate judge.

Its official ranking rule is:

1. `Accuracy`
2. `AUROC`
3. `AUPRC`

## Fixed Formal Training Rule

All thesis-facing stage-1 classification formal experiments use:

- `batch = 24`
- `epochs = 200`
- `save_period = 1`
- `patience = 0`

This means:

- capacity scan always runs the full 200 epochs
- every epoch checkpoint is preserved
- trainer-side `acc/top1/loss` never decides the formal winner

## Why Every Epoch Must Be Saved

Formal capacity scan is a checkpoint-selection problem, not a trainer-best problem.

Every epoch checkpoint is saved because:

- the formal best checkpoint may not match trainer `best.pt`
- external gate-aware metrics can peak after trainer `top1` stops improving
- thesis tables, appendix plots, and later audits need the full checkpoint trace

## Why External Evaluation Is Mandatory

Trainer metrics are allowed only as training-health signals:

- train loss
- validation top1 / accuracy
- GPU utilization
- epoch time
- checkpoint save status

They do not decide the formal winner.

The formal binary gate winner is chosen only after external evaluation over:

- `temperature T`
- `tau_r995`
- `tau_r990`
- `Spec@R99.5`
- `Spec@R99.0`
- `Prec@R99.0`
- `PTR@R99.0`
- `TN/FN @ R99.5`
- `TN/FN @ R99.0`

The formal cls6 winner is chosen only after external evaluation over:

- `Accuracy`
- `AUROC`
- `AUPRC`

## Follow-On Rule After Capacity Scan

Capacity scan is the only formal model-selection step for stage-1 backbone choice.

Once the formal binary gate ranking is fixed:

- the main model is the gate leader
- the second model is the gate runner-up
- later stage-1 experiments must reuse the same calibrated external evaluation protocol

This means:

- `calibration` is no longer treated as a separate training phase
- later HN / HardMix / information-sampling experiments are evaluated under the same `val-cal -> temperature fit -> val-op threshold scan` pipeline
- trainer `acc/top1` still remains a health-only signal and cannot replace gate-aware selection

## Directory Rule

All new thesis-facing stage-1 materials must live under:

- `research/materials/stage1_formal/`
- `research/results/stage1_formal/`

Formal run roots are:

- `YOLOv11/runs/stage1_formal_gate/`
- `YOLOv11/runs/stage1_formal_cls6/`

No new formal outputs may be written into old exploratory stage-1 directories.

## Immediate Artifact Rule

Do not wait until the end of training to start producing formal materials.

The system must append usable artifacts as early as possible:

- `run_manifest.json`
- `dataset_manifest.json`
- `env_snapshot.json`
- `pip_freeze.txt`
- `dataset_inventory.csv`
- `epoch_metrics.csv`
- `all_checkpoints_index.csv`

Per-epoch formal summaries are append-only:

- gate: `epoch_gate_summary.*`
- cls6: `epoch_cls6_summary.*`

The five-model total summaries are refreshed incrementally whenever a model summary exists.

## Interrupt / Resume Rule

Formal capacity scan must be interruption-safe.

The rule is:

1. Every saved checkpoint remains on disk.
2. Every already-written summary row remains valid.
3. Re-running the same formal task does not discard completed epoch summaries by default.
4. Before resuming training, the task first backfills external evaluation for checkpoints that already exist.
5. If target epoch 200 has not been reached and `weights/last.pt` exists, the task resumes from `last.pt`.
6. If checkpoints already reach epoch 200, training is skipped and only evaluation / summary refresh runs.
7. Only `--rerun` is allowed to archive the old run/material directory and restart that model from scratch.

This guarantees:

- completed epoch summaries remain usable after interruption
- rerun and resume rules are explicit
- no already-generated formal row is silently lost

## Required Per-Model Outputs

Gate runs must retain:

- `run_manifest.json`
- `dataset_manifest.json`
- `env_snapshot.json`
- `pip_freeze.txt`
- `dataset_inventory.csv`
- `epoch_gate_summary.csv`
- `epoch_gate_summary.json`
- `epoch_gate_summary.md`
- `best_epoch_manifest.json`
- `all_checkpoints_index.csv`
- `epoch_gate_dashboard.png`
- `per_epoch_gate/`
- `stdout.log`
- `stderr.log`

Cls6 runs must retain:

- `run_manifest.json`
- `dataset_manifest.json`
- `env_snapshot.json`
- `pip_freeze.txt`
- `dataset_inventory.csv`
- `epoch_cls6_summary.csv`
- `epoch_cls6_summary.json`
- `epoch_cls6_summary.md`
- `best_epoch_manifest.json`
- `all_checkpoints_index.csv`
- `epoch_cls6_dashboard.png`
- `stdout.log`
- `stderr.log`

## Required Global Outputs

Gate:

- `binary_gate_capacity_summary.csv`
- `binary_gate_capacity_summary.json`
- `binary_gate_capacity_summary.md`
- `binary_gate_capacity_comparison.png`

Cls6:

- `cls6_capacity_summary.csv`
- `cls6_capacity_summary.json`
- `cls6_capacity_summary.md`
- `cls6_capacity_comparison.png`

The formal registry file is:

- `research/materials/stage1_formal/manifests/formal_capacity_scan_registry.csv`

It must keep:

- `task`
- `model`
- `run_dir`
- `summary_dir`
- `best_epoch`
- `best_checkpoint_path`
- `commit_hash`
- `machine_name`
- `dataset_manifest_path`

## Legacy Material Policy

Earlier stage-1 documents, exploratory notes, and outdated summary prose are preserved under:

- `research/archive/stage1_preformal_legacy/`

Every moved document must be traceable through:

- `research/archive/stage1_preformal_legacy/archive_manifest.json`
- `research/archive/stage1_preformal_legacy/archive_manifest.md`

## Human-Facing Launch Commands

Computer A:

```powershell
uv run main_A.py
```

Computer B:

```powershell
uv run main_B.py
```

Explicit task entrypoints remain available:

```powershell
uv run main.py --task stage1_formal_gate_capacity
uv run main.py --task stage1_formal_cls6_capacity
```
