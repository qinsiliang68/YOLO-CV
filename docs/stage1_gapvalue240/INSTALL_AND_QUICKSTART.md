# Installation and quick start

## 1. Install the overlay

Run a dry collision check first:

```powershell
python tools/install_overlay.py --repo C:/GitHub/YOLO-CV
```

Then install:

```powershell
python tools/install_overlay.py --repo C:/GitHub/YOLO-CV --execute
```

The installer requires branch `push-info-sampling-lite` and commit prefix `07dc63763606`; it never writes protected trainer/evaluator/YOLO paths.

## 2. Edit one machine YAML

Only modify paths, GPU ID, workers, and optional runtime resource fields in `configs/stage1_gapvalue240/machines/machine_XX.yaml`. Scientific fields are rejected.

## 3. Central preparation

On the central node with the 2,000 raw OOF CSV files:

```powershell
python scripts/stage1_gapvalue240/prepare_experiment.py `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml
```

This validates frozen checksums, builds the float64 OOF cache once, generates all rankings, applies the overlap gate, freezes the final matrix, creates 240 selection manifests, and creates ten main-machine shards plus two empty reserve shards.

The successful preparation must contain:

```text
generated/PREPARATION_COMPLETE.json
generated/QUEUE_VALIDATION.json
generated/CROSS_TRIAD_AUDIT.json
generated/MACHINE_ALLOCATION_AUDIT.json
generated/FROZEN_QUEUE_FILE_MANIFEST.csv
generated/frozen_experiment_matrix.csv
generated/selection_index.csv
generated/selections/RUN_001..RUN_240/
generated/machine_shards/machine_01..machine_12_jobs.csv
```

Once `PREPARATION_COMPLETE.json` exists, preparation revalidates and returns `READY_REUSED`; it does not redraw or overwrite the selection CSVs.

## 4. Freeze, commit, and push queues before training-machine distribution

Training machines must never sample locally. Review the queue reports, then commit and push the frozen matrix, all 240 selection directories, selection index, audits, and 12 machine shards before copying a shard to any machine. The local memmap cache and reproducible ranking workspace are intentionally ignored by Git.

The training-machine checkout must use the exact commit containing these queue files. Each `run_NNN.py` consumes the committed `selection_manifest.csv` for that run.

The 80 complete triads are distributed round-robin across the ten main machines, so each machine receives eight triads (24 runs) spanning the experiment matrix. Within a triad, execution order rotates by triad ID among `T/R1/R2`, `R1/R2/T`, and `R2/T/R1` to reduce machine and run-order confounding while keeping every triad on one machine.

## 5. Optional real-image integration smoke

Before formal execution on a new environment:

```powershell
uv run python scripts/stage1_gapvalue240/smoke_real_integration.py `
  --output-root outputs/stage1_gapvalue240_smoke `
  --epochs 3 --batch 8 --device 0 --runs 5
```

This non-scientific smoke runs four small YOLO11n jobs and one YOLO11l job, each with 24 base rows plus 3 replay rows. It verifies real checkpoints, predictions, calibration, threshold sweep, and operational metrics without changing the frozen 240-run contract.

## 6. Execute one independent run

```powershell
python scripts/stage1_gapvalue240/runs/run_001.py `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml `
  --action run
```

The same file exposes `prepare()`, `train()`, `evaluate()`, `validate()`, and `run()` for Python use.

## 7. Execute a full triad or machine shard

```powershell
python scripts/stage1_gapvalue240/run_triad.py --machine-config <yaml> --triad-id TRIAD_001
python scripts/stage1_gapvalue240/run_machine_shard.py --machine-config <yaml>
```

A failed permanent machine requires the complete triad to be rerun on a reserve machine. Isolated arms are marked `SUPERSEDED` and excluded from paired analysis.

## 8. Aggregate only validated runs

```powershell
python scripts/stage1_gapvalue240/aggregate_results.py --machine-config <yaml>
```
