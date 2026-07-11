# Installation and quick start

Read `HANDOFF_RATIONALE_AND_STATUS_v1_2.md` first. It records the scientific origin, the exact 240-run composition, the boundary with AIOps, what has been verified locally, and which external canary gates still block formal release.

## 1. Install the overlay

Run a dry collision check first:

```powershell
uv run python tools/install_overlay.py --repo C:/GitHub/YOLO-CV
```

Then install:

```powershell
uv run python tools/install_overlay.py --repo C:/GitHub/YOLO-CV --execute
```

The imported expert overlay is already integrated in this repository. The archived trainer/evaluator/YOLO paths remain protected; runtime v1.2 is implemented in new GapValue adapters and wrappers. Do not use the expert ZIP installer to overwrite the integrated files.

## 2. Edit one machine YAML

Only modify paths, GPU ID, workers, and optional runtime resource fields in `configs/stage1_gapvalue240/machines/machine_XX.yaml`. Scientific fields are rejected.

## 3. Central preparation

On the central node with the 2,000 raw OOF CSV files:

```powershell
uv run python scripts/stage1_gapvalue240/prepare_experiment.py `
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

## 5. Verify runtime identity and machine assets

Before GPU work, verify the immutable runtime links and all committed selections:

```powershell
uv run python scripts/stage1_gapvalue240/runtime_integrity.py `
  --runtime-contract configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml `
  links --repo-root .

uv run python scripts/stage1_gapvalue240/runtime_integrity.py `
  --runtime-contract configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml `
  all-selections --repo-root .
```

Each machine then creates its one-time asset report from its own machine YAML. Formal runs reuse this report and do not rescan all images:

```powershell
uv run python scripts/stage1_gapvalue240/runtime_integrity.py `
  --runtime-contract configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml `
  build-machine-assets `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml `
  --output <machine_asset_report.json> `
  --image-verification existence
```

## 6. Local resource smoke

Before formal execution on a new environment:

```powershell
uv run python scripts/stage1_gapvalue240/smoke_real_integration.py `
  --output-root outputs/stage1_gapvalue240_smoke `
  --epochs 3 --batch 8 --device 0 --runs 5
```

This non-scientific smoke runs four small YOLO11n jobs and one YOLO11l job, each with 24 base rows plus 3 replay rows. It verifies real checkpoints, predictions, calibration, threshold sweep, and operational metrics without changing the frozen 240-run contract.

The formal-spec resource smoke uses YOLO11l, batch 128, workers 8, isolated subprocesses, and two sequential three-epoch jobs:

```powershell
uv run python scripts/stage1_gapvalue240/local_resource_smoke.py `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml
```

Its output is validation evidence, not a scientific run, and must not enter the 240-run aggregate.

## 7. Execute one independent run

```powershell
uv run python scripts/stage1_gapvalue240/runs/run_001.py `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml `
  --action run
```

The same file exposes `prepare()`, `train()`, `evaluate()`, `validate()`, and `run()` for Python use.

### Formal trainer worker and staging contract

Formal GPU execution uses `scripts/stage1_gapvalue240/formal_train_worker.py` as an isolated child process. Its CLI accepts only frozen run inputs, paths, device, and worker count; `epochs`, `batch`, `imgsz`, `patience`, determinism, and cache behavior come from the verified experiment contract and cannot be overridden on the command line.

`staging_root` must be on the same filesystem volume as `dataset_root`. The first worker invocation creates one reusable hardlink-only base cache containing the 120,000 training and 24,000 `val_model` images. A run holds an exclusive staging lock, adds only its `replay__*` hardlinks, trains, and removes replay links plus Ultralytics `train.cache`/`val.cache` files in `finally`. A hardlink or volume check failure stops the worker; image-copy fallback is forbidden.

The worker keeps YOLO-native `results.csv` and `args.yaml` under `<output-dir>/trainer/`, maintains the crash-resume checkpoint at `<output-dir>/training_state/last.pt`, and writes `<output-dir>/training_execution_audit.json`. Resume is explicitly recorded as `native_approximate` with segment and checkpoint provenance.

## 8. Execute a full triad or machine shard

```powershell
uv run python scripts/stage1_gapvalue240/run_triad.py --machine-config <yaml> --triad-id TRIAD_001
uv run python scripts/stage1_gapvalue240/run_machine_shard.py --machine-config <yaml>
```

A failed permanent machine is handed to AIOps/operator review. Reserve takeover normally reruns the complete triad with new attempt IDs. Old attempts remain auditable; a replacement supersedes them only after it passes validation.

## 9. Aggregate only validated runs

```powershell
uv run python scripts/stage1_gapvalue240/aggregate_results.py --machine-config <yaml>
```

Do not create the release tag or start all 240 runs merely because local unit and resource smoke tests pass. The required external gates are: a real 120.6k B600 T/R1/R2 three-epoch triad, an interruption/resume exercise, one representative 200-epoch canary, all 12 machine benchmarks, and a 15-day capacity estimate with 25% failure buffer. Only then create `stage1-gapvalue240-runtime-v1.2.0` at the reviewed commit.
