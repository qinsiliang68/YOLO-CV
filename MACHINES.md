# Machines — v3 Stage 1 binary gate capacity scan

Two machines, one command each. No pasting, no chains, no AI babysitting.

## What we're running

**v3 Stage 1 binary gate, 5-capacity scan (true 2-class: Normal vs Defect).**

The existing 7-folder dataset `data/sewerml_gate_v3_stage1/` is kept as-is. A 2-folder hardlink view `data/sewerml_gate_v3_stage1_binary/` is built on the fly by each launcher (zero extra disk).

## Split

| Machine | Capacities | Est. wall-clock |
|---|---|---|
| A | yolo11n, yolo11s, yolo11m | ~10 h |
| B | yolo11l, yolo11x | ~13 h |

## Prereqs (same on both machines)

1. Repo cloned, `uv` installed, CUDA GPU available.
2. `data/sewerml_gate_v3_stage1/` exists with the 7-folder layout (train / val_cal / val_op / test, each containing `Normal` + 6 defect folders).

## Run

**Machine A**
```
cd <repo-root>
uv run python scripts/machines/machine_A.py
```

**Machine B**
```
cd <repo-root>
uv run python scripts/machines/machine_B.py
```

Each launcher does, in order: `git pull` → build the hardlink binary view → train each assigned capacity (200 epochs, batch 64, imgsz 224). If any step fails the launcher aborts — do not re-run blindly, read the error.

## Output

```
research/materials/v3_stage1_binary/yolo11{n,s,m,l,x}/
├── weights/epoch0.pt .. epoch199.pt          # every epoch (save_period=1)
├── per_epoch_logits/
│   ├── val_cal_epoch{i}.npz                  # logits + labels + image_ids
│   └── val_op_epoch{i}.npz
├── best_epoch/
│   ├── test_logits.npz
│   ├── embeddings_train.npz
│   ├── embeddings_val_op.npz
│   └── embeddings_test.npz
├── run_meta.json                             # git hash, GPU, time, etc.
└── args.yaml, results.csv                    # ultralytics internal
```

Only **linearly independent raw materials** are saved. T*, τ*, spec/recall/prec, confusion matrix, per-class recall — all derivable via `scripts/aggregate_capacity_results.py` after the fact.

## After both machines finish

```
uv run python scripts/aggregate_capacity_results.py   # computes all metrics, writes capacity_comparison.md
```

## Notes

- The existing 7-class results under `research/materials/v3_stage1/` are **not deleted**. They get reinterpreted in essay3 as a **6-class defect sub-category capacity scan** (complementary to the binary gate scan here).
- Smoke test is optional. Both launchers treat full 200-epoch training as the default; pass no flags.
