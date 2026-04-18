# Machine Setup — Common Instructions

*Applies to: all 4 machines (A, B, C, D).*

This document covers the **common setup** before any machine-specific launcher runs.
Each machine's AI helper should confirm every step here, then open the
machine-specific `MACHINE_X_README.md` for launch instructions.

---

## 0 · Machine role assignment

| Machine | Capacity | Script | Expected time (3090/4090) | VRAM |
|---|---|---|---|---|
| **A** | `n` + `s` sequential | `scripts/machines/machine_A.py` | ~3 h + ~4 h = ~7 h | 4 GB / 6 GB |
| **B** | `m` | `scripts/machines/machine_B.py` | ~6 h | 8 GB |
| **C** | `l` | `scripts/machines/machine_C.py` | ~9 h | 12 GB |
| **D** | `x` | `scripts/machines/machine_D.py` | ~13 h | 16 GB |

---

## 1 · Clone / pull the repo

```bash
git clone https://github.com/qinsiliang68/YOLO-CV.git
cd YOLO-CV
git checkout push-info-sampling-lite
git pull
```

---

## 2 · Install dependencies (uv)

Install `uv` if not present:
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify Python version meets the `pyproject.toml` requirement (`>=3.11,<3.13`):
```bash
uv python list
```

No pip install step — `uv run python` auto-creates the venv on first invocation.

---

## 3 · Download and stage the dataset package

The dataset package is **`sewerml_gate_v3_stage1/`** (~13 GB). Download it
from the shared cloud location (provided by the local operator), then place
it at a path of your choosing, e.g.:

```
/data/sewerml_gate_v3_stage1/
├── manifests/         (7 CSV/JSON/README files)
├── train/             24,000 images in 7 class folders
│   ├── Normal/        12,000
│   ├── PF/             2,000 (and similar for DE, RB, AF, OB, FS)
├── val_cal/           2,400 images in 7 class folders (natural distribution)
├── val_op/            5,600 images in 7 class folders (natural)
└── test/             20,000 images in 7 class folders (natural)
```

**Integrity check** — after extraction, confirm:

```bash
# From repo root (with the above path exported as DATA_DIR):
export DATA_DIR=/data/sewerml_gate_v3_stage1

# Total image count should be 52,000
find "$DATA_DIR" -name "*.png" | wc -l       # -> 52000

# Train folder breakdown (rarity-priority assigned)
for f in Normal PF DE RB AF OB FS; do
  echo -n "train/$f: "
  ls "$DATA_DIR/train/$f" | wc -l
done
```

Expected train split breakdown (may vary ±1% on PF/DE/... due to multi-label
co-occurrence but Normal == 12,000 is exact):
- Normal 12000  •  PF ~2155  •  DE ~1921  •  RB ~2509  •  AF ~2298  •  OB ~2415  •  FS ~702

If counts are wrong, re-download.

---

## 4 · Run your machine-specific launcher

See the `MACHINE_X_README.md` matching your role (A, B, C, or D).

Example for Machine A:
```bash
uv run python scripts/machines/machine_A.py \
    --data-dir /data/sewerml_gate_v3_stage1 \
    --output-dir ./runs
```

The launcher will:
1. Auto-install dependencies (first run only)
2. Auto-download the pretrained yolo11{capacity}-cls.pt weights (first run)
3. Train for 200 epochs (`save_period=1`, i.e. save every epoch)
4. Run external per-epoch evaluation (T fit, τ search, Spec metrics)
5. Lex-rank and pick best epoch
6. Run final test evaluation with frozen `(θ*, T*, τ*)`
7. Save `per_epoch_metrics.csv`, `best_epoch.json`, `final_test_metrics.json`

---

## 5 · Smoke test first (recommended, ~2 min per capacity)

Before the full 200-epoch run, verify the pipeline works end-to-end with
`--smoke` (3 epochs, batch 4, whatever data you have):

```bash
uv run python scripts/machines/machine_A.py \
    --data-dir /data/sewerml_gate_v3_stage1 \
    --output-dir ./runs_smoke \
    --smoke
```

If this succeeds and you see `[DONE]` + `final_test_metrics.json` files
produced under `./runs_smoke/yolo11*/`, the pipeline is verified for your
machine. Delete `./runs_smoke/` and proceed to the full run.

---

## 6 · After training — what to upload

Zip the **entire `runs/yolo11{capacity}/` folder** and upload.

```
runs/yolo11{capacity}/
├── weights/
│   ├── epoch0.pt, epoch1.pt, ..., epoch199.pt     200 ckpts (save_period=1)
│   └── best.pt, last.pt                           (ultralytics convention)
├── per_epoch_logits/
│   ├── val_cal_epoch{i}.npz                       (logits, labels, image_ids) × 200
│   └── val_op_epoch{i}.npz                        (logits, labels, image_ids) × 200
├── best_epoch/
│   ├── test_logits.npz                            (logits, labels, image_ids) on test
│   ├── embeddings_train.npz                       (features, labels, image_ids) penultimate
│   ├── embeddings_val_op.npz
│   └── embeddings_test.npz
├── run_meta.json                                  (git, host, GPU, timestamps, duration)
└── args.yaml, results.csv                         (ultralytics internal, kept)
```

**Design principle — "linearly independent raw materials only"**: this folder
contains ONLY data that cannot be reproduced without re-running training.
Derived quantities (T*, τ*, Spec@R99.5, confusion matrix, per-class recall,
τ-Spec curves, etc.) are intentionally NOT saved — they are reconstructed by
`scripts/aggregate_capacity_results.py` from the raw logits + labels.

Expected total size per capacity (approximate):
- n: ~2 GB   (200 × 6 MB ckpts + ~400 MB logits/embeddings)
- s: ~5 GB   (200 × 20 MB + ~700 MB)
- m: ~9 GB   (200 × 40 MB + ~1 GB)
- l: ~11 GB  (200 × 50 MB + ~1.2 GB)
- x: ~24 GB  (200 × 110 MB + ~2 GB)

**What NOT to do** (the local operator handles these, not the machine AIs):
- Do NOT aggregate results across capacities — that's `aggregate_capacity_results.py`
- Do NOT write summary tables / MD / paper content
- Do NOT modify training config (epochs, batch, imgsz) unless VRAM forces a reduction

---

## 7 · Troubleshooting

**OOM on Machine D (`x` capacity)**: add `--batch 16` (or 12) to machine_D.py. Document the change in the uploaded folder's README.

**`val/` symlink fails on Windows**: the training script falls back to a
directory junction (`mklink /J`) or copy. No action needed.

**Pretrained weight download fails**: the script auto-fetches from Ultralytics
GitHub releases. If rate-limited, wait and retry, or manually download
`yolo11{capacity}-cls.pt` into the repo root.

**Training interrupted**: ultralytics resumes from `weights/last.pt` on
restart if you pass `resume=True` — but simplest is to restart from scratch.
All epoch checkpoints are preserved, so restart loses time only.
