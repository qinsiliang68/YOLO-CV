# Machine D — Capacity x (largest)

**You are Machine D. Your job: run yolo11x-cls training + evaluation.**

First read [`MACHINE_SETUP.md`](MACHINE_SETUP.md) and complete sections 1–3.

---

## Launch command (try batch 24 first)

```bash
uv run python scripts/machines/machine_D.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs
```

Training config: yolo11x-cls, 200 epochs, batch 24 (default), imgsz 224.
Expected wall-clock: ~13 h on 3090/4090 (VRAM ~16 GB needed at batch 24).

---

## If OOM — reduce batch

If VRAM is tight (< 16 GB available) and you hit CUDA OOM:

```bash
uv run python scripts/machines/machine_D.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs \
    --batch 16
```

Lower `--batch` further (12, 8) if necessary. Document the batch size used
in a `RUN_NOTE.md` inside the uploaded tarball so the aggregator knows.

---

## Smoke test first (~5 min)

```bash
uv run python scripts/machines/machine_D.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs_smoke \
    --smoke
```

Verify `runs_smoke/yolo11x/final_test_metrics.json` appears. Delete and
proceed to full run.

---

## After training — what to upload

```
runs/yolo11x/
├── weights/epoch0.pt ... epoch199.pt, best.pt, last.pt   (~15 GB for x)
├── per_epoch_metrics.csv
├── best_epoch.json
└── final_test_metrics.json
├── RUN_NOTE.md  (if you reduced batch size)
```

Zip and upload:
```bash
cd runs
tar czf yolo11x_machine_D.tar.gz yolo11x
```

Large upload — allow 30–60 min depending on connection.

---

## Troubleshooting

- OOM at batch 24: see "If OOM" above.
- Training interrupted mid-epoch: restart directly:
  ```bash
  uv run python scripts/train_v3_stage1.py --capacity x \
      --data-dir /path/to/sewerml_gate_v3_stage1 \
      --output-dir ./runs/yolo11x \
      --batch 16
  ```
- Thermal throttling on long runs: no special handling needed; training tolerates slow epochs.
