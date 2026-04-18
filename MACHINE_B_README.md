# Machine B — Capacity m

**You are Machine B. Your job: run yolo11m-cls training + evaluation.**

First read [`MACHINE_SETUP.md`](MACHINE_SETUP.md) and complete sections 1–3
(clone repo, install uv, download & verify dataset).

---

## Launch command

```bash
uv run python scripts/machines/machine_B.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs
```

Training config: yolo11m-cls, 200 epochs, batch 24, imgsz 224.
Expected wall-clock: ~6 h on 3090/4090 (VRAM ~8 GB).

---

## Smoke test first (optional, ~2 min)

```bash
uv run python scripts/machines/machine_B.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs_smoke \
    --smoke
```

If `runs_smoke/yolo11m/final_test_metrics.json` appears, the pipeline works.
Delete `runs_smoke/` and proceed.

---

## After training — what to upload

```
runs/yolo11m/
├── weights/epoch0.pt ... epoch199.pt, best.pt, last.pt   (~5 GB total for m)
├── per_epoch_metrics.csv
├── best_epoch.json
└── final_test_metrics.json
```

Zip and upload:
```bash
cd runs
tar czf yolo11m_machine_B.tar.gz yolo11m
```

---

## Troubleshooting

- OOM: add `--batch 16` to the subprocess call inside machine_B.py
  (edit `CAPACITIES`/`run()` to include `"--batch", "16"`).
- Training interrupted: re-launch to restart from scratch (safest). Or:
  ```bash
  uv run python scripts/train_v3_stage1.py --capacity m \
      --data-dir /path/to/sewerml_gate_v3_stage1 \
      --output-dir ./runs/yolo11m
  ```
