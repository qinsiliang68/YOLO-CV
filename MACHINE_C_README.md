# Machine C — Capacity l

**You are Machine C. Your job: run yolo11l-cls training + evaluation.**

First read [`MACHINE_SETUP.md`](MACHINE_SETUP.md) and complete sections 1–3.

---

## Launch command

```bash
uv run python scripts/machines/machine_C.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs
```

Training config: yolo11l-cls, 200 epochs, batch 24, imgsz 224.
Expected wall-clock: ~9 h on 3090/4090 (VRAM ~12 GB).

---

## Smoke test first (optional, ~3 min)

```bash
uv run python scripts/machines/machine_C.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs_smoke \
    --smoke
```

Verify `runs_smoke/yolo11l/final_test_metrics.json` is produced. Delete and
proceed to full run.

---

## After training — what to upload

```
runs/yolo11l/
├── weights/epoch0.pt ... epoch199.pt, best.pt, last.pt   (~10 GB for l)
├── per_epoch_metrics.csv
├── best_epoch.json
└── final_test_metrics.json
```

Zip and upload:
```bash
cd runs
tar czf yolo11l_machine_C.tar.gz yolo11l
```

---

## Troubleshooting

- OOM on 12 GB VRAM: add `--batch 16` to the subprocess call inside machine_C.py.
- Training interrupted: restart directly:
  ```bash
  uv run python scripts/train_v3_stage1.py --capacity l \
      --data-dir /path/to/sewerml_gate_v3_stage1 \
      --output-dir ./runs/yolo11l
  ```
