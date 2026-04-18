# Machine A — Capacity n + s (sequential)

**You are Machine A. Your job: run yolo11n-cls and yolo11s-cls sequentially.**

First read [`MACHINE_SETUP.md`](MACHINE_SETUP.md) and complete sections 1–3
(clone repo, install uv, download & verify dataset).

---

## Launch command

```bash
uv run python scripts/machines/machine_A.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs
```

This runs two training + evaluation passes back-to-back:

1. yolo11n-cls: 200 epochs, batch 24, imgsz 224 (~3 h on 3090/4090)
2. yolo11s-cls: 200 epochs, batch 24, imgsz 224 (~4 h on 3090/4090)

Total expected wall-clock: ~7 h.

---

## Smoke test first (optional, ~4 min)

```bash
uv run python scripts/machines/machine_A.py \
    --data-dir /path/to/sewerml_gate_v3_stage1 \
    --output-dir ./runs_smoke \
    --smoke
```

3-epoch version with batch 4. If it completes without crashing and produces
`runs_smoke/yolo11n/final_test_metrics.json` AND `runs_smoke/yolo11s/final_test_metrics.json`,
the pipeline works on this machine.

Delete `runs_smoke/` after verification and proceed to the full run.

---

## After training completes

You should find:
```
runs/yolo11n/
├── weights/epoch0.pt ... epoch199.pt, best.pt, last.pt
├── per_epoch_metrics.csv
├── best_epoch.json
└── final_test_metrics.json

runs/yolo11s/
├── weights/epoch0.pt ... epoch199.pt, best.pt, last.pt
├── per_epoch_metrics.csv
├── best_epoch.json
└── final_test_metrics.json
```

Zip both folders separately and upload to shared storage:
```bash
cd runs
tar czf yolo11n_machine_A.tar.gz yolo11n
tar czf yolo11s_machine_A.tar.gz yolo11s
# (or 7z/zip on Windows)
```

**Each tarball is large (~500 MB to 2 GB for n/s after 200 checkpoints).** Use
a resumable upload client if the shared storage is flaky.

That's your only responsibility. The local operator handles result aggregation.

---

## Troubleshooting

- OOM: unlikely on n/s, but if so add `--batch 16` to machine_A.py's subprocess calls.
- If yolo11n training finishes and yolo11s crashes: re-launch is safe, it won't
  repeat n (each capacity has its own `--output-dir/yolo11{cap}`). Or manually:
  ```bash
  uv run python scripts/train_v3_stage1.py --capacity s \
      --data-dir /path/to/sewerml_gate_v3_stage1 \
      --output-dir ./runs/yolo11s
  ```
