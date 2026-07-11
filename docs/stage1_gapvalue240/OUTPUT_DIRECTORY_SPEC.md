# Output directory specification

Each run writes to a unique attempt directory:

```text
output_root/
  runs/RUN_001/
    attempt_<timestamp>_<arm>_<uuid>.inprogress/
      00_identity/
      01_manifests/
      02_logs/
      03_checkpoints/
      04_predictions/
      05_metrics/
      06_figures/
      07_validation/
      08_status/
      work/
```

A successful postflight atomically renames the directory without `.inprogress`. Existing attempts are never overwritten. Only a single status marker exists at a time. Only `VALIDATED` attempts enter aggregation.

Permanent assets include resolved identity, manifests, environment, logs, best/last checkpoints, final val_cal and val_op predictions, calibration model, operational metrics, threshold sweep, and validation reports. Full per-epoch predictions are not a default 240-run output.
