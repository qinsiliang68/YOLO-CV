# Functional completeness map

| Responsibility | Implementation |
|---|---|
| Contract/hash/config boundary | `contract.py`, `machine.py`, active/archived YAML |
| Frozen input validation | `assets.py`, `identity.py` |
| Single-pass raw OOF cache | `oof.py` |
| Direct/dynamic rankings | `ranking.py` |
| Overlap gate/replacements | `overlap.py`, `prepare.py` |
| R1/R2 controls | `matching.py`, `selection.py` |
| Additive/guard manifests | `manifests.py` |
| 240 independent run APIs | `scripts/stage1_gapvalue240/runs/run_001.py` … `run_240.py` |
| Existing trainer integration | `integration.py`, `run_engine.py` |
| val_cal/val_op-only inference | `predictor.py` |
| Platt/operational-v2 | `calibration.py`, `metrics.py`, `evaluation.py` |
| Pre/postflight | `validation.py` |
| Atomic attempts/status/registry | `util.py`, `status.py`, `registry.py` |
| GPU/resource logging | `monitor.py` |
| 12-machine shards/pipeline | `shards.py`, `run_machine_shard.py` |
| Validated-only aggregation | `aggregate.py`, `statistics.py`, `reporting.py` |
| Cleanup/package checks | `cleanup.py`, `package_validation.py`, tools |
