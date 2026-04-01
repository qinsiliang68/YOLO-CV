# Decision Log

## Stable Decisions

### D1. Workflow

- Repository uses a single `main` branch.
- Training machines sync from `main`.
- Training machines push only `research/materials/` and `research/results/`.

### D2. Delete Policy

- Any requested deletion goes to `_recycle_bin/`.
- Do not physically delete by default.

### D3. Stage-1 Role Definition

- `s` = default-threshold leader
- `l` = high-recall leader
- `m` = AUPRC reference

This mapping must not be changed unless raw materials prove otherwise.

### D4. Calibration Policy

- Use Temperature Scaling
- Use unified `val-cal / val-op = 30% / 70%`
- Use `seed = 20260330`

### D5. HN Policy

- HN ratio sweep was already done on `l`
- Mainline HN ratio is fixed at `2%`
- Full ratio sweep remains as evidence chain
- Follow-up experiments compare `0%` vs `2%`

### D6. Human-Friendly Entrypoint

- Default human-facing training entrypoint should be:

```powershell
uv run main.py
```

- The active task is controlled by:
  - `YOLOv11/configs/runtime/main_entry.json`

### D7. What To Keep

Keep:

- `dataset_manifest.csv`
- `split_train.csv`
- `split_val.csv`
- key csv/json result files
- thesis figures actually used in manuscript

Prune aggressively:

- redundant JPG artifacts
- duplicate profile files
- temporary generated configs

### D8. Stage-1 Cross-Capacity HN Validation

- `yolo11s-cls + hn02` has been completed and calibrated.
- Compared with calibrated `yolo11s-cls` baseline:
  - `Spec@R99.5`: `0.3571 -> 0.4167`
  - `Spec@R99.0`: `0.4167 -> 0.4881`
  - `Prec@R99.0`: `0.8949 -> 0.9063`
  - `PTR@R99.0`: `0.9246 -> 0.9107`
- This confirms that `hn02` has cross-capacity benefit, but `yolo11l-cls + hn02` remains the mainline because its calibrated high-recall operating points are still stronger.

### D9. Stage-1 Next Method Candidate

- The next stage-1 method candidate is **PTSG** under the broader **SNSG/selective gate** framing.
- Immediate implementation priority is:
  - keep the current `G2 = yolo11l-cls + calibration + hn02`
  - add a post-hoc safe-normal score
  - do **not** retrain the backbone first
- First comparison set is fixed to:
  - `P0`: calibrated `p_abnormal`
  - `P1`: `p_abnormal + uncertainty`
  - `P2`: `p_abnormal + trust`
  - `P3`: `p_abnormal + trust + uncertainty`
  - `P4`: `P3 + HN-aware normal bank`
- Ranking remains:
  - `Spec@R99.5`
  - `Spec@R99.0`
  - `Prec@R99.0`
  - `PTR@R99.0`
