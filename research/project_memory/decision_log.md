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
