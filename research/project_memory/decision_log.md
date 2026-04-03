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

### D10. First-Round PTSG Result

- First-round post-hoc PTSG on `yolo11l-cls + hn02` has been completed.
- Best variant is `P2 = calibrated p_abnormal + trust`.
- Compared with `P0`:
  - `Spec@R99.5`: `0.5000 -> 0.5238`
  - `Spec@R99.0`: `0.5476 -> 0.5595`
  - `Prec@R99.0`: `0.9165 -> 0.9183`
  - `PTR@R99.0`: `0.9028 -> 0.8988`
- `P1` (uncertainty only) is negative.
- `P3/P4` do not beat `P2`.
- Current interpretation:
  - the useful gain comes from `trust`
  - stage-1 should be framed more explicitly as a selective safe-normal gate, not as a plain binary classifier

### D11. PTSG Next Wave Scope

- The next-wave stage-1 PTSG experiment stays post-hoc only.
- Fixed conditions:
  - main model `yolo11l-cls`
  - `hn02`
  - existing `val-cal / val-op` split
  - no backbone retraining
- Fixed comparison set:
  - `P2`: single-prototype trust baseline
  - `P5a`: `K=4` multi-prototype normal trust
  - `P5b`: `K=8` multi-prototype normal trust
  - `P6a`: `K=4` multi-prototype + margin trust
  - `P6b`: `K=8` multi-prototype + margin trust
- Stop rule:
  - if none of `P5/P6` clearly beats `P2`, stop deepening stage-1 and pivot to stage-2.

### D12. Stage-1 Strong-Embedding Route

- The final stage-1 heavy route is:
  - keep `yolo11l-cls`
  - keep `hn02`
  - keep the existing `val-cal / val-op` calibration protocol
  - strengthen the embedding space
  - then reuse the proven `P2 = calibrated p_abnormal + trust` gate
- First implementation uses `SupCon`, not CCL-SC.
- Comparison is fixed to:
  - `H0`: current best `yolo11l-cls + hn02 + P2`
  - `H1`: contrastive-enhanced backbone + calibration + plain score
  - `H2`: contrastive-enhanced backbone + calibration + trust gate
- If `H2` does not clearly beat `H0`, stage-1 is considered fully saturated and the mainline moves to stage-2 detector work.

### D13. Human-Friendly Execution

- The strong-embedding route must stay one-command friendly on training machines.
- Required command:
  - `uv run main.py`
- The active entry for this route is controlled through:
  - `YOLOv11/configs/runtime/main_entry.json`

### D14. Stage-1 Max-Filter Suite

- The current one-click default task is now `stage1_gate_maxfilter_suite`.
- It is designed as the final broad stage-1 continuation suite before fully pivoting to stage-2.
- The suite fixes:
  - main model `yolo11l-cls`
  - `hn02`
  - existing calibration and PTSG evaluation protocol
- It then compares four method families plus one hard-mix variant:
  - selective / recall-constrained loss
  - hard positive + hard normal mining
  - weighted BCE
  - focal BCE
  - defect oversampling
- All candidates must still be judged by the shared stage-1 ranking rule:
  - `Spec@R99.5`
  - `Spec@R99.0`
  - `Prec@R99.0`
  - `PTR@R99.0`
- This suite is meant to answer whether stage-1 still has meaningful headroom under the current protocol, not to change the role definition of stage-1 itself.
