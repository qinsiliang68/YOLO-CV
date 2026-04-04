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

### D15. Stage-1 RCIS First-Wave

- RCIS is currently positioned as a **first-pass information-driven resampling strategy**, not as a full dynamic sampling system.
- The current stage-1 baseline before RCIS remains:
  - `G4 current best HardMix + P0`
- The default first-wave RCIS suite is fixed to:
  - `R1`: boundary-only
  - `R2`: rcis_core
  - `R3`: rcis_full_exploratory
- First-wave signal priority is fixed to:
  - `boundary`
  - `hardness`
  - `redundancy`
- `uncertainty` stays as a light auxiliary signal only.
- `flip` is currently just a `P0/P2` proxy disagreement signal and is disabled in `rcis_core`.
- `quality_penalty` is disabled in `rcis_core` because sewer low-quality frames may still be deployment-real hard samples.
  - defect oversampling
- All candidates must still be judged by the shared stage-1 ranking rule:
  - `Spec@R99.5`
  - `Spec@R99.0`
  - `Prec@R99.0`
  - `PTR@R99.0`
- This suite is meant to answer whether stage-1 still has meaningful headroom under the current protocol, not to change the role definition of stage-1 itself.

### D15. Stage-1 Max-Filter Suite Result

- The full `max-filter suite` has completed.
- Best new experiment is:
  - `Hard positive + hard normal mining`
- Under the shared ranking rule, it becomes the new stage-1 training-side best candidate because:
  - `Spec@R99.5` ties the previous best `H0`
  - `Spec@R99.0` improves from `0.559524 -> 0.583333`
  - `Prec@R99.0` improves from `0.918322 -> 0.922395`
  - `PTR@R99.0` improves from `0.89881 -> 0.894841`
  - count-level change at `R99.0`: `47/4 -> 49/4`
- The best post-hoc variant for this new winner is no longer `P2`, but:
  - `P0 = calibrated p_abnormal`
- Current stage-1 final candidate is therefore:
  - `yolo11l-cls + calibration + hn02 + HardMix`
  - best gate decision uses `P0`

### D16. Stage-1 Formal Traceability Unification

- Official six-class source wording is now unified to the uniform rerun raw materials:
  - `yolo11x-cls` is the accuracy leader
  - `yolo11n-cls` is the AUROC/AUPRC leader
- The old `yolo11l-cls` six-class source leader wording is kept only as historical archive, not as current prose.
- The calibration-table `0% HN` baseline and the later `hn00` baseline are now treated as two different chains:
  - calibration table: model-selection baseline
  - `hn00`: HN/PTSG/max-filter mainline baseline
- These two baselines are not to be compared across tables item by item.
- The second-model supplementary table no longer reuses `G` numbering, to avoid confusion with the mainline ablation table.

### D17. Stage-1 Formal Capacity Protocol

- Stage-1 formal capacity scan is now governed by:
  - `research/project_memory/stage1_formal_protocol.md`
- Formal thesis-facing stage-1 materials must be written under:
  - `research/materials/stage1_formal/`
  - `research/results/stage1_formal/`
- Binary gate formal selection no longer trusts trainer-internal `top1_acc` or `best.pt`.
- Formal binary gate selection must come from checkpoint-level external summaries ranked by:
  - `Spec@R99.5`
  - `Spec@R99.0`
  - `Prec@R99.0`
  - `PTR@R99.0`
- Capacity-scan stage runs to a fixed `200` epochs and does not use early stopping as the official stop rule.
- All thesis-facing stage-1 formal classification runs use:
  - `batch = 24`
  - `save_period = 1`
  - `patience = 0`

### D18. Stage-1 Formal Dual-Machine Entrypoints

- Computer A formal launcher:
  - `uv run main_A.py`
  - fixed task `stage1_formal_gate_capacity`
- Computer B formal launcher:
  - `uv run main_B.py`
  - fixed task `stage1_formal_cls6_capacity`
- Formal reruns archive previous run/material directories into repo-local recycle paths.
- Formal resumes keep completed checkpoint summaries and backfill missing external evaluations before continuing training.
