# Experiment Handoff Workflow

This repository now uses a **single-branch workflow**.

## Roles

### Local working machine

- updates code, scripts, configs and thesis
- curates experiment materials
- pushes the canonical repository state to `main`

### Training machine

- syncs from `main`
- runs experiments
- writes raw materials into:
  - `research/materials/`
  - `research/results/`
- pushes only experiment outputs back to `main`

## Why this workflow

- both machines always keep the same directory structure
- the training machine can directly pull and run
- the local machine can directly pull results and continue writing
- there is no extra branch logic to remember

## Standard commands

Before changing defaults or deciding the next experiment, read:

- `PROJECT_MEMORY.md`
- `research/project_memory/stage1_memory.md`
- `research/project_memory/decision_log.md`
- `research/project_memory/stage1_gate_theory_and_maxfilter_report.md`

### Sync code before training

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\git_sync_main.ps1
```

### Push results after training

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\git_push_results_main.ps1
```

## Rules

- do not push datasets, weights, runs, or local caches
- do not edit thesis files on the training machine
- do not force push from the training machine
- keep experiment outputs under `research/materials/` and `research/results/`
