# Stage-1 Gate Capacity Scan Report

## Experiment Setup
- dataset: `C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200`
- train samples: `6480`
- val samples: `720`
- train class counts: `Abnormal=5400`, `Normal=1080`
- val class counts: `Abnormal=600`, `Normal=120`
- batch size: `24`
- epochs: `200`
- checkpoint saved: `yes (per epoch)`
- calibration inside evaluator: `yes (temperature fitted on val-cal, evaluated on val-op)`
- selection rule:
  1. `Spec@R99.5`
  2. `Spec@R99.0`
  3. `Prec@R99.0`
  4. `PTR@R99.0`

## Best Checkpoint per Model (Gate-aware)

| model | best_epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | tau_r995 | tau_r990 |
|------|------------|------------|------------|------------|------------|----------|----------|
| yolo11n | 76 | 0.511905 | 0.583333 | 0.922395 | 0.894841 | 0.25 | 0.36 |
| yolo11s | 77 | 0.523810 | 0.535714 | 0.914286 | 0.902778 | 0.25 | 0.26 |
| yolo11m | 78 | 0.595238 | 0.654762 | 0.934831 | 0.882937 | 0.28 | 0.31 |
| yolo11l | 54 | 0.523810 | 0.571429 | 0.920530 | 0.898810 | 0.28 | 0.30 |
| yolo11x | 125 | 0.547619 | 0.595238 | 0.924444 | 0.892857 | 0.29 | 0.36 |

## Epoch Curve Summary

### yolo11n
- best epoch: `76`
- early peak: `no`
- trend:
  - epoch 1-20: low-start regime, best `Spec@R99.5=0.285714`, best `Spec@R99.0=0.345238`
  - epoch 20-80: clear main climb, reaches global best at epoch `76`
  - epoch 80-200: stays useful and relatively stable, but does not surpass epoch `76`
- observation:
  - this model is not a "very early best" case
  - later training still matters
  - post-80 behavior is a stable plateau rather than collapse

### yolo11s
- best epoch: `77`
- early peak: `no`
- trend:
  - epoch 1-20: weak early operating-point performance
  - epoch 20-80: main improvement window, best checkpoint appears at epoch `77`
  - epoch 80-200: remains competitive, with close follow-up checkpoints around epoch `81`, but no real overtake
- observation:
  - similar to `n`, this is a mid-training best rather than an early one
  - metric curve looks like a short plateau around the best window rather than a single accidental spike

### yolo11m
- best epoch: `78`
- early peak: `no`
- trend:
  - epoch 1-20: stronger early ceiling than `n/s/l`
  - epoch 20-80: strongest rise among all five models, global best appears at epoch `78`
  - epoch 80-200: `Spec@R99.0` remains strong and even reaches a local later high, but overall ranking still favors epoch `78`
- observation:
  - this is the most convincing formal leader
  - not only best overall, but best emerges after substantial training rather than by luck
  - later epochs keep useful gate ability, which suggests real capacity rather than fragile overfit

### yolo11l
- best epoch: `54`
- early peak: `no`
- trend:
  - epoch 1-20: moderate early rise
  - epoch 20-80: main useful window, with best around epochs `47-54`
  - epoch 80-200: declines in `Spec@R99.5` and does not recover enough to retake the lead
- observation:
  - `l` is a mid-training best with visible later degradation on the primary gate anchor
  - this is consistent with "former exploratory favorite, but not formal final leader"

### yolo11x
- best epoch: `125`
- early peak: `no`
- trend:
  - epoch 1-20: best early-stage gate signal among the five, but still not global-best quality
  - epoch 20-80: already strong, with an early mature checkpoint near epoch `25`
  - epoch 80-200: keeps improving slowly and reaches global best only at epoch `125`
- observation:
  - `x` is the latest-best model among the five
  - it clearly benefits from longer training
  - it is a strong second-place model, but still does not catch `m`

## Top-3 Checkpoints per Model

### yolo11n
| rank | epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 |
|------|------|------------|------------|------------|------------|
| 1 | 76 | 0.511905 | 0.583333 | 0.922395 | 0.894841 |
| 2 | 59 | 0.511905 | 0.535714 | 0.914286 | 0.902778 |
| 3 | 199 | 0.511905 | 0.523810 | 0.912281 | 0.904762 |

### yolo11s
| rank | epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 |
|------|------|------------|------------|------------|------------|
| 1 | 77 | 0.523810 | 0.535714 | 0.914286 | 0.902778 |
| 2 | 81 | 0.511905 | 0.535714 | 0.914474 | 0.904762 |
| 3 | 74 | 0.511905 | 0.511905 | 0.910870 | 0.912698 |

### yolo11m
| rank | epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 |
|------|------|------------|------------|------------|------------|
| 1 | 78 | 0.595238 | 0.654762 | 0.934831 | 0.882937 |
| 2 | 60 | 0.523810 | 0.571429 | 0.920354 | 0.896825 |
| 3 | 77 | 0.511905 | 0.619048 | 0.928571 | 0.888889 |

### yolo11l
| rank | epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 |
|------|------|------------|------------|------------|------------|
| 1 | 54 | 0.523810 | 0.571429 | 0.920530 | 0.898810 |
| 2 | 48 | 0.488095 | 0.595238 | 0.924444 | 0.892857 |
| 3 | 47 | 0.488095 | 0.523810 | 0.912473 | 0.906746 |

### yolo11x
| rank | epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 |
|------|------|------------|------------|------------|------------|
| 1 | 125 | 0.547619 | 0.595238 | 0.924444 | 0.892857 |
| 2 | 126 | 0.547619 | 0.571429 | 0.920530 | 0.898810 |
| 3 | 95 | 0.547619 | 0.571429 | 0.920354 | 0.896825 |

## Top1 vs Gate-best

### yolo11n
- top1 best epoch: `120`
- top1 best value: `0.92500`
- gate best epoch: `76`
- same or different: `different`

### yolo11s
- top1 best epoch: `88`
- top1 best value: `0.92083`
- gate best epoch: `77`
- same or different: `different`

### yolo11m
- top1 best epoch: `197`
- top1 best value: `0.93472`
- gate best epoch: `78`
- same or different: `different`

### yolo11l
- top1 best epoch: `103`
- top1 best value: `0.93056`
- gate best epoch: `54`
- same or different: `different`

### yolo11x
- top1 best epoch: `120`
- top1 best value: `0.93194`
- gate best epoch: `125`
- same or different: `different`

## Final Ranking (Gate)

1. `yolo11m`
2. `yolo11x`
3. `yolo11l`
4. `yolo11s`
5. `yolo11n`

## Auto Observations
- This first-wave formal gate scan does **not** support the idea that the best model is chosen by a very early checkpoint. All five gate-best epochs are later than epoch `20`.
- `yolo11m` is the most convincing formal leader because it wins on the primary anchor `Spec@R99.5` and also leads the remaining tie-break metrics.
- `yolo11x` is the only model whose formal best appears very late (`125`), which suggests it benefits most from long training among the five.
- `yolo11l` remains strong, but its best window is earlier and its later epochs do not preserve the same primary-anchor strength.
- `yolo11n` and `yolo11s` are not weak in absolute terms, but their ceilings are clearly below `m/x` under the formal gate-aware ranking rule.
- Trainer-side `top1` would **not** have selected the same checkpoint for any of the five models:
  - `n`: `120` vs gate `76`
  - `s`: `88` vs gate `77`
  - `m`: `197` vs gate `78`
  - `l`: `103` vs gate `54`
  - `x`: `120` vs gate `125`
- The strongest anti-top1 evidence appears on `yolo11m`, where trainer `top1` prefers epoch `197` but the formal gate-aware optimum is much earlier at epoch `78`.

## Archive Scope Note
- This report is built from formal gate-aware materials:
  - `binary_gate_capacity_summary.csv`
  - `best_checkpoint_registry.csv`
  - per-model `epoch_gate_summary.csv`
  - per-model `best_epoch_manifest.json`
- It also uses trainer-side `results.csv` copied inside the formal archive under `trainer_run_copies/.../results.csv` only to compare `top1-best` with `gate-best`.
- This report does **not** rely on screenshot evidence or trainer-generated `best.pt`.
