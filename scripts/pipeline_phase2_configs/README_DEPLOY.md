# Phase 2 Pipeline 部署说明

## 一句话

把仓库代码 pull 到 ASUS 机器,跑 `python scripts/run_phase2_pipeline.py --phase all --device 0`,等它自己跑完 21 个实验。

## 前置条件

1. 仓库代码 `git pull` 到最新
2. `YOLOv11/datasets/sewerml_gate2_train7200/` 数据集在位
3. yolo11m-cls hn00 best checkpoint (epoch_078.pt) 可被代码访问
4. `research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/candidate_pool_master.csv` 存在(250 样本的 R/C/D 分数)
5. Python 环境: `pip install ultralytics torch matplotlib numpy`

## 运行命令

### 单台机器一口气跑完(推荐)

```bash
cd /path/to/YOLO-CV
python scripts/run_phase2_pipeline.py --phase all --device 0
```

预计耗时: ~75 小时(21 run × ~3.5h/run)

### 查看实验列表(不执行)

```bash
python scripts/run_phase2_pipeline.py --list
```

### 按阶段分机器跑

```bash
# 机器 1: 定峰 R+C (4 run, ~14h)
python scripts/run_phase2_pipeline.py --phase peak --experiment p1_R_Q2a --device 0
python scripts/run_phase2_pipeline.py --phase peak --experiment p1_R_Q2b --device 0
python scripts/run_phase2_pipeline.py --phase peak --experiment p1_C_Q4a --device 0
python scripts/run_phase2_pipeline.py --phase peak --experiment p1_C_Q4b --device 0

# 机器 2: 定峰 D+T (7 run, ~25h)
python scripts/run_phase2_pipeline.py --phase peak --experiment p1_D_Q3a --device 0
# ... 以此类推

# 机器 3: 定效 (4 run, ~14h)
python scripts/run_phase2_pipeline.py --phase validate --device 0

# 机器 4: Robustness (6 run, ~21h)
python scripts/run_phase2_pipeline.py --phase robustness --device 0
```

### 跑单个实验

```bash
python scripts/run_phase2_pipeline.py --experiment p1_R_Q2a --device 0
```

## 21 个实验清单

### Phase 1: 定峰 (11 run) — 找到每个信号的 Goldilocks 精确位置

| # | ID | 信号 | Rank | 回答的问题 |
|---|-----|------|------|-----------|
| 1 | p1_R_Q2a | R (边界风险) | 51-75 | R 的 peak 在 Q2 上半还是下半? |
| 2 | p1_R_Q2b | R | 76-100 | 同上 |
| 3 | p1_D_Q3a | D (抱团密度) | 101-125 | D 的 peak 在 Q3 上半还是下半? |
| 4 | p1_D_Q3b | D | 126-150 | 同上 |
| 5 | p1_C_Q4a | C (TTA稳定性) | 151-175 | C 的 peak 在 Q4 上半还是下半? |
| 6 | p1_C_Q4b | C | 176-200 | 同上 |
| 7 | p1_T_Q1 | T (训练动力学) | 1-50 | 跨 epoch 最不稳定的样本有没有价值? |
| 8 | p1_T_Q2 | T | 51-100 | 中等稳定性 |
| 9 | p1_T_Q3 | T | 101-150 | |
| 10 | p1_T_Q4 | T | 151-200 | |
| 11 | p1_T_Q5 | T | 201-250 | 最稳定的样本有没有价值? |

### Phase 2: 定效 (4 run) — 证明方法有用

| # | ID | 目的 |
|---|-----|------|
| 12 | p2_closedloop | 用 Goldilocks 函数选样训练, 打赢 G0 = 闭环成功 |
| 13 | p2_seed2_R_Q2 | R-Q2 换 seed 重跑, 排除偶然性 |
| 14 | p2_seed3_R_Q2 | 同上 |
| 15 | p2_seed4_R_Q2 | 3 个 seed 算 mean±std |

### Phase 3: Robustness (6 run) — 证明温度校准教师更好

| # | ID | 目的 |
|---|-----|------|
| 16 | p3_top1_R_Q1 | 用 acc top1 checkpoint 当教师算 R, 看 Q1 |
| 17 | p3_top1_R_Q2 | Q2 还是不是最好? |
| 18 | p3_top1_R_Q3 | |
| 19 | p3_top1_R_Q4 | |
| 20 | p3_top1_R_Q5 | |
| 21 | p3_top1_G0 | 均匀基线对照 |

## 输出位置

- 每个实验的结果: `research/materials/stage1_formal/gate_phase2/{experiment_id}/`
- 汇总 CSV: `research/results/stage1_formal/gate_phase2/phase2_summary.csv`
- 训练权重: `YOLOv11/runs/stage1_phase2/{experiment_id}/`
- 数据集视图: `YOLOv11/datasets/stage1_phase2/{experiment_id}/`

## 故障处理

- 如果某个实验失败, pipeline 会跳过继续下一个, 最后报告哪些失败了
- 已完成的实验不会重复执行(检测到 last.pt 和 best_epoch_manifest.json 就跳过)
- 如果需要重跑某个实验, 删掉对应的 `runs/stage1_phase2/{id}/` 和 `materials/gate_phase2/{id}/` 目录

## 注意

- 训练动力学信号 (T) 需要从 hn00 的 per_epoch_gate 逐样本预测中提取
- 如果 `research/materials/stage1_formal/gate_hn_m_sweep/hn00/per_epoch_gate/` 不存在, T 信号会退化为 R 信号的代理
- Phase 2 的闭环验证当前使用 R-Q2 作为代理, 等 Phase 1 结果出来后需要更新 Goldilocks 函数参数
