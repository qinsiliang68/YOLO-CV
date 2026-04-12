# Goldilocks Campaign 部署说明 (78 run)

## 我们在干什么

精确定位四个样本价值信号(R/D/T/C)各自的 Goldilocks zone,然后组合成最终的高价值样本判别函数。

## 实验总量

| 阶段 | 内容 | run 数 |
|------|------|--------|
| 定峰 R | 边界性滑窗 (step=20) | 11 |
| 定峰 T | 训练动力学滑窗 | 11 |
| 定峰 D | 抱团性 k-sweep (k={8,15,25} × 11) | 33 |
| 定峰 C | 扰动稳定性滑窗 | 11 |
| 随机对照 | Rand50 × 3 seed | 3 |
| 组合验证 | 单路/双路/三路/四路 | 9 |
| **总计** | | **78** |

## 数据依赖 (跑之前必须确认)

1. **数据集**: `YOLOv11/datasets/sewerml_gate2_train7200/` 在位
2. **教师权重**: `epoch_078.pt` 可访问 (yolo11m hn00 gate-best)
3. **候选池**: `research/materials/stage1_formal/gate_bucket_pilot/score_inputs/candidate_pool_master.csv`
4. **T 信号依赖** (机器 B 可选但推荐): `research/materials/stage1_formal/gate_hn_m_sweep/hn00/per_epoch_gate/` 目录下有逐 epoch 的预测文件。如果不存在,T 信号退化为 R 的代理,11 个 T run 白跑
5. **D k-sweep 依赖** (机器 A 推荐): `$out/scratch/stage1_formal/gate_info_sampling_lite/teacher_train_features/train_embeddings.npy`。如果不存在,D 的三个 k 值用同一组 D 值,k-sweep 无效

## 两台机器分工

### 机器 A: R + T + 对照 (25 run, ~2.7 天)

```powershell
git pull origin push-info-sampling-lite
uv run main_goldilocks_machineA.py
```

跑: R 定峰(11) + T 定峰(11) + 随机对照(3) + 本机可完成的组合(F-R, F-T, F-RT)

### 机器 B: D + C (44 run, ~4.8 天)

```powershell
git pull origin push-info-sampling-lite
uv run main_goldilocks_machineB.py
```

跑: D k-sweep(33) + C 定峰(11) + 本机可完成的组合(F-D, F-C)

### 两台都跑完后: 合并 + 跨机器组合 (在任意一台)

```powershell
# 1. 把另一台的 peak_results.json 合并过来
# (通过 git push/pull 或手动拷贝)
# 路径: research/results/stage1_formal/gate_goldilocks_campaign/peak_results.json

# 2. 跑跨机器组合验证
uv run main_goldilocks_postmerge.py
```

跑: F-RD, F-TD, F-RTD, F-RTDC (需要两台机器的 peak 结果)

## 断点续跑

任何时候中断,重新运行同一命令即可。已完成的 run 自动跳过。

## 输出

- 每个 run 的结果: `research/materials/stage1_formal/gate_goldilocks_campaign/{exp_id}/`
- Peak 结果缓存: `research/results/stage1_formal/gate_goldilocks_campaign/peak_results.json`
- 汇总 CSV: `research/results/stage1_formal/gate_goldilocks_campaign/goldilocks_campaign_summary.csv`
- 训练权重: `YOLOv11/runs/stage1_goldilocks/{exp_id}/`
