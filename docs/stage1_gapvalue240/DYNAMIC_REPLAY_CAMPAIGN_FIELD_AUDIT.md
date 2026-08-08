# 动态回流实验字段审计与容量约束

## 研究位置

新实验 ID 为 `dynamic_replay_budget_efficiency_20260807`，与既有 40-run、120-run、OOF/240-run
实验并列。旧实验只作为不可变证据源，不向旧目录追加新训练结果。

新实验回答的不是“哪一张照片有固定价值”，而是：

```text
Value = V(selection, model_state, training_stage,
          replay_schedule, seed, surrounding_samples)
```

具体检验动态衰减回流和弱缺陷保护，能否让同一份有限回流预算在新 seed 上稳定产生双端收益。

## 为什么要重新做字段审计

旧 `DATA_USAGE_LEDGER_REFINED.csv` 已完整审计 240-run 包内字段，但它不能单独回答：

- 配置的回流权重是否等于实际抽到的 replay 次数；
- normal replay 的 loss 下降是否以 weak defect 的 loss/score 下降为代价；
- 同一 selection 在不同 seed 中的反转发生在哪个 epoch；
- 反转来自采样顺序、增强实现、优化器状态，还是梯度方向；
- 相对随机回流的变化是否真由 replay 造成，因为旧矩阵没有 no-replay arm；
- 机制结论能否通过未打开的 blind holdout。

因此新账本给每个字段补齐定义、原始/派生、粒度、时间点、覆盖率、空值率、单位、可复现性、
泄漏风险、假设映射、采集成本、P0-P3 优先级和缺口状态。

## 正式生成

先初始化并列实验目录：

```powershell
uv run python scripts/stage1_gapvalue240/initialize_dynamic_replay_campaign.py
```

再从旧 240-run 审计账本和完整 OOF 数据生成新字段审计：

```powershell
uv run python scripts/stage1_gapvalue240/audit_dynamic_replay_campaign_fields.py
```

产物固定写入：

```text
artifacts/stage1_sample_value_experiments/experiments/
  dynamic_replay_budget_efficiency_20260807/
    01_field_audit/
      FIELD_INVENTORY.csv
      FIELD_GAP_REPORT.md
      DATA_LINEAGE.json
      DATA_VOLUME_FORECAST.csv
      RETENTION_POLICY.md
      AUDIT_VALIDATION.json
```

## P0 启动门槛

84-run confirmatory matrix 启动前必须先在 pilot 中验证并保存：

1. `NR_NO_REPLAY` 因果对照；
2. 每 epoch 的 replay/guard 配置权重；
3. 每样本实际 replay exposure count；
4. base、normal replay、defect guard 的分角色 loss；
5. 固定 difficult-normal 和 weak-defect probe membership；
6. epoch `120, 140, 150, 160, 180, 200` 的 checkpoint 与 raw val_op 预测；
7. checkpoint/threshold 选择规则和完整来源；
8. 最终只打开一次的 blind/external holdout。

梯度字段属于 P1 pilot：先测最后一层梯度的耗时、显存、维度、压缩率和预测价值，再决定是否扩展。
不得因为梯度听起来“更底层”就让 10 台机器全量空转。

## 存储规则

容量按物理路径核算，绝不把同一路径的字节乘以字段数。冻结输入 manifest 和 probe/selection
manifest 只保存一份，run 目录仅记录 hash 引用。阈值扫描、图表和 HTML 从 raw prediction 重建，
不在每个 run 中重复保存。

当前容量数字是下界，因为梯度 payload 尚未实测。任何扩大到 22 或 30 seeds 的决定，都必须先更新
`DATA_VOLUME_FORECAST.csv`，并证明中央存储、上传带宽和失败重试窗口可承受。

## 时间边界

- 8 月 7-9 日：字段审计、文献证据、采集代码和容量 benchmark；
- 8 月 10-13 日：六臂小规模 pilot，验证 P0 字段和单 run 时长；
- 8 月 14-31 日：默认 `6 arms x 14 unseen seeds = 84 runs`；
- 9 月 1-5 日：失败重试、产物汇总和梯度补充 pilot；
- 9 月 6-8 日：冻结协议后执行 blind evaluation；
- 9 月 9-10 日：校验 hash、补缺、上传和归档，不再启动长训练。

## 正式 v2 施工补充（2026-08-08）

本字段审计文档描述科学字段与保留策略；正式执行面、控制面和路径身份以以下 v2 文档为准：

- `docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md`
- `docs/stage1_gapvalue240/RUN_QUEUE_V2_README.md`
- `configs/stage1_gapvalue240/README_MACHINE_CONFIG.md`

路径角色明确区分：

```text
历史 v1 queue/release/assignment：只读证据，不得激活新 campaign
新 v2 queue/release/assignment：版本化 sibling，engineering gate v2 后 dry-generate
coordination root：机器外部共享根，仅保存 claim/lease/heartbeat/fencing 状态
run output root：每个物理 job 的原子、不可覆盖产物根
```

训练机只允许填写机器路径和资源字段；不得修改科学 queue、release、assignment、canonical lock、
replay schedule、seed 或 arm。每个物理 job 必须通过单任务 worker 独立进程执行，controller 仅为可选调度层。
