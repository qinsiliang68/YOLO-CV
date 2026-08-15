# SCTSR v4 简化随机部署补充合同（2026-08-15）

## 1. 决策与边界

本补充合同记录所有者在 2026-08-15 的明确决定：SCTSR v4 不建立 GPU UUID、驱动版本、主机证明或本地 GPU 文件锁控制面。部署只使用简单的机器 ID 和可复现随机分配。

这一简化不改变科学合同、八臂训练逻辑、共同父模型、固定 base step、R2 addendum、固定 E200 endpoint 或 test/blind 禁令。它只改变 13 台 RTX 3090 的作业分派方式。

物理并发的剩余操作风险由部署者承担：每台机器在同一 wave 只能启动计划中的一个 job。代码仍使用共享 execution claim registry 和 logical-job fencing，因而同一 logical job 不能被两个活进程同时合法 claim。

## 2. 机器口径

- 机器总数：13。
- 活跃机器：12，使用部署者提供的 12 个唯一 `machine_id`。
- 缓冲机器：1，使用独立 `buffer_machine_id`。
- 缓冲机器不会出现在常规 placement 中；需要替补时，部署者停止故障机器并重新生成计划，或人工将缓冲机作为新的 active machine ID 后重新冻结部署证据。
- `machine_id` 是部署标签，不声称绑定 GPU UUID、主板、驱动或不可迁移的物理身份。

## 3. 作业与依赖

`build_phase1_logical_jobs` 只接受恰好 8 个 discovery seed 和 14 个 confirmation seed，并生成：

- 22 个 E1–E120 common-parent jobs；
- 每个 seed 对应 8 个 E121–E200 branch jobs，共 176 个；
- 合计 198 个 logical jobs；
- 每个 branch 只依赖同 seed 的 `PARENT_<training_seed>`；
- phase 顺序固定为 `DISCOVERY_PARENT`、`DISCOVERY_BRANCH`、`CONFIRMATION_PARENT`、`CONFIRMATION_BRANCH`。

生成计划不等于允许跨阶段自动推进。Discovery 结果必须按科学合同完成判定后，才能由所有者签发 confirmation 所需的后续执行凭证。

## 4. 随机分配算法

公开策略名为 `SEEDED_SHUFFLED_WAVES_12_ACTIVE_PLUS_1_BUFFER`：

1. 每个 phase 内按 `job_id` 排序；
2. 使用显式 `assignment_seed` 初始化独立 PRNG；
3. 随机打乱该 phase 的 jobs；
4. 每 12 个 jobs 构成一个 wave；
5. 每个 wave 再随机打乱 12 个 active machine IDs；
6. 一个 wave 内每台机器至多获得一个 job；
7. 第 13 台 buffer 不参与常规分配。

相同 seeds、机器 ID、buffer ID 和 `assignment_seed` 必须产生逐字节相同的 plan；更换 `assignment_seed` 会改变 placement。计划保存完整 placement、依赖、wave、机器 ID 和 `plan_digest`。

## 5. 生成命令

以下命令只生成 `PLANNED_NOT_RELEASED` 部署计划，不会启动训练、签发 release/token 或创建 claim：

```powershell
uv run python scripts/stage1_sctsr_v4/build_deployment_plan.py `
  --seed-registry <OWNER_FROZEN_SEED_REGISTRY.json> `
  --active-machine RTX3090_01 `
  --active-machine RTX3090_02 `
  --active-machine RTX3090_03 `
  --active-machine RTX3090_04 `
  --active-machine RTX3090_05 `
  --active-machine RTX3090_06 `
  --active-machine RTX3090_07 `
  --active-machine RTX3090_08 `
  --active-machine RTX3090_09 `
  --active-machine RTX3090_10 `
  --active-machine RTX3090_11 `
  --active-machine RTX3090_12 `
  --buffer-machine RTX3090_13_BUFFER `
  --assignment-seed <OWNER_SELECTED_NONNEGATIVE_INT64> `
  --plan-output <DEPLOYMENT_PLAN.json> `
  --output <BUILD_DEPLOYMENT_PLAN_RECEIPT.json>
```

机器 ID 示例仅表示格式，正式计划必须替换为部署者实际使用的 13 个标签。生成后的计划仍包含：

- `formal_assignments_generated=false`；
- `formal_training_started=false`；
- `release_authorization_required=true`。

## 6. 启动纪律

每台训练机 AI 必须：

1. 读取自己的 placement 行并确认 `machine_id`、`phase`、`wave`、`job_id`、`training_seed`、`arm_id` 和 `depends_on`；
2. branch 启动前验证对应 parent 已 canonical complete；
3. 用相同 `machine_id` 生成 run-intent acknowledgement；
4. 使用该 logical job 的单次 execution token 和共享 claim registry；
5. 只有 claim 成功后才构造 trainer；
6. 同一 wave 完成或明确失败后再进入下一 wave；
7. 失败恢复使用新的 RESUME token，不复用 START token；
8. 不调用旧 `dynamic_campaign_train_worker.py --job-id`。

没有 GPU 文件锁意味着部署者不得在同一物理 GPU 上手工并行启动两个不同 logical jobs。此风险不影响 logical-job claim 的防重，但会导致显存竞争、OOM 或性能数据失真，因此属于明确保留的操作要求。

## 7. 代码与测试映射

- 计划实现：`stage1_sctsr_v4/deployment_plan.py`。
- 生成 CLI：`scripts/stage1_sctsr_v4/build_deployment_plan.py`。
- 失败优先测试：`tests/stage1_sctsr_v4/test_deployment_plan.py`。
- 公共 schema：`stage1.sctsr.deployment_plan.v1`。
- 逻辑 job 防重：`stage1_sctsr_v4/formal_execution.py` 中的 claim registry、lease 与 fencing。

本补充合同不声称任何 SCTSR arm 有效，也不授权正式训练。它只把所有者批准的简化部署语义写成可复现代码和可审计计划。
