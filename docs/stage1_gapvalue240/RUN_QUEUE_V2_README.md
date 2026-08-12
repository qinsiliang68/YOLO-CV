# Dynamic Campaign Run Queue v2

## 身份边界

Run queue v2 是科学 job 图的不可变编译产物，绑定：

- preregistration selection/schedule/job graph；
- `JOB_EXECUTION_REGISTRY.csv` SHA256；
- canonical training lock 文件 SHA256；
- monitor manifest 与 selection digest；
- dependency closure 与 release state。

机器放置不属于科学身份。机器授权由独立 assignment v2 决定。

## 旧证据与新正式路径

- 旧 `03_preregistration/`：v1 历史预注册，不满足当前 queue-v2 身份字段要求。
- 新 `03_preregistration_v2/`：当前冻结预注册，包含 30 seeds、80 个 Cycle-1/2 逻辑 run、296 个物理 segment job。
- 旧 `04_run_queue/`：v1 历史证据，308 jobs，不能用于新 release。
- 新 `04_run_queue_v2/`：由当前 preregistration 重新生成，不能覆盖旧目录。
- `RUN_QUEUE_VALIDATION.json` 必须是 `stage1.dynamic_campaign_run_queue.v2` 且携带 canonical lock SHA。

截至 2026-08-08，当前仓库的 v2 产物已经生成并通过内容校验：296 个唯一 job，Cycle 1 的 88 个 job 处于 `ENGINEERING_GATE`，Cycle 2 的 208 个 job 保持 `HELD`，1200 个 OOF-only monitor 样本已冻结。该状态不等于 release 已放行。

## 正式生成命令

```powershell
uv run python scripts/stage1_gapvalue240/build_dynamic_replay_preregistration.py
uv run python scripts/stage1_gapvalue240/build_dynamic_campaign_run_queue.py
```

两个命令只写入并列的 `03_preregistration_v2/` 和 `04_run_queue_v2/`，遇到非空目录会 fail closed，不会覆盖旧证据。

## Dry-generation

```powershell
uv run python scripts/stage1_gapvalue240/dry_generate_dynamic_campaign_v2.py <arguments>
```

链路只生成并验证：

```text
preregistration -> queue_v2 -> engineering_gate_v2
-> release_v2 -> assignment_v2
```

它绝不激活 `ACTIVE_ASSIGNMENT.json`，也不启动训练。

## Release 规则

- pilot 只可包含 Cycle-1 `ENGINEERING_GATE` job；
- dependency 必须在 release 内闭合；
- HELD cycle 不得泄漏；
- release 必须绑定 gate v2、queue registry SHA、canonical lock SHA；
- release 目录保存不可变 `ENGINEERING_GATE_REPORT.json` 副本。

## Assignment 规则

- 每个 release job 生成一条 single-job command；
- 每条 command 恰好一个 `--job-id`；
- whole cycle/seed block 不跨物理机器；
- 改派生成新 assignment 代际并保留 parent SHA/reason；
- assignment 只改变 placement，不改变 job、selection、schedule、seed、dependency 或输出身份。
