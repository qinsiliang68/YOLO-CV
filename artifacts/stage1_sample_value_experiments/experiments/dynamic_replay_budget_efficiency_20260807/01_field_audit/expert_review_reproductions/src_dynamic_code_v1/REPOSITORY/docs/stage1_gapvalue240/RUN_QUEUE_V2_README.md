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

- 旧 `04_run_queue/`：v1 历史证据，308 jobs，不能用于新 release。
- 新 `04_run_queue_v2/`：由当前 preregistration 重新生成，不能覆盖旧目录。
- `RUN_QUEUE_VALIDATION.json` 必须是 `stage1.dynamic_campaign_run_queue.v2` 且携带 canonical lock SHA。

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
