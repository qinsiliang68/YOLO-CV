# Stage1 Dynamic Replay Campaign v2 运维手册

## 1. 范围与不可变边界

本手册只覆盖正式训练前的控制面、单任务训练面、验证、故障接管与只读汇总。科学矩阵、训练超参数、arm、终点和统计口径来自冻结 preregistration 与 `CANONICAL_TRAINING_LOCK_v1.json`，训练机不得修改。

旧 `DESIGN_EVIDENCE/04_run_queue/` 是 v1 历史证据；新链必须输出到新的版本化 sibling，例如 `04_run_queue_v2/`、`05_engineering_gate_v2/`、`06_releases_v2/`、`07_assignment_v2/`，不得覆盖旧目录。

## 2. 控制面与训练面

```text
preregistration -> queue_v2 -> engineering_gate_v2 -> release_v2
                                             |
                                             v
                                     assignment_v2
                                             |
                              ACTIVE_ASSIGNMENT.json
                                             |
          +------------------ shared coordination root ------------------+
          | claim / lease / heartbeat / fencing / canonical completion   |
          +------------------------+--------------------------------------+
                                   |
                    single-job worker (one process, one job)
                                   |
                       checkpoint / telemetry / result
```

控制器只是可选调度便利层。它不得成为唯一入口，也不得把多个 job 合并到一个训练进程。控制器退出不终止已经领取的 worker；恢复后通过共享完成态和 lease 判断下一项。

## 3. 单物理 job 命令

每个 release job 在 assignment 中生成一条独立命令，命令必须只有一个 `--job-id`：

```powershell
uv run python scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py `
  --machine-config configs/stage1_gapvalue240/machines/machine_01.yaml `
  --campaign-root artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807 `
  --job-id <ONE_PHYSICAL_JOB_ID> `
  --release <release-v2-json> `
  --assignment <assignment-v2-manifest> `
  --expected-release-id <release-id> `
  --expected-canonical-lock-sha256 <64-hex>
```

禁止 `--job-list`、`--job-range`、`--count`、`--max-jobs`、`--next-job`，也禁止重复传入 `--job-id`。进程完成一个 job 后必须退出。完成态幂等跳过；合法中断按 sidecar 校验恢复；非法半写直接失败。

## 4. Assignment 改派

科学 queue/release 不随机器变化。整块改派只生成新的 assignment 目录，并保留 parent assignment SHA 与 reason：

```powershell
uv run python scripts/stage1_gapvalue240/build_dynamic_campaign_assignment.py `
  --queue-dir <queue-v2> `
  --release <release-v2-json> `
  --output-dir <new-assignment-dir> `
  --assignment-id ASSIGNMENT_V2_R002 `
  --machine-configs-dir configs/stage1_gapvalue240/machines `
  --slot-map configs/stage1_gapvalue240/DYNAMIC_MACHINE_SLOT_MAP_v1.csv `
  --seed-override S001=machine_11 `
  --supersedes-assignment <old-assignment-manifest> `
  --reassignment-reason "machine_01 unavailable"
```

同一 cycle/seed block 必须落在一台机器。queue 中的 planning slot 只是提示；assignment v2 以第一个 queue-order slot 为确定性锚点并合并整个 seed block。旧 assignment 永久保留。

激活前先确认没有活跃 claim：

```powershell
uv run python scripts/stage1_gapvalue240/activate_dynamic_campaign_assignment.py `
  --coordination-root <shared-root> `
  --assignment <new-assignment-manifest>
```

## 5. 共享 coordination root canary

先为十台机器生成独立命令：

```powershell
uv run python scripts/stage1_gapvalue240/build_coordination_root_canary_commands.py `
  --machine-configs-dir configs/stage1_gapvalue240/machines `
  --shared-root <shared-root> `
  --campaign-id <campaign-id> `
  --canary-generation <generation-id> `
  --output-dir <canary-command-dir>
```

每台机器只执行自己的命令。节点报告检查 read/write/create/rename/delete、O_EXCL 唯一竞争、跨节点 token/hash 可见性、时钟偏移诊断与 root identity。聚合必须 `10/10`：

```powershell
uv run python scripts/stage1_gapvalue240/aggregate_coordination_root_canary.py `
  --reports-dir <node-reports-dir> `
  --expected-machine-ids machine_01,machine_02,machine_03,machine_04,machine_05,machine_06,machine_07,machine_08,machine_09,machine_10 `
  --output <TEN_MACHINE_COORDINATION_CANARY.json>
```

lease 不依赖精确同步时钟；clock offset 只用于诊断。

## 6. Lease 与 fencing 状态机

```text
UNCLAIMED -> CLAIMED -> RUNNING -> COMPLETE
                    \-> FAILED
CLAIMED/RUNNING --TTL expired--> STALE -> REAPED -> new CLAIMED
old generation holder --assignment switch--> FENCED
```

claim 使用 exclusive create，拥有随机 fencing token。heartbeat 与完成发布必须同时匹配 active assignment SHA 和 fencing token。旧 holder 在新 assignment 激活后不得 heartbeat 或发布完成态。Windows sharing contention 使用有界重试并规范化为 `LockHeldError`；真实 ACL/目录权限错误不得被吞掉。

## 7. 十机 one-job real-data canary

生成十条命令，每台机器只跑一个冻结真实轻量子集的一 epoch：

```powershell
uv run python scripts/stage1_gapvalue240/build_ten_machine_real_data_canary_commands.py `
  --standalone-commands <assignment/STANDALONE_JOB_COMMANDS.csv> `
  --machine-configs-dir configs/stage1_gapvalue240/machines `
  --output-dir <real-data-canary-dir>
```

每机必须验证 canonical lock、machine config SHA、real-image identity、workers=4、单 job lease、逐 epoch telemetry、checkpoint/sidecar 原子结果、GPU/CPU/RAM/disk、退出后显存和 child workers 释放。聚合严格要求 `10/10`。

## 8. Engineering gate 与 v2 release 顺序

放行顺序不可跳过：

1. canonical lock validation；
2. queue v2 validation；
3. standalone entry validation；
4. assignment reassignment validation；
5. source tree immutability validation；
6. local real-data smoke；
7. crossed numerical parity；
8. failure injection；
9. all-epoch telemetry；
10. lease concurrency；
11. lease fencing；
12. coordination root canary；
13. ten-machine real-data canary；
14. disk/GPU preflight；
15. documentation handoff validation；
16. build engineering gate v2；
17. build pilot release v2；
18. build assignment v2；
19. activation；
20. only small pilot.

Gate 必须重新读取 envelope 与底层 payload，校验 schema/status/path/hash/source-tree/queue/canonical identity。人工写一个顶层 `PASS` 无效。旧 gate v1 对新 release 明确拒绝。

## 9. 热备与恢复

默认策略是整 cycle/seed block 从 canonical 初始状态在热备机器重跑。原机器已完成的孤立 arm 标记 `SUPERSEDED/FENCED`，不进入正式配对统计。跨机 resume 默认禁止；只有 model、EMA、optimizer、scaler、RNG、sampler、workspace、telemetry 边界的完整状态包 validator 全部通过后才允许显式开启。

失败 attempt 永久保留。聚合只接受一个身份、sidecar、lease 和 artifact manifest 均通过的 canonical completion。

## 10. 全 epoch 与资源验收

低成本字段必须覆盖 epoch `1..200` 且唯一：base/replay/guard exposure、step、分角色 loss、LR、资源、阶段时间。动态衰减离散 slots 和累计面积必须与 prereg 一致。no-replay 的 replay exposure 必须严格为零。

重型产物只在冻结关键 epoch 保存。资源 preflight 不得自动改变 batch=128、workers=4 或任何科学参数。

## 11. 日报与周期收口

`DAILY_CAMPAIGN_STATUS.json/.md` 是只读聚合，不创建 arm，不读取 blind/external。`CYCLE_CLOSEOUT_VALIDATION.json` 只在 job set 完整、状态终结、每 job 只有一个 canonical completion 时 PASS。

## 12. 故障排查

- `LockHeldError`：正常竞争 loser；检查唯一 winner 和残留 lock。
- ACL/PermissionError：检查共享目录权限，不得无限重试。
- gate identity mismatch：重新生成对应 evidence envelope，不要手改 SHA。
- queue/release schema v1：只读历史证据，重新生成 v2。
- half-write：删除可重建临时文件，保留 failed attempt；禁止把 `.tmp` 当 COMPLETE。
- GPU preflight NOT_RUN：代码可继续审查，但 owner canary 不得标 PASS。
- assignment switch blocked：等待 live claim 结束或按 TTL/fencing 流程处理，不得强行覆盖。

## 13. 术语兼容

英文运维记录中，外层调度器写作 `controller`。热备默认恢复策略写作 `full-block restart`，含义仍是整 cycle/seed block 从 canonical 初始状态重跑，而不是跨机拼接孤立 arm。
