# SCTSR v4 logical-job lease 与 fencing 修复附录

状态：`CODE_IMPLEMENTED_NOT_TRAINING_AUTHORIZATION`

本附录只修复正式执行控制面的并发与恢复安全性，不改变八臂、T/R1/R2、样本预算、schedule、seed、endpoint 或科学 estimand。它取代
`SCTSR_V4_READINESS_ADDENDUM_20260814.md` 中“仅按 execution nonce 排他即可避免重复逻辑作业”的旧实现口径；旧文档保留为历史证据，不删除。

## 1. 逻辑作业身份

同一个正式逻辑作业由以下不可变字段共同确定：

- `run_role`
- `logical_run_id`
- `arm_id`
- `training_seed`
- `output_root_digest`
- `parent_checkpoint_sha256`
- `lineage_digest`
- `schedule_digest`

`START` 与后续 `RESUME` 的 `action`、`resume_checkpoint_sha256` 和
`resume_from_receipt_digest` 不进入该稳定身份；因此不同 token 或 nonce 不能绕过同一逻辑作业的排他约束。

## 2. claim、lease 与 fence

每次有效尝试仍先消费一次性 token nonce，同时必须在共享 claim registry 中追加一个逻辑作业 fence：

- 第一个 `START` 只能创建 `fence_generation=1`；已有任意 fence 时，第二个 `START` 必须返回 `LOGICAL_JOB_LEASE_ACTIVE`。
- `RESUME` 必须已有 prior fence，且 prior heartbeat 已明确失败或超过 21,600 秒未续租；fresh lease 禁止接管。
- 合法 `RESUME` 只能以 exclusive-create 追加下一代 fence。两个并发 RESUME 读取同一代时只有一个能创建下一代。
- 每个 fence 绑定上一代 fence digest、execution claim SHA、完整稳定逻辑身份和 execution ID，形成连续 append-only chain。
- 最新 fence 是唯一有权发布 epoch 的 attempt；旧 attempt 返回 `LOGICAL_JOB_FENCED`。

共享 registry 中的逻辑作业控制锁只覆盖 claim/fence 更新和 epoch canonical publication 的短临界区。锁目录若因机器在临界区内死亡而残留，系统必须 fail closed，并由操作员核对 owner、claim、fence、receipt 后处理；代码不得猜测或自动删除。

## 3. heartbeat 与接管边界

claim 时生成 `ACTIVE` heartbeat；每次 epoch canonical commit 前，在持有同一逻辑作业控制锁时重新验证最新 fence 并续租 heartbeat。六小时超时远大于预期单 epoch 时间，用于防止活跃进程被普通 RESUME 干扰，同时允许断电/kill 后由新签名 token 在审计后恢复。

即使旧进程在新 fence 创建时仍处于 epoch 计算中，它在 publication critical section 会重新检查 fence；发现自己不是最新 generation 后不能 rename generation、追加 receipt 或更新 recovery pointer。

## 4. RESUME 的只读/变更顺序

正式 runner 必须严格按以下顺序执行：

1. `inspect_formal_resume_context` 只读核验 checkpoint、generation chain、receipt chain、RNG、history、磁盘需求和 resume epoch；不得移动 `.inprogress` 或重建 metadata。
2. 用只读预检结果构造并验证 signed RESUME job binding。
3. 在共享 registry 中 claim 新 fence。
4. claim 成功后才调用 `prepare_formal_resume_context`，允许 reconciliation 与 quarantine。
5. 再次比较 checkpoint SHA、receipt-chain digest 和 resume epoch；预检与变更阶段不一致即失败，不构造 trainer。

因此 claim 失败的竞争进程不会移动仍在工作的原 attempt 的 partial generation。

## 5. epoch 发布原子边界

正式 epoch 完成 forward/backward、checkpoint 和 evidence staging 后，必须在逻辑作业控制锁内执行：

1. 重新验证 token claim、registry、完整 fence chain 和当前 generation；
2. 续租 heartbeat；
3. 校验 transaction 文件；
4. `.inprogress` 原子 rename 为 `.complete`；
5. 追加 canonical receipt；
6. 更新 secondary index 与 rolling pointer。

RESUME fence 创建与 epoch 发布使用同一个控制锁，因此不会出现“旧进程通过校验后、新 RESUME 插入 fence、旧进程仍发布”的检查/使用竞态。

## 6. schema 变更

- `formal_execution_claim`: `stage1.sctsr.formal_execution_claim.v2`
- `execution_attempt_snapshot`: `stage1.sctsr.execution_attempt_snapshot.v2`
- `logical_job_fence`: `stage1.sctsr.logical_job_fence.v1`
- `logical_job_heartbeat`: `stage1.sctsr.logical_job_heartbeat.v1`
- `logical_job_control_lock`: `stage1.sctsr.logical_job_control_lock.v1`

正式 release、token、source-tree manifest 与 runbook 必须在最终部署冻结提交上重新生成；此前生成的任何控制面 SHA 不得沿用。

## 7. 本回滚单元测试

失败优先测试覆盖：

- 两个不同 nonce/token 绑定同一逻辑作业时只有一个 START 成功；
- 不同逻辑作业仍可各自 claim；
- fresh heartbeat 拒绝 RESUME；
- stale heartbeat 允许唯一下一代 RESUME；
- 新 fence 产生后旧 binding 和旧 epoch publication guard 均返回 `LOGICAL_JOB_FENCED`；
- RESUME 只读预检不移动 `.inprogress`；
- 两个正式 runner 的源码调用顺序均为 preview → claim → mutating prepare；
- execution attempt snapshot 同时复制并校验 immutable fence；
- 全量 SCTSR v4 回归在 Python 3.11 下通过。

本附录不代表正式训练已授权，也不代表共享文件系统已经完成跨两机 exclusive-create 现场探针。
