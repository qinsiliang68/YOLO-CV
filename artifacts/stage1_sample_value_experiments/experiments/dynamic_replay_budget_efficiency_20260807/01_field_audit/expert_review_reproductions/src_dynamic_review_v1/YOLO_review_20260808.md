# YOLO-CV `push-info-sampling-lite` 最新代码专项审查

审查日期：2026-08-08

## 审查范围

- 仓库：`qinsiliang68/YOLO-CV`
- 分支：`push-info-sampling-lite`
- 当前 HEAD：`3a08d22fb683be409a308e20503822e9d8d38a82`（文档提交）
- 最新运行时代码提交：`0c127f16a136f6e930106a20aec1225cac78647f`
- 冻结矩阵/队列提交：`fe8a775deeff5167fda31776d355db9e1536305d`

审查只针对当前分支最新状态，不使用 `main` 或旧 240-run 代码代替。

## 总结判断

当前代码不应进入正式 pilot。建议将状态从 `CODE_READY_FOR_OWNER_CANARY` 临时降为：

`OWNER_CANARY_BLOCKED_PENDING_P0_FIXES`

共享盘与十机 canary 可以在 P0 修复后执行；不能依赖 canary 自动发现以下全部问题。

## P0：正式放行阻断

### F01：worker 没有核对 release 绑定的 source-tree identity

release/gate 保存了 `source_tree_sha256`，但 worker 只检查当前工作区“tracked clean”，不把当前 source tree 或 Git HEAD 与 release/gate 比较；并且 `git status` 使用 `--untracked-files=no`。因此干净但错误的 commit，或某些未跟踪运行时代码，可越过正式入口。

违反：W01、W06、W07。

修复：worker 在 GPU 前重新生成 source-tree manifest，比较 release 中的 source-tree SHA，并冻结 expected Git commit；正式模式同时检查会参与 import/执行的未跟踪源码。

### F02：lease 实际依赖跨机器 wall clock，可出现第二个 holder

heartbeat 写入本机 `time.time()`，新 claimant 再用另一台机器的 `time.time()` 判断 TTL。时钟快 120 秒的节点可把仍存活但暂停 heartbeat 的 holder 判 stale 并获得第二个 token。

本地最小复现：`clock_skew_second_claim=True`。

违反：W03、W04。

修复：不要用不同节点的 wall clock 直接比较；使用共享文件系统可验证的租约年龄机制、保守 skew budget，或由单一协调时钟/服务决定 TTL。canary 必须注入时钟偏差，而不是只记录 offset。

### F03：共享 coordination 没有 durable completion tombstone

`release(status=COMPLETE)` 会归档并移除 active claim，但 `claim_job_lease()` 不检查共享完成态。完成后另一台机器可以立即再次 claim 同一个 job。机器本地输出存在时通常会 `SKIP_COMPLETE`，但换机器、换 output root 或改派后会重新执行；系统没有自动区分“合法 full-block restart”和“误重复执行”。

本地最小复现：`second_claim_after_complete=True`。

违反：W01、W03、W09。

修复：在 coordination root 原子发布 `canonical_completions/<job_id>.json`，绑定 assignment generation、lease token、result SHA、artifact manifest SHA。新 claim 默认拒绝已完成 job；只有显式 supersession generation 才允许重跑，并必须把旧 completion 标成 superseded。

### F04：cycle closeout 可在全部 job FAILED 时 PASS

`FINAL_STATES` 包含 `COMPLETE/FAILED/FENCED`，closeout 只拒绝非终态；它不要求每个 expected job 恰好一个 canonical COMPLETE。`canonical_completion` 列不存在时也不会报错。

本地最小复现：单个 expected job，唯一事件为 `FAILED`，输出 `status=PASS`。

违反：W09、W11。

修复：每个 expected job 必须恰好一个可验证的 canonical COMPLETE；FAILED/FENCED/SUPERSEDED 只能作为历史 attempt，不能满足 closeout。必须校验 lease、result sidecar、artifact manifest、assignment/release/canonical/source-tree identities。

### F05：关键 val_op prediction 只校验数量，不校验精确样本身份

`_validate_prediction()` 检查列、ID 唯一、标签数量和概率范围，但不将 `sample_id/y_true` 与输入 manifests 做精确集合/顺序/hash 比较。任意错误 sample IDs，只要 defect/normal 数量正确，就能 PASS。

本地最小复现：`WRONG_DEFECT` 与 `WRONG_NORMAL` 两个虚构 ID 在 1/1 计数下通过。

worker 虽验证 machine asset report，但正式执行时仅重新 hash train 与 val_model manifests，没有重新 hash val_op manifests；报告生成后 val_op 文件若漂移，当前 worker 未直接捕获。

违反：W08、W09 及 endpoint 数据身份合同。

修复：prediction sidecar 绑定 defect/normal manifest SHA、期望 sample identity digest、label digest；输出逐行或集合重算必须相等。worker 每次执行关键预测前重新 hash val_op manifests。

## P1：应在 owner canary 前修复

### F06：幂等 `SKIP_COMPLETE` 会覆盖原始完成证据

worker 对相同 `job_results/<job_id>.json` 使用 `overwrite=True`。再次运行完成 job 时，会用新的 assignment、lease token、代码 provenance 覆盖第一次完成记录。原始 canonical completion 证据不再不可变。

修复：首次 canonical result 永久不可覆盖；幂等重放只新增 replay/audit event，并引用原 completion SHA。

### F07：全 epoch telemetry 缺少完整 identity 与 base-role 验收

`validate_all_epoch_telemetry()` 没有校验 sidecar 的 `run_id/arm_id/segment_id/identity_index_sha256`。角色验收只强制 `normal_replay` 和 `defect_guard`，没有强制 `base_normal/base_defect` 的 exposure 与 loss。

分支 clone 会把 parent telemetry sidecars 原样带入 child；child 顶层 audit 改了 run/arm，但 inherited sidecar 仍是 parent identity，而当前 validator 不会拒绝。

违反：W08。

修复：sidecar/role summary 必须绑定 run、arm、segment、identity index、selection、schedule、canonical lock；base normal/defect 的数量及有限 loss 必须逐 epoch 验证。继承数据应使用显式 lineage reference，不能伪装成 child 原生产物。

### F08：branch child 缺少自身完整的六个关键 checkpoint prediction 入口

共享 prefix job 产生 epoch 120/140 prediction；child clone 不复制 `key_checkpoint_predictions`，child 后续只产生 150/160/180/200。仓库没有发现一个正式 lineage resolver，将 parent 120/140 prediction 绑定为 child logical arm 的不可变证据。

文档却要求每个 arm 具备 120/140/150/160/180/200。

修复：生成 logical-run artifact index，明确每个关键 epoch 的物理来源与 SHA；或把 parent prediction 以不可变引用/硬链接加 lineage manifest 发布到 child。

### F09：AIOps 日报/closeout 没有从真实 runtime 自动物化事件

日报需要 `attempt_id/bytes_written_24h/disk_free_bytes` 等列，但 worker job-state 没这些字段，仓库没有实际 job state/lease/result/resource 到 `status_events` 的 materializer。

其他逻辑问题：

- `next_ready_block` 实际选择第一个“没有事件”的 job，不看 dependency，也不是 block；
- `complete_seed_blocks` 只数 COMPLETE 行中出现过多少 block，不验证 block 全部完成；
- stop/scale gate 只是原样嵌入，不计算是否触发；
- daily report 顶层 `status=PASS`，没有严格交叉验证 release/assignment/canonical identities。

违反：W11。

修复：从 coordination claim/completion、job state、result、artifact manifest、resource log 建立 append-only canonical event ledger；ready block 用队列 dependency 与完整 block 状态计算；stop/scale gate输出 evaluated status/reason。

### F10：资源统计存在占位 0，validator 仍会 PASS

`write_seconds` 和 `queue_idle_seconds` 每 epoch 初始化为 0 且没有真实更新；epoch resource aggregation没有计算 disk write throughput。`validate_resource_log()` 只拒绝缺失或负数，200 行全 0 资源表也会 PASS。

本地最小复现：全 0 的 200-epoch resource CSV 输出 PASS。

违反：W10。

修复：记录真实 write/queue idle/disk read-write delta 与采样覆盖数；validator 要求训练 epoch 至少存在资源样本、GPU/CPU/时间分解满足基本守恒和合理性；明确 `interbatch_wait` 只是 DataLoader wait proxy。记录 OOM 清理后的显存。

### F11：冻结 preregistration 仍保存控制机绝对路径

`CANONICAL_TRAINING_LOCK_BINDING.json` 写入 `Path(...).resolve()` 的绝对路径；queue regeneration 会按该路径找 lock。换 repo root/盘符时，即使同一文件与 SHA 在仓库中，也不能从冻结 preregistration 重建 queue。

assignment 同样允许 repo 外 machine config 退化为绝对路径，任务书要求的是 repo-relative path + SHA。

违反：W02、W07。

修复：冻结相对 logical path + SHA；运行时由可信 repo root 解析。正式 assignment validator 禁止 ABSOLUTE machine-config mode。

## P1/P2：科学解释与统计口径问题

### F12：当前“时间效应”同时改变活跃样本集合

queue template 每个 segment 都取 ranking 的 `head(normal_slots)`：

- 2.5% continuous：始终 Top 3000；
- same-peak taper：Top 3000 → Top 2000 → Top 1000 → 0；
- dose-matched taper：Top 4000 → Top 2667 → Top 1333 → 0。

因此对比不仅改变时间/累计曝光，还改变 active selection composition；dose-matched 还引入 ranks 3001–4000。若论文将其解释为纯 timing effect，当前实现不支持该因果解释。

处理方式二选一：

1. 保持固定 core selection，只改变每样本 repeat weight/频率；或
2. 将 estimand 明确改名为“动态 Top-k replay policy”，不再声称纯时间效应。

这需要新版本 preregistration，不能静默改冻结 v2。

### F13：文档称“线性衰减”，实现是两级阶梯

实际 segment slots：

- 0.5% same-peak：600/400/200/0；dose-matched：800/533/267/0；
- 1.0%：1200/800/400/0；dose-matched：1600/1067/533/0；
- 2.5%：3000/2000/1000/0；dose-matched：4000/2667/1333/0。

实现不是 epoch-by-epoch linear taper。若阶梯是冻结科学合同，文档必须改；若线性才是合同，则代码/队列需要新版本重建。

### F14：dose-matched 样本曝光相同，但 optimizer step 不完全相同

因 batch=128 向上取整：

- 0.5% continuous 188600 steps，dose-matched 188500（-100）；
- 1.0% 两者均 189400；
- 2.5% continuous 192200，dose-matched 192250（+50）。

差异很小，但若宣称“唯一变化是 timing”，必须报告并处理。

### F15：梯度探针尚未进入正式 release/worker 链

仓库有 gradient candidate builder 和探针库；文档也明确其为独立 P1 pilot，不是正式 job 默认步骤。当前 296-job queue 不会自动产出关键 checkpoint 梯度结果。

这不是 Cycle 1 运行时 bug，但在声称“已补齐梯度采集”前，还缺冻结 manifest、release、独立命令、结果 validator 和 aggregation binding。

## 测试与验证情况

- GitHub 当前 HEAD 没有 CI/status checks。
- 对与当前 Git blob 完全一致的 lease、AIOps、protocol、queue、assignment 模块运行 6 个相关测试文件：`34 passed`。
- 但上述反例仍可复现，说明现有测试缺失关键负例，而不是测试已经证明系统安全。

应补的首批测试：

1. +TTL 以上跨节点 clock skew 不得产生第二 holder；
2. COMPLETE job 不得在同 generation 再 claim；
3. all FAILED/FENCED cycle 必须 closeout FAIL；
4. clean wrong Git commit/source tree 必须在 GPU 前失败；
5. prediction IDs 与 manifest 不一致必须失败；
6. all-zero resource log 必须失败；
7. child inherited telemetry identity/lineage错误必须失败；
8. SKIP_COMPLETE 不得覆盖原 canonical result。

## 已做得正确的部分

- 单 worker 严格要求恰好一个 `--job-id`，拒绝 batch flags；
- queue-v2/release-v2/assignment-v2/canonical lock 基本哈希链清楚；
- Windows `control.lock` sharing contention 已采用有界重试，并区分真实 ACL；
- assignment 代际与 fencing token 基础结构存在；
- 分段 checkpoint、zero-epoch archive、原子写入、失败注入和 dirty tracked-code 拒绝已实现；
- Cycle 3/4 保持 HELD，blind holdout 仍 UNBOUND；
- Cycle 1/2 不设 R2 是预注册的 common-support 决定，不是漏写代码。

## 建议修复顺序

1. F01–F05：source tree、lease/completion、closeout、prediction identity；
2. F06–F08：不可变完成证据、telemetry/branch lineage；
3. F09–F10：真实 AIOps event ledger 与资源闭环；
4. F11：路径可移植；
5. F12–F14：由负责人决定新 prereg 版本中的 estimand；
6. F15：独立 gradient pilot 接入。

在 F01–F05 修复并添加负例测试前，不建议构建正式 engineering gate 或 pilot release。
