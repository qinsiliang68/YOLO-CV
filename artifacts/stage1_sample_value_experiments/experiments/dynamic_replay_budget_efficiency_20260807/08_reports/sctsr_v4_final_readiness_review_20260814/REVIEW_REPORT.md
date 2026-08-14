# SCTSR v4 正式训练前代码 Review 与修复定案

日期：2026-08-14
审查角色：`CODEX_PRIMARY_AGENT_SELF_REVIEW_NOT_INDEPENDENT`
审查方式：独立 clean worktree、怀疑式逐链路检查、失败优先修复、真实工程 canary；未使用子代理。

## 1. 最终结论

本轮得到两个必须同时阅读、不能相互替代的结论：

1. **代码层：`CODE_REVIEW_PASS_NOT_TRAINING_AUTHORIZATION`。**
   - 之前 review 的 3 个 P1 已按失败优先测试修复；
   - 本轮继续发现并修复了资产物理字节、异常清理、run-intent、工程 canary、Windows 长路径和仓库审计语义等缺口；
   - 当前没有未解决的 P0/P1 实现缺陷；
   - Python 3.11 与 3.12 各 `395 passed`，v4 无核心 skip；
   - 真实 Sewer-ML 图片与真实 `yolo11l-cls.pt` 已完成一次工程级 forward、base/replay backward、optimizer、EMA、Zstd Parquet、checkpoint、评价和故障恢复；
   - `stage1_gapvalue240`、`stage1_dynamic_replay_v3`、`YOLOv11/ultralytics` 相对历史基线没有内容变化。

2. **正式训练层：`FORMAL_TRAINING_READINESS=BLOCKED`。**
   - 冻结的 R2 定义要求 3,000 个 unique IDs、与 T 零身份重叠，并对 `label + historical_dynamic_bucket + oof_fold + oof_group_id` 做精确联合配额；
   - 冻结资产实际有 172 个不足 strata、缺 378 个 occurrence，其中 30 个 strata 完全没有零重叠候选；
   - 因此八臂矩阵目前无法构造；代码正确地返回 `R2_QUOTA_INFEASIBLE`，没有偷偷放宽；
   - 推荐的最低改动方案会改变 R2 的科学 estimand，必须由研究负责人批准 addendum，不能由 review 代码替你决定；
   - 正式 release、8 个 discovery seeds、per-job one-use token 和 10 台机器共享 claim registry 也尚未签发，这是 R2 定案后的控制面工作。

这不是“SCTSR 方法有效”的结论。本轮只证明：**在当前规格可实现的部分，代码机制已经通过训练前审查；当前科学规格仍阻止正式八臂训练。**

机器判定见 [REVIEW_VERDICT.json](REVIEW_VERDICT.json)，逐条问题见 [FINDINGS.json](FINDINGS.json)。

## 2. 固定身份与审查边界

| 角色 | 固定身份 |
|---|---|
| 历史基线 | `a70ba60485dd32c2f8b4268b8f28ea2d3549f42f` |
| 专家实现冻结 | `e9b6df61b0eb02e1d32c29175644f1c2af545afc` |
| 专家交付提交 | `f285754108c7b8e37afd7f5f0fa58fe8fb23d38a` |
| 本轮最终代码冻结 | `9a2f41ec6864764314f57e776a64f7e12ac771a2` |
| 任务书 Git blob | `b201d021712e9c6614e119d35f0e14bdf405c6be` |
| T identity digest | `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B` |
| source-tree digest | `E4F8BECDB4C31296B5B945CB093584EC5BE5348AE151E5BA76728A7699862D65` |

代码审查在：

`C:\Users\28898\AppData\Local\Temp\YOLO-CV-sctsr-v4-readiness-fix-f285754`

的 clean worktree 中完成。原始工作区 `C:\GitHub\YOLO-CV` 没有被这次 review 修改。为了规避 Windows MAX_PATH，命令使用一个短 junction：

`C:\Users\28898\AppData\Local\Temp\sctsr-v4-fix-short`

审查边界为：

- 判断代码是否实现任务书、是否公平、是否 fail-closed；
- 不判断 SCTSR、T 或任一 Q/R/A/D 候选信号是否有效；
- 不跑 200 epoch；
- 不生成正式 seed、assignment、engineering gate、pilot release；
- 不打开 blind holdout/test；
- 不用 `val_op` 选择方法、停止点或 checkpoint；
- 不把 synthetic/engineering canary 当科学结果。

最终 review 报告会作为后续证据提交；源代码 manifest 刻意绑定 `9a2f41e` 这一代码冻结点。后续仅增加 review 文件的证据提交不是训练源码身份，正式 release 应继续绑定代码冻结点或重新生成等价 source manifest，不可把两者混为一谈。

## 3. 这次实验在证明什么

Phase-1 不是在证明“T 是好样本”，而是在固定基础训练、固定累计 replay 暴露和固定 optimizer steps 后，区分三类效应：

1. **一般回流效应**：匹配随机回流是否优于 no replay；
2. **样本身份效应**：T 是否优于严格匹配的随机 R2；
3. **时间与停止效应**：相同身份、相同总剂量、相同 multiplicity 下，均匀 U 与前置 F 是否不同，以及 E160 后继续 T、退回 R2 或停止回流是否不同。

八臂固定为：

| arm | E1-E120 | E121-E160 | E161-E200 | 总 replay occurrence |
|---|---:|---:|---:|---:|
| `NR` | common parent | 0 | 0 | 0 |
| `R1_U` | common parent | 600/epoch | 600/epoch | 48,000 |
| `R2_U` | common parent | 600/epoch | 600/epoch | 48,000 |
| `T_U` | common parent | 600/epoch | 600/epoch | 48,000 |
| `R2_F` | common parent | 1,200/epoch | 0 | 48,000 |
| `T_F` | common parent | 1,200/epoch | 0 | 48,000 |
| `T_TO_R2_AT_160` | common parent | T 600/epoch | R2 600/epoch | 48,000 |
| `T_TO_NR_AT_160` | common parent | T 600/epoch | 0 | 24,000 |

U/F 对同一 3,000-ID pool 都给每个身份 16 次累计额外曝光；差异只在时间分布。`T_TO_R2_AT_160` 中 T 与 R2 各出现 24,000 次、各自每个 ID 8 次。`T_TO_NR_AT_160` 的设计目的不是 dose-match，而是回答停止全部后期回流的效应。

主要对比必须按合同解释：

- `T_U - R2_U`：均匀时间表下的身份效应；
- `T_F - R2_F`：前置时间表下的身份效应；
- `T_TO_R2_AT_160 - T_U`：停止定向 T、退回匹配随机的效应；
- `R2_U - NR`：一般匹配回流效应；
- `T_TO_NR_AT_160 - T_TO_R2_AT_160`：后期停止所有回流与保留随机回流的差异。

任何一项都不能因为平均值好看就宣称有效；必须进入预注册的 unseen-seed、worst-seed、安全非劣和 Holm 判定。

## 4. 资产与“看错文件”防线

### 4.1 代码与历史身份

[SOURCE_TREE_MANIFEST.json](SOURCE_TREE_MANIFEST.json) 记录：

- 347 个注册源文件；
- 19 个 include paths；
- Git HEAD `9a2f41e...`；
- tracked worktree clean；
- source-tree digest `E4F8...2D65`；
- runtime-environment digest `79ADE9...DF63`。

[REPOSITORY_STATE_AUDIT.json](REPOSITORY_STATE_AUDIT.json) 对 baseline 到 reviewed source 的 1,989 个变更路径做允许范围审计，并对 35 个历史证据文件重算 baseline/source Git blob OID、blob SHA 和 normalized worktree OID。

之前的审计把 mtime 当作内容身份，clean checkout 会因新 mtime 被误判。现在 mtime 只记录为 `INFORMATIONAL_NOT_CONTENT_IDENTITY`；真正决定“历史是否被改”的是 Git 内容身份。保护目录的显式 diff stdout 为 0 bytes。

### 4.2 数据清单与物理图片

仅验证 CSV 文件名、sample ID 和 label 不够：同一路径的图片可以被替换。现在正式资产绑定：

- 8 个非 test manifest；
- 384,000 个 image identities；
- 每张图片的 canonical relative path、label、bytes、SHA-256；
- 19,350,859-byte Zstd Parquet content ledger；
- ledger SHA `B2B615...96D2C`；
- dataset content digest `EDA939...DD6E`。

全量物理重哈希实际覆盖：

| split role | rows |
|---|---:|
| canonical base train | 120,000 |
| val_model | 24,000 |
| val_cal | 120,000 |
| val_op | 120,000 |
| 合计 | 384,000 |

总图片字节数为 `82,637,967,451`。验证未接触 blind holdout/test。正式 runner 在 dataset iteration 前验证 ledger，错误路径、缺失文件、同名异字节、额外身份、错误 manifest SHA 都失败。

### 4.3 权重和训练锁

工程 canary 使用的真实 `yolo11l-cls.pt`：

- bytes：28,553,700；
- SHA-256：`6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`。

正式配置继续固定 `yolo11l`、200 epochs、batch 128、现有 canonical training lock，不修改 archived learner。任何自动找“最新权重”、相似文件、`best.pt` 或路径逃逸都不是合法恢复。

## 5. T、R1、R2 与公平性审查

### 5.1 T 的角色

T 是历史上表现出跨 seed 符号反转的 3,000 个正常训练样本，约为 canonical base 的 2.5%。它的角色是**压力测试集合**，不是已验证 selector。

冻结事实：

- unique IDs：3,000；
- label：全部 0；
- historical dynamic bucket：全部 `learnable_hard`；
- OOF folds：10 个；
- `oof_group_id`：959 个 filename-bucket surrogate；
- identity digest：`85D462...3B4B`。

### 5.2 R1

R1 是 global random，对全局 canonical base 进行随机抽样。它可以与 T 重叠；重叠是随机基线的自然结果，必须记录而不能人为去掉。

### 5.3 R2 的终端泄漏防线

R2 只允许读取预终端字段：

- label；
- historical dynamic bucket；
- OOF fold；
- `oof_group_id` surrogate；
- sample identity（只用于排除 T 和唯一性）。

它禁止读取 loss、confidence、RHO、gradient、AUM、GapCritical、correct rate、future endpoint 或任何由终端训练结果派生的字段。`TerminalFieldGuard` 与 matcher 都有负向测试。

### 5.4 为什么 R2 构造不出来

T 的 3,000 个 occurrence 落在 959 个四字段联合 strata。排除 T 后：

- 172 个 strata 候选数量不足；
- 一共缺 378 个 occurrence；
- 30 个 strata 候选为 0；
- 这些零候选 strata 涉及 61 个 occurrence。

replacement 不能解决 0-candidate cell；允许 T overlap 会污染 treatment/control；把 T 缩到 2,622 会改变 treatment pool 和 2.5% rate；向 canonical base 增加新样本会改变固定基础过程。详情见 [R2_SPECIFICATION_AUDIT.json](R2_SPECIFICATION_AUDIT.json)。

审计后推荐但**尚未批准**的候选是：

`ZERO_OVERLAP_UNIQUE_EXACT_LABEL_DYNAMIC_FOLD_MINIMUM_OOF_GROUP_DISPLACEMENT_RANDOM_FILL`

它保留：

- 3,000 unique；
- 与 T 零 overlap；
- label/dynamic/fold 精确；
- 终端字段不可访问；
- deterministic selection seed。

它改变：

- 不再精确匹配 `oof_group_id`；
- 对不可满足的 378 个 occurrence 在同 label/dynamic/fold cell 内做最小 group displacement；
- group total variation 的理论下限与该方案实值均为 `0.126`；
- proposed identity digest 为 `075FC31FE487D3646E89BA1043E5124D9FE49CE9FCC61C1A8041A9CB8196BECC`。

因此它是**科学规格变更**，不是 bug fix。正式训练前必须明确接受以下解释：R2 控制 label、历史难度和 fold，但 12.6% occurrence 无法保持 filename-bucket surrogate 的精确配额；R1 继续作为共同主随机对照，group residual imbalance 必须完整报告。

## 6. 固定 base-step 训练逻辑审查

### 6.1 基础过程

正式 base dataset 始终是 120,000 张 canonical base；batch 128，因此每 epoch `ceil(120000/128)=938` 个 base batches。所有 arm 的：

- base ID 顺序；
- base augmentation seed；
- base batch 边界；
- DataLoader 长度；
- optimizer step 数；
- scheduler epoch transition；
- EMA update 数；
- global step；

都必须相同。Replay 不能拼接进 base Dataset 来增加 batch 或 step。

### 6.2 Replay 注入

每个 base step 的顺序是：

1. base forward；
2. base loss 按冻结 Ultralytics 分类实现 backward；
3. 若本 step 有 replay，在独立 RNG domain 中做 replay forward；
4. replay 逐样本 CE 求和后除以固定 canonical base batch size 128；
5. replay backward 将梯度累加到同一参数；
6. 恢复 replay 前的 Python/NumPy/Torch CPU/CUDA RNG；
7. 恢复 replay forward 改写的 BatchNorm running buffers；
8. 一次 unscale、clip、optimizer/scaler update、zero-grad；
9. 一次 EMA update；
10. global step 增加 1。

Replay microbatch 上限为实际 base batch的 25%。尾 batch也按实际 base batch检查 cap，不能拿128掩盖越界。代码关键点：

- `stage1_sctsr_v4/fixed_step_runtime.py:301-371`：cap 与 `sum/128`；
- `stage1_sctsr_v4/fixed_step_runtime.py:412-443`：单 optimizer 与单 EMA；
- `stage1_sctsr_v4/fixed_step_runtime.py:495-510`：scheduler 只做 epoch transition。

真实 canary 观察到：base=4、replay=1、optimizer step=1、EMA update=1，且 model state 前后发生合法变化、RNG/BN 均恢复。

### 6.3 明确失败条件

以下情况不会“尽量跑完”，而是作废当前 epoch/run：

- world size > 1；
- accumulation != 1；
- replay microbatch 超 cap；
- OOM 后自动减 batch；
- 隐式拆成额外 optimizer step；
- NaN/Inf；
- base/replay occurrence 数与 schedule 不符；
- scheduler/EMA/global-step drift；
- RNG/BN restore digest 不符；
- checkpoint、parent、seed、source、runtime 或 dataset identity 不符。

## 7. Common parent、分支与授权

每个 training seed 先独立训练 E1-E120 no replay parent。E120 checkpoint 必须包含：

- model 与 EMA；
- optimizer；
- scheduler；
- AMP scaler；
- Python/NumPy/Torch CPU/CUDA RNG；
- epoch/global step；
- source/runtime/asset/training seed identity。

同一 seed 的八个 branch 都绑定同一 E120 parent SHA；child 不得修改 parent。逻辑 artifact index 对 E1-E120 指向 parent 物理产物，对 E121-E200 指向 child 产物，不复制后伪装成 child 原生历史。

正式授权现在分两层：

1. 一份签名 matrix release；
2. 每个 exact job 一份签名 one-use execution token。

Token 绑定 run_id、arm、seed、parent/source/runtime/assets、release nonce 和共享 claim registry digest。共享 registry 用 `O_EXCL` 原子写 claim；同 nonce 第二次使用、并发 double-claim、损坏 claim、复制 registry 到另一目录、job substitution 都在 trainer 构造前失败。

本轮没有创建真实 release/token/registry，只验证了控制逻辑。

## 8. Run-intent 与训练机 AI 交付

“AI 看一眼能跑起来”不算安全交付。现在每个正式 parent/branch 在 token claim 前都必须提供 `RUN_INTENT_ACKNOWLEDGEMENT.json`，它包含：

- 23 个 exact job/context 字段；
- 16 个必须为 JSON boolean `true` 的理解声明；
- runbook 17 个文档的 path/bytes/SHA；
- runbook digest；
- source/runtime/asset/contract/seed/arm/parent/output identity；
- 生成时间与最长 7 天 freshness。

它不是授权；签名 token 才是授权。它的作用是证明当前训练机操作者或 AI 阅读并绑定了这次实验的真实含义、禁止项和恢复规则。缺失、过期、false statement、改 arm/seed/path、runbook drift、digest tamper 都失败。

主要文档：

- `READ_ME_FIRST.md`；
- `EXPERIMENT_INTENT.md`；
- `FAIRNESS_CONTRACT.md`；
- `ASSET_IDENTITY_LEDGER.md`；
- `TRAINING_OPERATIONS_MANUAL.md`；
- `MACHINE_RUNBOOK.md`；
- `FAILURE_AND_RECOVERY.md`；
- `ARTIFACT_AND_SCHEMA_GUIDE.md`；
- `DEPLOYMENT_CHECKLIST.md`；
- `INDEPENDENT_REVIEW_CHECKLIST.md`。

冻结的 runbook v2：17 个文档、190,795 bytes、digest `40BE9E9E...EE24`；manifest bytes 2,839、SHA `6FAC3F16...233A`。

## 9. 事务、故障与恢复审查

Epoch generation 遵循 `.inprogress -> validate -> manifest -> rename .complete -> append receipt -> artifact index -> rolling pointer`。

本轮重点修复了两个隐蔽问题：

1. rename 成 `.complete` 后、receipt 前写盘失败，过去可能留下“看起来完成但不在 receipt chain”的 orphan；现在它会被验证并 quarantine；
2. telemetry stop 等 cleanup 自身失败时，过去可能覆盖原始 OOM/训练异常或阻止 transaction abort；现在所有 cleanup 都尝试，原始异常继续抛出，cleanup 异常作为 notes 附着。

故障注入覆盖：

- post-rename/pre-receipt；
- post-receipt/index；
- post-index/pointer；
- corrupt receipt；
- truncated checkpoint；
- partial generation；
- telemetry stop failure；
- quarantine rename failure；
- disk-full/write failure；
- wrong RNG/source/parent/assets；
- Windows >260 字符路径。

恢复只从最后完整、receipt-chain 合法、checkpoint 可重载的 epoch 开始。不能自动搜索 latest，不能跨 run/seed/arm/parent/generation 拿 checkpoint。

## 10. Prediction、评价与统计审查

正式 endpoint 固定：

- checkpoint：E200；
- model variant：EMA；
- split：`val_op`；
- semantic：`ENDPOINT_ONLY_NOT_FOR_SELECTION`。

`best.pt` 明确拒绝。E120/140/150/160/180 只允许 `val_model` trajectory，不能选方法、停止点或 checkpoint。test/blind role 全部拒绝。

Prediction artifact 绑定：

- run/arm/seed/generation；
- checkpoint path/bytes/SHA/E200；
- source/runtime；
- split manifest identity；
- sample ID、label、logits、raw probability；
- sample-label digest；
- model state 与 checkpoint MODEL/EMA state digest。

安全前沿在 tie group 边界上计算，不拆相同 raw probability。FN budget 从 0 到 95，必须恰好 96 行。主要指标是 raw frontier normalized AUC；同时保存：

- `TN_at_FN95` 与其 threshold；
- `FN_at_TN68253` 与另一独立 threshold；
- target 不可达语义；
- tie size/group count；
- seed win rate；
- worst seed；
- 双端同时恶化；
- exact paired sign-flip；
- frozen comparison family 的 Holm step-down。

两个锚点阈值不能拼成同一个 confusion matrix，也不能假设相等。

## 11. 动态验证结果

所有命令、exit code 与日志 SHA 见 [COMMAND_INDEX.json](COMMAND_INDEX.json)。最终代码冻结点结果：

| 检查 | 结果 |
|---|---|
| Python 3.11 v4 | `395 passed in 80.84s` |
| Python 3.12 v4 | `395 passed in 83.80s` |
| v4 core skips | 0 |
| v3 regression | `181 passed, 3 skipped in 5.16s` |
| compileall 3.11 | PASS |
| compileall 3.12 | PASS |
| `uv lock --check` | PASS |
| `git diff --check` | PASS |
| source manifest | PASS |
| repository state audit | PASS |
| synthetic canary x2 | PASS |
| real-image/real-weight engineering canary | PASS |

v3 的三个 skip 均有明确外部证据原因：一个需要已注册 Desktop mirror，两个需要 21 个 local literature-anchor source files。旧 review 计划里的 `183 passed, 1 skipped` 与 clean worktree 当前事实不同，作为 P3 记录，不用错误数字冒充。

## 12. 真实工程 canary

Canary 使用真实 train 角色图片：

| sample | label | bytes | SHA-256 |
|---|---:|---:|---|
| `00841085.png` | 1 | 370,015 | `3BB900...D6D5` |
| `00832687.png` | 1 | 381,574 | `3B640B...42C4` |
| `00145723.png` | 0 | 488,931 | `237C9F...9CAA` |
| `00145724.png` | 0 | 467,575 | `47D5DC...2549` |

运行环境为 Python 3.11.14、torch 2.11.0+cu128、RTX 4060。结果：

- forward PASS；
- base backward PASS；
- replay backward PASS；
- RNG restore PASS；
- BN restore PASS；
- optimizer=1；EMA=1；
- occurrence Parquet：5 rows、ZSTD、SHA `718FD0...A772`；
- prediction Parquet：4 rows；
- frontier Parquet：96 rows；
- checkpoint：103,328,575 bytes、SHA `00592B...75B9`，reload PASS；
- corrupt checkpoint rejected；
- partial generation quarantined；
- formal side effects 全部 false。

Canary 的 evaluation mode/split 使用 synthetic 语义，因为 4 张图片不可能代表正式 val_op；其 receipt 标记 `ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT`，不能进入科学结果。

## 13. 确定性验证

在同一稳定 runtime、同一 source digest、同一 seed 下重复 synthetic canary：

- parent checkpoint SHA 相同；
- 11/11 checkpoint 文件 byte-equal；
- 1/1 asset 文件 byte-equal；
- 3/3 selection ledgers byte-equal；
- 8/8 predictions byte-equal；
- 24/24 evaluations byte-equal。

根 receipt/artifact index 因绝对 output root、PID 和 timing 不同而字节不同，这是保留真实 provenance 的预期行为；不能为了“哈希一样”抹掉这些字段。详见 [DETERMINISM_COMPARISON.json](DETERMINISM_COMPARISON.json)。

## 14. 10 台 RTX 3090 工期估算

历史同仓库 `yolo11l` 训练日志显示，单 RTX 3090 完成 200 epochs 用 12.998 小时，E200 为 938 steps、约 3:25。基于此：

- 每 seed：120-epoch parent + 8×80-epoch branches = 760 base-epoch equivalents；
- 每 seed 约 49.3924 GPU-hours；
- 每 seed replay occurrences 共 312,000，相当于 2.6 个 base epoch 的样本 occurrence，但 replay forward/backward 与证据写盘仍有额外开销。

10 台 3090 的规划区间：

| 阶段 | base GPU-hours | 依赖约束理想下限 | 建议排期 |
|---|---:|---:|---:|
| 8-seed discovery | 395.14 | 44.19 h | 53–65 h，约 2.2–2.7 天 |
| 14-seed confirmation | 691.49 | 70.19 h | 84–101 h，约 3.5–4.2 天 |
| 全部 22 seeds | 1,086.63 | 109.18 h | 137–166 h，约 5.7–6.9 天 |

这是容量规划，不是 SLA。R2 定案后应先在真实 3090 上做一个完整 epoch 的 engineering benchmark，再冻结 queue duration。详情见 [RTX3090_CAPACITY_ESTIMATE.json](RTX3090_CAPACITY_ESTIMATE.json)。

## 15. Findings 汇总

| ID | 严重度 | 状态 | 结论 |
|---|---|---|---|
| RV4-001 | P1 | RESOLVED | clean checkout OOF bytes |
| RV4-002 | P1 | RESOLVED | post-rename orphan generation |
| RV4-003 | P1 | RESOLVED | one-use formal execution + shared claim root |
| RV4-004 | P2 | OPEN | R2 exact quota 科学规格不可行；正式训练硬阻断 |
| RV4-005 | P3 | OPEN | v3 clean-worktree pass/skip 数与旧口径不同 |
| RV4-006 | P1 | RESOLVED | 384k physical image byte identity |
| RV4-007 | P1 | RESOLVED | cleanup 不再覆盖主异常 |
| RV4-008 | P1 | RESOLVED | run-intent acknowledgement |
| RV4-009 | P2 | RESOLVED | canary vectorized events/occurrence rows |
| RV4-010 | P2 | RESOLVED | canary 自身评价产物 fail-closed |
| RV4-011 | P1 | RESOLVED | Windows junction extended path |
| RV4-012 | P2 | RESOLVED | 精确允许 v4 prereg scope |
| RV4-013 | P2 | RESOLVED | protected history 内容身份不依赖 mtime |

全部复现命令、expected/observed、修复 commit 和回归测试在 [FINDINGS.json](FINDINGS.json)。

## 16. 正式部署前唯一正确的下一步

当前不能直接把 10 台机器转起来。最短且安全的顺序是：

1. 研究负责人阅读 R2 audit；
2. 明确接受或拒绝 `minimum_displacement_zero_overlap` addendum；
3. 若接受，新增独立 TDD 回滚单元：
   - 冻结新 R2 semantic；
   - 输出 exact 3,000-ID pool、digest、stratum displacement ledger；
   - 保持 zero overlap、label/dynamic/fold exact；
   - 更新 fairness contract、contrast interpretation、runbook manifest；
   - 重新跑 395×2、真实 canary、source/repository audit；
4. 先做 1 台 RTX3090 的一个完整工程 epoch benchmark，不生成科学 seed；
5. 由负责人签发 8 个 unseen discovery seeds、matrix release、64 个 branch job tokens 与 8 个 parent tokens；
6. 10 台机器连接同一 canonical shared claim registry；
7. 每台机器先生成并验证 exact `RUN_INTENT_ACKNOWLEDGEMENT`；
8. 先启动 E1-E120 parents；parent closeout 完成后才启动对应八臂 branches；
9. discovery 结束按预注册规则判定，不能看 test，也不能临时改停止点；
10. 只有 discovery advance 后才签发 14-seed confirmation。

如果负责人拒绝任何 R2 规格变化，则该八臂实验应保持 BLOCKED，而不是删掉 R2 或把普通随机冒充 R2。

## 17. 自我审计与测试需求清单

以下是交付/训练负责人必须逐项确认的最终 checklist；机器版见 [SELF_AUDIT_CHECKLIST.json](SELF_AUDIT_CHECKLIST.json)。

### 17.1 身份与范围

- [x] baseline、taskbook、delivery、reviewed source commit 已固定；
- [x] source tree 347 个注册文件有 manifest；
- [x] 原 dirty checkout 未被 review 修改；
- [x] 历史训练代码与产物没有删除或重写；
- [x] 保护目录 content diff 为 0；
- [x] checkout mtime 不冒充内容身份；
- [ ] 正式 release 还未签发；
- [ ] 正式 seeds 还未冻结；
- [ ] shared claim registry 还未部署。

### 17.2 数据与选择公平性

- [x] 384,000 个非 test 图片有 path/bytes/SHA；
- [x] T 为冻结 3,000 IDs，且只作压力测试；
- [x] R1 是 global random；
- [x] R2 禁止终端字段并要求 zero overlap；
- [x] exact R2 不可行时 fail closed；
- [ ] R2 addendum 尚未由研究负责人批准；
- [ ] 新 R2 identity pool 尚未正式生成；
- [x] U/F 48,000 occurrence 和 per-ID multiplicity 相等；
- [x] stop/fallback 的 E160 边界明确；
- [x] CURRENT_LOSS 与 Q/R/A/D phase2 保持 HELD；
- [x] 无 val_target 时 A 硬阻断。

### 17.3 训练逻辑

- [x] canonical base=120,000；batch=128；938 base steps/epoch；
- [x] replay 不改变 DataLoader 长度；
- [x] replay CE=`sum(per_sample_ce)/128`；
- [x] replay microbatch≤实际 base batch×25%；
- [x] 每 base batch 一次 optimizer/scaler update；
- [x] 每 base batch一次 EMA update；
- [x] scheduler/global step 只跟 base；
- [x] replay RNG 与 BN buffers 恢复；
- [x] replay 参数梯度保留；
- [x] world_size>1、accumulation、auto-batch、OOM continuation 禁止；
- [x] common parent/branch lineage/checkpoint identity 有测试。

### 17.4 产物与恢复

- [x] 正式大表只允许 Zstd Parquet；
- [x] occurrence/step/exposure/selection/telemetry schemas 有验证；
- [x] generation、receipt、index、pointer 原子顺序有故障测试；
- [x] post-rename orphan 会 quarantine/reconcile；
- [x] cleanup error 不覆盖主异常；
- [x] resume 只从最后 canonical complete epoch；
- [x] 长路径、半写、损坏 checkpoint/receipt 测试覆盖；
- [x] 同 token/nonce 重复 job 会在 trainer 前拒绝；
- [x] copied claim registry 不能绕过 single-use。

### 17.5 评价与统计

- [x] 正式 endpoint 为 E200/EMA/val_op；
- [x] `best.pt` 禁止；
- [x] val_op 不做方法/checkpoint/stop selection；
- [x] test/blind holdout 访问禁止；
- [x] FN=0..95 恰好 96 个 tie-safe points；
- [x] normalized AUC、TN_at_FN95、FN_at_TN68253 保存；
- [x] 两个 threshold 独立；
- [x] paired seeds、sign-flip、Holm、worst seed、双端恶化有实现测试；
- [x] canary 没有被登记为科学结果。

### 17.6 动态测试与最终判定

- [x] Python 3.11：395 passed；
- [x] Python 3.12：395 passed；
- [x] v4 core skips=0；
- [x] v3：181 passed、3 skip 原因完整；
- [x] compileall 3.11/3.12 PASS；
- [x] uv lock PASS；
- [x] git diff check PASS；
- [x] synthetic same-runtime determinism PASS；
- [x] 真实图片+真实权重工程 canary PASS；
- [x] open P0=0、open P1=0；
- [x] 代码判定为 `CODE_REVIEW_PASS_NOT_TRAINING_AUTHORIZATION`；
- [x] 正式训练判定仍为 `BLOCKED`；
- [x] `formal_training_started=false`；
- [x] `assignments_generated=false`；
- [x] `engineering_gate_generated=false`；
- [x] `pilot_release_generated=false`；
- [x] `blind_holdout_opened=false`；
- [x] `test_accessed=false`；
- [x] `method_effectiveness_claimed=false`。

## 18. 证据入口

- [REVIEW_VERDICT.json](REVIEW_VERDICT.json)：双层最终判定；
- [FINDINGS.json](FINDINGS.json)：逐条问题、复现、修复与剩余风险；
- [COMMAND_INDEX.json](COMMAND_INDEX.json)：命令、exit code、日志 bytes/SHA；
- [R2_SPECIFICATION_AUDIT.json](R2_SPECIFICATION_AUDIT.json)：R2 不可行与候选规格；
- [SOURCE_TREE_MANIFEST.json](SOURCE_TREE_MANIFEST.json)：代码身份；
- [REPOSITORY_STATE_AUDIT.json](REPOSITORY_STATE_AUDIT.json)：变更边界、历史保护与副作用；
- [CHANGED_FILE_LEDGER.json](CHANGED_FILE_LEDGER.json)：逐文件变更 ledger；
- [DETERMINISM_COMPARISON.json](DETERMINISM_COMPARISON.json)：同 runtime 重复结果；
- [RTX3090_CAPACITY_ESTIMATE.json](RTX3090_CAPACITY_ESTIMATE.json)：10 台 3090 容量规划；
- [SELF_AUDIT_CHECKLIST.json](SELF_AUDIT_CHECKLIST.json)：机器可读自审清单；
- [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json)：本目录文件 bytes/SHA。

最终大白话：**代码已经查到可以放心继续做“正式放行准备”，但还不能按下 10 台机器的正式训练按钮。现在卡住的不是训练循环，而是 R2 这个公平对照在现有数据上无法按旧定义凑齐。先签科学 addendum，再做 3090 单 epoch 工程基准，再发正式 token。**
