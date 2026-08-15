# SCTSR v4 部署前加固与最终工程验证报告

## 1. 结论

本轮对训练机审查员提出的七类部署阻断进行了逐项修复，并在修复后的精确实现提交
`fc7367ab8efbc311b1efe5a5516ad5e0395f8c69` 上完成双 Python 全回归、两次同 seed
synthetic 确定性复跑，以及真实 Sewer-ML 图像、真实 `yolo11l-cls.pt`、真实 CUDA 的
单步工程 canary。

判定分开记录：

- `CODE_AND_ENGINEERING_READINESS=PASS`：当前分支可作为 SCTSR v4 正式部署代码源；
- `METHOD_EFFECTIVENESS=NOT_TESTED`：没有提前声称 SCTSR、T 或任一 Q/R/A/D 信号有效；
- `FORMAL_TRAINING_STARTED=false`：本轮没有签发正式 seed、release、token、assignment、
  engineering gate 或 pilot，也没有访问 blind/test；
- 第一阶段八臂 timing/stop/fallback 可以部署；A/gradient-alignment 仍按合同在没有独立
  `val_target` 时保持 `BLOCKED_BY_VAL_TARGET`，不影响第一阶段；
- 真正启动前，所有者仍需冻结 8 个 discovery seed 和 14 个 confirmation seed、生成
  13 台机器的随机部署计划、签名 release、为每个 logical job 签发单次 token，并提供
  所有机器共享的 execution-claim registry。这些是启动输入，不是剩余代码缺陷。

## 2. 冻结身份

| 角色 | 身份 |
|---|---|
| 历史基线 | `a70ba60485dd32c2f8b4268b8f28ea2d3549f42f` |
| R2 addendum 审查基线 | `5ce6fa7` |
| 部署加固实现 | `fc7367ab8efbc311b1efe5a5516ad5e0395f8c69` |
| 分支 | `codex/sctsr-v4-deployment-hardening` |
| 训练方法 | 每 seed 一个 E1-E120 common parent；同一 E120 全状态 checkpoint 分叉八个 E121-E200 branch |
| 正式 endpoint | E200、EMA、`val_op`；禁止 `best.pt` 和 test oracle |

本轮在隔离 clean worktree 中施工，没有清理或覆盖原工作树，也没有修改
`stage1_gapvalue240`、`stage1_dynamic_replay_v3`、`YOLOv11/ultralytics` 或任何历史训练产物。

## 3. 阻断逐项处置

### F-01 数据 split 的相同图像字节泄漏

原始 384,000 行内容账本包含 383,887 个唯一图像 SHA、113 个重复内容组、76 个跨科学
角色内容组。修复建立 SHA 级 fail-closed disjointness overlay，并冻结 102 个 evaluation
occurrence 排除项。基础训练集合优先保留，不为了制造“干净数据”回写历史 manifest。

代码入口为 `stage1_sctsr_v4/dataset_disjointness.py` 和
`stage1_sctsr_v4/dataset_adapter.py`。T 压力集合同时修复为 3,000 个内容唯一身份；R2 仍为
3,000 unique、与 T 零 identity overlap，三个 R2 arm 共用同一池。

### F-02 实际训练图片未绑定内容账本

materialized Dataset 不再只凭 basename+label 接受图片。正式 loader 会逐身份核对注册路径、
label、字节数和 SHA-256，错文件、同名错字节、遗漏 exclusion overlay 均失败关闭。相关实现
位于 `stage1_sctsr_v4/dataset_adapter.py:317` 之后。

### F-03 E200 endpoint 使用错误数据根

endpoint 不再硬编码仓库内默认 data 根。`publish_formal_endpoint` 显式接收并重新验证当前
trainer 已绑定的 `dataset_root`，路径与内容身份都会进入 endpoint receipt。入口为
`stage1_sctsr_v4/prediction_runtime.py:307`。

### F-04 同一 logical job 可被不同 token 同时 claim

在单次 token nonce claim 之外增加 logical-job invariant digest、append-only fence generation、
heartbeat lease 和共享 control lock。START 已有 fence 时拒绝；RESUME 必须接续旧 fence，且
活动 lease 未过期时拒绝抢占。epoch publish 前再次验证当前 fence。入口为
`stage1_sctsr_v4/formal_execution.py:167`、`:239`、`:581` 和 `:975`。

runner 顺序也已调整为先 claim，再对旧 `.inprogress` 做恢复动作，因此失败的第二个进程
不能先移动第一个活进程的目录。

### F-05 false-COMPLETE 窗口

epoch 训练结束只能写 `PENDING_FINALIZATION`；branch 必须先完成 E200 endpoint 与最终 logical
artifact index，最后由 `publish_formal_completion` 原子发布唯一 canonical completion receipt。
部分 endpoint 和半完成 finalization 会进入 quarantine，并可从终端 epoch 执行仅 finalization
恢复。入口为 `stage1_sctsr_v4/formal_completion.py:90` 和
`stage1_sctsr_v4/formal_training.py:1363` 附近。

### F-06 13 台 RTX 3090 的部署控制

按所有者明确要求采用简单部署，不实现 GPU UUID、驱动证明或 GPU 文件锁。计划代码生成
22 个 parent + 176 个 branch = 198 个 logical jobs，以固定 assignment seed 打乱后按 wave
分配到 12 个 active machine ID，第 13 台只作 buffer；同一 wave 每台最多一个 job。

实现位于 `stage1_sctsr_v4/deployment_plan.py:80`、`:182`、`:225`，操作合同见
`03_preregistration_v4_sctsr/SCTSR_SIMPLE_RANDOM_DEPLOYMENT_ADDENDUM_20260815.md`。共享
logical-job fencing 仍保留，避免同一 job 双跑。不同 job 被人工同时塞到同一物理 GPU 的
风险由“一台一 wave 只启动一个 placement”操作纪律控制。

### F-07 GitHub archive、Windows 与 Linux 的文本身份漂移

正式 CSV 和 R2 证据日志改为 path-specific LF identity；asset registry 与 evidence manifest
绑定 Git blob 的 LF bytes/SHA。`.gitattributes:13-26` 记录该策略。历史 GapValue CRLF 合同和
原始审计 evidence 的 `-text` 规则保持不变。

### F-08 本轮新增发现：isolated Python 临时路径污染 source identity

两次 `uv run --isolated` 初次复跑时，唯一差异是 Python 位于不同的随机 `.tmp*` 目录，
但该绝对路径被纳入 source-tree digest，进一步导致 parent checkpoint SHA 漂移。失败优先
测试复现后，将 Python executable identity 收敛为稳定文件名，继续绑定 implementation、
精确版本、Torch、CUDA、GPU 和 driver。修复位于 `stage1_sctsr_v4/source_identity.py:49`。

修复后同 seed 两次复跑得到相同 source-tree digest、相同 parent checkpoint SHA，以及
11/11 checkpoint、1/1 asset、3/3 selection ledger、8/8 prediction 和 24/24 evaluation
artifact 字节相等。

## 4. 动态验证结果

所有命令均在 clean worktree、实现提交 `fc7367a` 上串行执行，原始输出、exit code、字节数
和 SHA 由本目录 `COMMAND_INDEX.json` 与 `EVIDENCE_MANIFEST.json` 登记。

| 验证 | 结果 |
|---|---|
| `uv lock --check` | PASS，exit 0 |
| Python 3.11 compileall | PASS，exit 0 |
| Python 3.12 compileall | PASS，exit 0 |
| Python 3.11 v4 全套 | `435 passed`，exit 0 |
| Python 3.12 v4 全套 | `435 passed`，exit 0 |
| Python 3.11 v3 regression | `181 passed, 3 skipped`，exit 0 |
| `git diff --check` | PASS，exit 0 |
| synthetic canary A/B | 两次 PASS；训练相关身份和字节确定；均明确非科学结果 |
| 真实工程 canary | PASS；4 张真实图、真实权重、CUDA 单步、真实 Zstd Parquet |

真实工程 canary 使用本地 `C:/GitHub/YOLO-CV/data/final_sewerml_dataset` 的两张 defect
train 图和两张 normal_train 图，使用注册权重 SHA
`6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`。
在本机 RTX 4060 上验证 forward、base/replay backward、仅一次 optimizer step、仅一次 EMA
update、BN 恢复、Zstd Parquet、checkpoint save/reload、损坏 checkpoint 拒绝、半 generation
quarantine 和 FN=0..95 的 96 点前沿。该结果只证明机械链路可运行，不外推 RTX 3090 的科学
收益或完整 200 epoch 性能。

## 5. 正式启动顺序

1. 将本分支精确 HEAD 合并或建立 clean checkout，并重算 source-tree manifest；
2. 所有者冻结 8 个 discovery seed、14 个 confirmation seed，以及 assignment seed；
3. 用 `build_deployment_plan.py` 生成 12 active + 1 buffer 的 198-job 计划；
4. 在新 source/runbook identity 上生成 runbook manifest；runbook 必须包含任务书、R2 addendum、
   数据内容 addendum、logical-job fencing addendum、atomic-completion addendum、简单随机部署
   addendum 和机器手册；
5. release authority 签发与 exact source/assets/runtime/seeds 绑定的 release；
6. 为每个 logical job 签发单次 START token，并准备所有训练机可访问的共享 claim root；
7. 每台机器先执行真实工程 canary，再依 placement 启动 discovery parent；
8. parent canonical complete 后才启动同 seed 的八个 branch；
9. discovery 判定完成且所有者批准后，才签发 confirmation token；
10. 禁止调用旧 `dynamic_campaign_train_worker.py --job-id`。

## 6. 保留边界

- 没有 `val_target`，所以 A/gradient alignment 仍不能进入正式配置；
- `T` 是历史符号反转压力集合，不是已验证 selector；
- `R2` 是严格控制臂，不是替代 treatment；
- 当前 PASS 不等于 SCTSR 优于 R2/R1/current-loss/no-replay；
- test/blind 继续密封；
- 训练有效性只能由未来未见 seed 的真实配对干预给出。

## 7. 自我审计清单

- [x] 所有审查员提出的代码阻断均有代码、失败优先测试和通过证据；
- [x] 修复分为七个小提交并逐笔推送；
- [x] 真实数据、真实权重、真实 CUDA 路径已执行；
- [x] 双 Python 全套测试与 v3 regression 已执行；
- [x] synthetic 两次同 seed 的训练相关产物已做字节比较；
- [x] 原工作树、历史代码和历史训练产物未删除或覆盖；
- [x] 正式训练、assignment、gate、pilot、blind/test 均未启动；
- [x] 未把工程 canary 或 synthetic canary 冒充科学结果；
- [x] 未声称方法有效；
- [x] 剩余事项仅为所有者启动输入和正式授权。
