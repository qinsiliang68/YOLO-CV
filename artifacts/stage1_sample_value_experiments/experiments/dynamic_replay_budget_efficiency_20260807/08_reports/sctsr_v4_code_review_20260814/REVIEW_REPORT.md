# SCTSR v4 训练前代码 Review 报告

审查日期：2026-08-14

审查角色：`CODEX_PRIMARY_AGENT_SELF_REVIEW_NOT_INDEPENDENT`

审查性质：怀疑式、只读实现审查；不是方法有效性判断；不是正式训练授权。

## 1. 结论先行

本次不能签发 `CODE_REVIEW_PASS_NOT_TRAINING_AUTHORIZATION`。

机器判定为：

```text
CODE_REVIEW_FAIL_P1_OPEN_NOT_TRAINING_AUTHORIZATION
FORMAL_TRAINING_READINESS=BLOCKED
```

原因不是 SCTSR v4 只有空壳，也不是训练链路完全不能工作。相反，真实 Sewer-ML 图片、真实 `yolo11l` 权重、真实 Ultralytics `ClassificationTrainer` 的一步工程 canary 已完成 base/replay forward、backward、一次 optimizer/EMA update、Zstd Parquet、checkpoint、故障 quarantine、resume 以及 96 点 frontier。

不能通过的直接原因是 3 个开放 P1：

1. clean checkout 中 `oof_metadata` 被 Git 规范化为 LF，字节数和 SHA 与 asset registry 冻结的 CRLF 表示不一致，导致 Python 3.11 和 3.12 都只有 `337 passed, 3 errors`；
2. epoch transaction 在目录 rename 成 `.complete` 后、receipt/index/pointer 发布前发生异常时，会留下未入 receipt chain、也未 quarantine 的孤立 complete generation；
3. 正式签名 release 的 nonce 只校验格式，没有消费或幂等执行账本，同一 release 可重复通过验证。

另有两个较低等级问题：任务书关于 R2 “已核实可行”的事实陈述被当前冻结资产反驳；v3 clean-worktree 回归口径为 `181 passed, 3 skipped`，不是计划中的 `183 passed, 1 skipped`。

完整 finding 以 [FINDINGS.json](FINDINGS.json) 为唯一机器可读清单，双层判定以 [REVIEW_VERDICT.json](REVIEW_VERDICT.json) 为准。

## 2. 冻结身份和审查边界

| 身份 | 冻结值 |
| --- | --- |
| 历史基线 | `a70ba60485dd32c2f8b4268b8f28ea2d3549f42f` |
| 实现冻结提交 | `e9b6df61b0eb02e1d32c29175644f1c2af545afc` |
| 当前交付提交 | `f285754108c7b8e37afd7f5f0fa58fe8fb23d38a` |
| 任务书 blob | `b201d021712e9c6614e119d35f0e14bdf405c6be` |
| review 分支 | `codex/sctsr-v4-review` |
| clean review worktree | `C:\Users\28898\AppData\Local\Temp\YOLO-CV-sctsr-v4-review-f285754` |

从 `f285754` 建立独立 worktree 后开始审查。实现源码没有在 review 过程中修改；全部新增文件仅位于本报告目录。当前仓库原工作树中的未跟踪历史、文献和本地资产没有纳入 review 分支。

重新构建的 source-tree manifest 结果：

| 字段 | 结果 |
| --- | --- |
| registered source files | 255 |
| include paths | 19 |
| Git HEAD | `f285754108c7b8e37afd7f5f0fa58fe8fb23d38a` |
| Git dirty（构建 manifest 时） | `false` |
| source tree digest | `A987B90A869FEBEB2B0797FDE06638C9C1B4541CB25400B3FFC567970779ECA5` |
| runtime environment digest | `82D378027B168DB18E1900A445134EAF0CB8274F8A8CB4D2A27B0DDAB63D7280` |
| manifest SHA-256 | `C2D7D3649DB13499C9FD3A97099C00F102123C9528F6A8A74E16CBE25E4871C8` |

证据见 [SOURCE_TREE_MANIFEST_RECOMPUTED_RECEIPT.json](snapshots/SOURCE_TREE_MANIFEST_RECOMPUTED_RECEIPT.json)。

保护边界核对：

- `stage1_gapvalue240`：基线到交付提交零 diff；
- `stage1_dynamic_replay_v3`：基线到交付提交零 diff；
- `YOLOv11/ultralytics`：基线到交付提交零 diff；
- `e9b6df61..f285754` 在 `stage1_sctsr_v4`、scripts、configs、integration 和 tests 内零源码漂移；
- 历史训练产物、旧 queue、旧 release 和旧 assignment 未被 review 修改。

## 3. Findings

| ID | 等级 | 状态 | 结论 | 训练影响 |
| --- | --- | --- | --- | --- |
| `SCTSR-RV4-001` | P1 | OPEN | clean checkout 的 OOF metadata 字节与 registry 不一致 | 阻断 |
| `SCTSR-RV4-002` | P1 | OPEN | post-rename 发布失败留下孤立 complete generation | 阻断 |
| `SCTSR-RV4-003` | P1 | OPEN | 正式 release nonce 可重复验证 | 阻断 |
| `SCTSR-RV4-004` | P2 | OPEN | R2 可行性事实与冻结资产矛盾；matcher 正确 fail-closed | 数据/规格阻断 |
| `SCTSR-RV4-005` | P3 | OPEN | v3 clean profile 为 181 passed、3 skipped | 不单独否决 v4 |

### 3.1 P1：clean checkout 资产身份不可复现

`.gitattributes:4` 对所有 JSON 规定 `eol=lf`；`configs/stage1_sctsr_v4/asset_registry_v1.json:179-182` 却登记了旧 Windows worktree 中 CRLF 版本的 `metadata.json`：1,076 bytes、SHA `759B...F97C`。

clean worktree 实际文件是 1,049 bytes、SHA `B4AE...D4F0`。`stage1_sctsr_v4/asset_registry.py:208-213` 正确执行字节数和 SHA fail-closed，因此三个真实 formal-pool 输入测试在 trainer 构造前报错。

复现：

```powershell
.\.venv\Scripts\python.exe `
  artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\08_reports\sctsr_v4_code_review_20260814\repro\repro_clean_checkout_asset_identity.py `
  --comparison-root C:\GitHub\YOLO-CV
```

此问题不能通过“在当前机器把旧 CRLF 文件复制回来”修复，因为正式交付必须从 clean Git identity 可重建。需要统一 Git 属性、tracked bytes、registry 和任务书 addendum，并增加 fresh-worktree 测试。

### 3.2 P1：epoch 原子发布存在 rename 后故障窗口

`stage1_sctsr_v4/epoch_transaction.py:248-265` 的顺序是：

1. `inprogress -> complete`；
2. `_committed=True`；
3. append receipt；
4. 更新 artifact index；
5. 更新 rolling recovery pointer。

但异常处理 `:271-278` 只在 `.inprogress` 仍存在时调用 quarantine。注入 rename 后 receipt append 的 `OSError` 后，实测：

```json
{
  "complete_exists": true,
  "inprogress_exists": false,
  "quarantined_generation_count": 0,
  "receipt_exists": false,
  "artifact_index_exists": false,
  "recovery_pointer_exists": false
}
```

代码会拒绝把它当成有效 resume 前缀，这是 fail-closed 的一部分；但它没有按任务书把失败 generation 移到 quarantine，也没有提供自动 reconcile，run 会被孤立 `.complete` 卡住。修复必须覆盖 receipt、index、pointer 三个独立失败边界，不能只测 rename 前的半写 JSON/Parquet。

### 3.3 P1：正式 release 缺少可执行粒度的 anti-replay

`stage1_sctsr_v4/formal_release.py:130-136` 只验证 nonce 是长度至少 32 的字符串；`:175-186` 验证 HMAC 后直接返回 manifest。没有已消费 nonce、release/job execution token 或并发幂等账本。

签名、key window、expiry、contract/asset/runtime/source/seed bindings 本身已经实现，`run_branch.py:88-99` 也确实在 trainer 构造前调用 authorization；缺的是重复执行语义。同一合法 release 在同一进程连续验证两次，第二次仍成功。

任务书只写“未来签名 release”，没有明确一份 release 是矩阵级还是 job 级；批准的 review 计划明确要求 duplicate nonce 被拒绝。因此修复前必须先冻结授权粒度，再实现原子消费/幂等执行，不应简单加一个进程内 `set()`。

### 3.4 P2：R2 数据/规格前提被实物反驳

`random_controls.py:107-109` 在 matcher 前调用 `TerminalFieldGuard.project_rows`；`:110-124` 排除 T、逐 stratum 统计并在任一缺口时返回 `R2_QUOTA_INFEASIBLE`。这是正确实现。

用当前登记的 CRLF 资产字节重建得到：

| 字段 | 实测 |
| --- | ---: |
| canonical base | 120,000 |
| T identities | 3,000 |
| T identity digest | `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B` |
| shortfall strata | 172 |
| missing occurrences | 378 |
| matching relaxation | 禁止 |

因此任务书 `:190-192` 的“已核实可行”与实物不一致。不能为了开跑放宽 quota；必须新冻结资产或签署改变 estimand 的 preregistration addendum。

### 3.5 P3：v3 回归口径漂移

clean worktree 串行运行结果为 `181 passed, 3 skipped`。三个 skip 都有明确原因：一个依赖 Desktop mirror，两个依赖 21 个本地 anchor source files。suite exit 0，v3 源码零 diff。

依据用户批准的 review 规则，此差异登记但不单独否决 v4。若要把 v3 count 作为正式 gate，需要增加外部证据 profile，而不是只写一个固定数量。

## 4. A：身份、边界与正式授权审查

### 已确认存在

- source tree manifest 覆盖 255 个登记源文件，重算 clean；
- 保护目录零修改；
- formal release 检查 schema、authorization flag、HMAC-SHA256、trusted key、key validity window、issued/expiry 和七项完整 identity bindings；
- 空 release、错误签名、错误绑定、过期 release、错误 key 等负路径由测试覆盖；
- authorization 调用顺序位于 trainer 构造之前；
- validator/synthetic canary 不生成 formal seed、assignment、engineering gate 或 pilot release；
- test、blind holdout 和正式 `val_op` 未访问。

### 未通过

- clean checkout 资产绑定见 `SCTSR-RV4-001`；
- duplicate nonce 见 `SCTSR-RV4-003`。

## 5. B：T、R1、R2 与八臂 schedule 审查

### 已确认存在

- `ArmId` 固定顺序为 `NR`、`R1_U`、`R2_U`、`T_U`、`R2_F`、`T_F`、`T_TO_R2_AT_160`、`T_TO_NR_AT_160`；
- 对照 C01-C08 在 `arm_spec.py:79-86` 明确登记；
- T 从 canonical base 的 2.5% 构建，实测 3,000 identities 和冻结 digest 一致；
- R2 在 matcher 前做字段白名单投影，排除 T identity，匹配 `label + dynamic_bucket + oof_fold + oof_group_id`；
- 任一 quota 不足时返回 `R2_QUOTA_INFEASIBLE`，没有 fallback、近似或放宽；
- `CURRENT_LOSS_U` 保持 HELD；
- Q/R/A/D weighted total score 被合同拒绝；
- `val_target` 缺失时 A/gradient-alignment 固定返回 `BLOCKED_BY_VAL_TARGET`；
- U/F、E160 stop/fallback、E1-E120 zero replay 和累计 dose/multiplicity 约束已有测试覆盖。

### 未通过或阻断

- clean checkout 无法完成 formal-pool 三项集成测试；
- 当前资产无法生成正式 R2，见 `SCTSR-RV4-004`。

## 6. C：固定 base-step 训练链路审查

这一部分没有发现新的 P0/P1 finding。

逐行追踪 CLI、`formal_cli`、`integrations/ultralytics/sctsr_classification_trainer.py` 和 `stage1_sctsr_v4/ultralytics_overlay.py` 后确认：

- canonical base denominator 为 120,000，batch 128，对应每 epoch 938 base loader steps；
- replay 不拼接到 base Dataset，也不增长 base DataLoader；
- 每个 step 先 base backward，再在独立 RNG domain 内执行 replay forward/backward；
- replay loss 为逐样本 CE sum 除以 canonical base batch size 128；
- replay microbatch cap 按实际 base batch 的 25% 验证；
- replay 后恢复 Python/NumPy/Torch CPU/CUDA RNG 和 BatchNorm running buffers，并比较 digest；
- base/replay backward 后只调用一次上游 `trainer.optimizer_step()`；
- 上游一次调用完成 unscale、clip、scaler step/update、zero-grad 和 EMA update；
- scheduler 每 base epoch 只前进一步；global step 和 EMA 只随 base step；
- world size > 1、隐式 accumulation、自动减 batch、拆 step 和 OOM 后继续都被拒绝；
- checkpoint schema 包含 model、EMA、optimizer、scheduler、scaler、RNG、epoch、global step 和输入身份。

动态证据：关键训练/授权/评价/事务 74 项定向回归全绿；真实数据一步 canary 的 optimizer update count 为 1，RNG/BN restore 和 EMA update 均通过。

这里仍有范围限制：一步 canary 不证明完整 938-step epoch、200 epochs 或 10 台 RTX 3090 的性能和稳定性。

## 7. D：产物、事务与恢复审查

### 已确认存在

- occurrence、optimizer-step、exposure、selection、telemetry、prediction 和 frontier schema 覆盖任务书要求的身份、曝光、loss、梯度、RNG/BN、资源、checkpoint 与 split 字段；
- formal 大表路径只接受 Zstd Parquet；portable fallback 仅允许 synthetic 标记；
- 真实 canary 的 occurrence、step、exposure、telemetry、prediction 和 frontier 均为真实 Zstd Parquet，且有 schema/row count/SHA；
- rename 前的 kill、OOM、disk-full、半写 JSON、半写 Parquet 和损坏 receipt 注入会 fail-closed/quarantine；
- checkpoint reload、receipt chain 和 rolling pointer 正常路径通过。

### 未通过

- rename 后的 receipt/index/pointer 故障窗口见 `SCTSR-RV4-002`。

## 8. E：Prediction、评价、统计与 closeout 审查

这一部分没有发现新的 P0/P1 finding。

确认：

- formal endpoint 固定 E200、EMA、`val_op`；`best.pt` 禁止；
- prediction 绑定 checkpoint SHA、split manifest、sample/label、logits、raw probability、source tree 和完整 sample-label digest；
- tie-safe frontier 只在完整 tie group 边界移动阈值；
- FN budget 精确为 0..95 共 96 点；
- normalized AUC 使用 96 点 normalized-TN 曲线的 trapezoid 规则；
- `TN_at_FN95` 与 `FN_at_TN68253` 独立选择阈值，并独立记录 unreachable；
- paired sign-flip、Holm、seed win rate、worst seed、双端恶化和 unreachable seed 均有明确实现；
- formal `validate_run`/`closeout_run` 对缺失 endpoint、替换产物、错误身份和 test/blind role fail-closed。

关键 golden fixture 和真实 canary 都产生 96 个 frontier rows；真实 canary 因只有 3 个 normal，`TN=68253` 正确标记不可达，而没有伪造数值。

## 9. 动态验证总表

所有原始 stdout、stderr、exit code、bytes 和 SHA 登记在 [COMMAND_INDEX.json](COMMAND_INDEX.json)。预期失败复现的 nonzero exit 是 finding 证据，不会被记作 PASS。

| 命令/验证 | 结果 | 解释 |
| --- | --- | --- |
| `uv lock --check` | PASS | lock 一致 |
| v4 Python 3.11 clean | FAIL | 初次缺权重；绑定真实权重后仍为 337 passed、3 asset errors |
| v4 Python 3.12 clean | FAIL | 337 passed、3 asset errors |
| compileall Python 3.11 | PASS | v4 source/scripts/integration/tests |
| compileall Python 3.12 | PASS | v4 source/scripts/integration/tests |
| v3 Python 3.11 | PASS WITH SKIPS | 181 passed、3 skipped |
| critical regression subset | PASS | 74 passed |
| synthetic canary C/D | PASS | 同一个持久 Python 3.11 runtime、同 seed |
| synthetic determinism comparison | PASS | 7 个登记比较全为 true |
| real Sewer-ML + yolo11l canary | PASS | 一步工程机制，不是科学结果 |
| protected tree diff | PASS | 3 个保护目录零 diff |
| source drift e9b6..f285 | PASS | v4 实现源码零漂移 |
| source-tree manifest rebuild | PASS | 255 files、clean |
| `git diff --check` | PASS | 实现 diff 和 review worktree |

最初 A/B synthetic canary 使用两个 `uv --isolated` 环境，source digest 因绝对 Python executable 路径不同而不可比较，且旧比较脚本触发 Windows 长路径问题。随后在同一持久 `.venv`、同一 source tree、短外部 artifact root 下重跑 C/D，确定性比较通过。这是环境身份约束，不登记为实现缺陷。

## 10. 真实数据工程 canary

机器可读摘要见 [REAL_DATA_ENGINEERING_CANARY_REVIEW_SNAPSHOT.json](snapshots/REAL_DATA_ENGINEERING_CANARY_REVIEW_SNAPSHOT.json)。原始 receipt 在审查时为：

```text
C:\Users\28898\AppData\Local\Temp\sctsr_review_realdata_f285754_20260814\REAL_DATA_ENGINEERING_CANARY_RECEIPT.json
bytes=7467
sha256=A00AAC65DE0DD17A383753498003234ADC8F26BEC123886B43DF2619864FD481
```

使用内容：

- 5 张真实 Sewer-ML 图片；
- 真实 `yolo11l-cls.pt`；
- 真实 Ultralytics classification trainer；
- CUDA 设备：本机 `NVIDIA GeForce RTX 4060`；
- 一个 base/replay optimizer step；
- 真实 Zstd Parquet；
- 154,740,417-byte checkpoint，SHA `6560...773F`；
- partial epoch kill 注入、quarantine 和 recovery；
- 5 行 prediction、96 行 frontier。

其语义固定为：

```text
REAL_DATA_ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT
```

它不用于判断 SCTSR 有效，不进入科学结果目录，也不能用来估计 10 台 RTX 3090 的正式训练时长。154 MB checkpoint 保留在外部临时 canary root，本 review 分支只提交其 bytes/SHA 和关键产物哈希，避免把工程大文件塞进 Git。

## 11. 双层最终判定

### 11.1 代码 review

要求的 PASS 条件是双 Python 340 全绿、无 P0/P1、保护目录零变化、真实 canary 完成、正式副作用全 false。

当前只有后三项满足，前两项不满足。因此：

```text
CODE_REVIEW_FAIL_P1_OPEN_NOT_TRAINING_AUTHORIZATION
```

### 11.2 正式训练 readiness

即使三个 P1 修复，当前也仍然：

- R2 exact quota：172 strata、378 occurrence 缺口；
- `val_target`：不存在，A 保持 `BLOCKED_BY_VAL_TARGET`；
- signed formal release：不存在；
- formal seed registry：不存在；
- blind holdout/test：继续密封。

因此：

```text
FORMAL_TRAINING_READINESS=BLOCKED
```

R2 和 `val_target` 是数据/规格阻断；release/seeds 是授权阻断；不能用修代码自动越过。

## 12. 建议的修复顺序

Review 分支不边审边修。若决定修复，应从 `f285754` 新建独立修复分支，按以下最小回滚单元执行：

1. `SCTSR-RV4-001`：先加 fresh-worktree red test，再统一 OOF metadata bytes、Git attributes、registry 和 addendum；双 Python 340/340；
2. `SCTSR-RV4-002`：先加 receipt/index/pointer 三个 post-rename red tests，再实现 journal/reconcile/quarantine；
3. `SCTSR-RV4-003`：先签署 release 粒度 addendum，再加并发/重启/崩溃 red tests和原子消费账本；
4. 单独处理 `SCTSR-RV4-004`：不得放宽匹配；由科学负责人选择新资产或新 estimand；
5. 重跑本报告全部命令并由不同 reviewer 复核，才能签发代码 PASS。

## 13. Review 自我审计

- [x] 从冻结 delivery commit 建立独立 clean worktree；
- [x] 未修改 `stage1_sctsr_v4` 实现源码；
- [x] 未修改 v3、GapValue240、Ultralytics 上游和历史训练产物；
- [x] 每个 finding 包含等级、精确文件/行号、条款、复现、expected/observed、影响、修复和回归测试建议；
- [x] 原始命令日志、exit code、bytes 和 SHA 已登记；
- [x] 预期失败复现没有伪装成 PASS；
- [x] synthetic 与 real-data canary 均明确不是科学结果；
- [x] 未启动 200 epochs；
- [x] 未生成 formal seed、assignment、engineering gate、pilot release 或 signed release；
- [x] 未访问 test、blind holdout 或正式 val_op；
- [x] 未声称 SCTSR、T 或任何 Q/R/A/D signal 有效；
- [x] 明确披露本次是 Codex primary agent review，不伪称独立专家审稿；
- [x] review 证据在独立 `codex/sctsr-v4-review` 分支提交，修复必须另开分支。
