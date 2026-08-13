# SCTSR v4 实施、自审计与训练容量最终报告

## 1. 最终判定

本轮已在冻结的 YOLO-CV 基线上完成隔离的 SCTSR v4 实现、失败优先测试、真实本地训练数据工程 canary、完整 synthetic canary、逐提交 TDD 历史审计、逐行自审计和 RTX 3090 十机容量估算。

冻结身份如下：

```text
repository: qinsiliang68/YOLO-CV
baseline: a70ba60485dd32c2f8b4268b8f28ea2d3549f42f
branch: codex/sctsr-v4-taskbook
implementation_source_commit: e9b6df61b0eb02e1d32c29175644f1c2af545afc
taskbook_blob_sha: b201d021712e9c6614e119d35f0e14bdf405c6be
source_tree_file_count: 255
source_tree_digest: 06279F1229235897B08A820D424AC1BFF000A2AA21C48859F98B841704F0C3CE
```

准确状态是：

```text
implementation: IMPLEMENTED_AND_ENGINEERING_TESTED
appendix_d_self_audit: SELF_AUDIT_FAIL
formal_training_authorized: false
formal_training_started: false
scientific_effectiveness_known: false
```

Appendix-D 共 206 项，当前为 205 PASS、1 FAIL。唯一失败是 `SA-266`：冻结任务书要求旧 v3 回归至少 `231 passed`，而当前冻结仓库的真实结果是 `183 passed, 1 skipped`。该差异没有被改写、填充或伪装成 PASS。

此外，正式 phase-1 仍有独立硬阻断：当前资产无法在零身份重叠和四维精确 quota 下构造 R2，实际有 172 个不足分层、累计短缺 378 个 occurrence，最大单层短缺 11。实现按合同返回 `R2_QUOTA_INFEASIBLE`，没有放宽匹配。

这些阻断不否定 v4 实现质量，也不证明 SCTSR 有效或无效。没有正式 replay 干预，就没有 utility evidence。

## 2. 实施边界

新增实现严格隔离在：

- `stage1_sctsr_v4/`
- `scripts/stage1_sctsr_v4/`
- `configs/stage1_sctsr_v4/`
- `tests/stage1_sctsr_v4/`
- `integrations/ultralytics/`
- `docs/stage1_sctsr_v4/`

以下历史代码和训练证据未被改写：

- `stage1_gapvalue240/`
- `stage1_dynamic_replay_v3/`
- `YOLOv11/ultralytics/`
- 既有 40/120/240-run 训练产物
- 旧 queue、release、assignment、checkpoint 与审计材料

旧 gate、pilot release 和 assignment 作为 legacy evidence 保留，但不被解释为 SCTSR v4 的 active 状态。

## 3. 已实现的科学与工程合同

### 3.1 百分比预算和八臂

- `ReplayRateSpec` 只接受整数有理数，不接受浮点比例或方法配置中的绝对样本数。
- treatment identity pool 固定为 canonical base 的 `25/1000`。
- U 为 E121–E200 每 epoch `5/1000`。
- F 为 E121–E160 每 epoch `10/1000`，E161–E200 为 0。
- U/F 的 identity digest、累计 occurrence、逐 ID multiplicity 完全相同，只改变时间分布。
- 八臂顺序固定为 `NR`、`R1_U`、`R2_U`、`T_U`、`R2_F`、`T_F`、`T_TO_R2_AT_160`、`T_TO_NR_AT_160`。
- `CURRENT_LOSS_U` 仅有 HELD 接口，不进入第一阶段。

### 3.2 T、R1、R2

- T 绑定 3000 行历史符号反转压力集合，identity digest 为 `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`。
- T 不是已验证 selector，不得被描述为高价值集合。
- R1 从完整 eligible canonical base 做全局随机，并报告与 T 的自然重叠。
- R2 必须与 T 身份零重叠，并精确匹配 label、historical dynamic bucket、OOF fold 和 `oof_group_id` quota。
- `oof_group_id` 明确是 filename-bucket surrogate，不冒充真实 video ID。
- R2 在匹配前执行字段白名单投影，禁止读取 GapCritical、loss、confidence、mean/std probability、correct rate 等终端信号。
- 任一 quota 不可满足即失败；没有 nearest、relaxed 或隐式 fallback。

### 3.3 common parent 与 lineage

- 每个 training seed 只允许一个 E1–E120 no-replay common parent。
- checkpoint 包含 model、EMA、optimizer、scheduler、AMP scaler、Python/NumPy/Torch CPU/CUDA RNG、epoch、global step 和全部输入身份。
- child 必须绑定 parent SHA、seed、asset、source tree 和 generation identity。
- logical artifact index 将 E1–E120 指向 parent，E121–E200 指向 child，不复制后伪装为 child 原生产物。
- 固定评价 checkpoint 为 E200；禁止 `best.pt`。

### 3.4 fixed base-step replay

- base Dataset 长度、base batch 数、base order、base augmentation、optimizer steps、scheduler、warmup 和 EMA 轨迹不因 replay 增长。
- replay 作为独立 microbatch 注入既定 base step。
- replay microbatch 不超过实际 base batch 的 25%，包括尾 batch。
- replay CE 为逐样本求和后除以 canonical base batch size 128。
- base 与 replay backward 后，每个 base step 只调用一次 optimizer step。
- AMP unscale、clip、scaler step/update 的顺序和调用次数固定。
- replay forward 后恢复 BatchNorm running buffers，并恢复 Python、NumPy、Torch CPU 和全部 CUDA RNG。
- OOM、隐式梯度累积、自动减 batch 和 phase-1 `world_size>1` 均失败封闭。

### 3.5 证据、恢复与评价

- occurrence ledger：每个 base/replay occurrence 一行。
- optimizer-step ledger：每个 base optimizer step 一行。
- exposure ledger：每 epoch 的计划/实际 denominator、numerator、unique、repeat、累计曝光和 steps。
- selection ledger：保存候选全集、选择结果、匹配层和选择原因。
- 大表使用真实 PyArrow Zstd Parquet，按 run/epoch 分区。
- telemetry 记录进程、系统、GPU、CUDA、磁盘、IO 和采样时间；不可用值带 reason code，不填伪造 0。
- epoch 通过 in-progress generation、文件 schema/count/SHA、原子 rename、receipt chain 和 recovery pointer 发布。
- kill、OOM、disk full、半写文件、错误 receipt、错误 RNG、错误 generation 和错误 source identity 均进入 quarantine。
- prediction artifact 绑定 split、manifest、checkpoint、sample-label digest、raw probability 和固定 epoch。
- evaluation 产生 FN=0..95 共 96 个 tie-safe frontier 点，分别保存 `TN_at_FN95` 与 `FN_at_TN68253` 的独立阈值。

### 3.6 Q/R/A/D 与 phase 2

- Q/R/A/D 只允许顺序 gate、stratum 或 factorial 语义。
- 任意加权总分递归拒绝。
- confidence、loss、RHO、gradient、forgetting、AUM 和 coverage 不得登记为 utility。
- 当前无独立 `val_target`，A/gradient alignment 保持 `BLOCKED_BY_VAL_TARGET`。
- short branch、predictor 和 selector 默认为 disabled；第一阶段通过前不可训练 predictor。
- 没有实现或启用 RL selector。

## 4. 最终测试结果

所有最终命令均绑定 `e9b6df6`，原始 stdout/stderr、exit code、bytes 和 SHA-256 登记在 `COMMAND_INDEX.json` 与 `commands/`。

| 范围 | 结果 |
|---|---:|
| 完整 SCTSR v4 | 340 passed |
| 旧 v3 regression | 183 passed, 1 skipped |
| 最终审计工具 | 25 passed |
| CLI 与禁止副作用 | 32 passed |
| Python compileall | PASS |
| uv lock check | PASS |
| synthetic canary A | PASS |
| synthetic canary B | PASS |
| synthetic semantic determinism | 7/7 PASS |
| 真实本地数据工程 canary | PASS |

完整 v4 没有 skip/xfail。旧 v3 命令本身 exit 0，但其真实数量低于任务书的 231 下限，所以 `SA-266` 仍为 FAIL。

## 5. 真实本地数据工程 canary

最终 canary 为 `REALDATA_r8`，绑定最终源码提交与 source-tree digest。它不是 synthetic tensor-only 测试，而是实际完成：

- 加载真实 `yolo11l-cls.pt`，权重 SHA-256 为 `6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`；
- 从 `C:\Sewer-ML\sewerml_train_images` 读取 5 张冻结的真实训练图；
- 使用真实 YOLO11l ClassificationModel 和 ClassificationTrainer 接入层；
- 执行 base forward、replay forward、两部分 backward 和一次 optimizer step；
- 检查 replay 梯度贡献、BN 恢复、RNG 恢复和 optimizer-step 锁定；
- 写出并重新读取 154,740,481-byte checkpoint；
- 写出真实 Zstd Parquet occurrence、optimizer-step、exposure、telemetry、prediction 和 frontier 分区；
- 产生 96 个 FN frontier 点；
- 注入一次 partial kill，并验证 quarantine、receipt chain 和 resume pointer。

最终 receipt：

```text
path: real_data_canary/r8/REAL_DATA_ENGINEERING_CANARY_RECEIPT.json
sha256: 87E55393D9A594FCB8BA3097F40CE178249CE76AB6DF9366C80AA4E0B4BD0A07
checkpoint_bytes: 154740481
checkpoint_sha256: 50D436ECFAE3E31378048A204ABC86831E875FAA31231B8B5281BE92DD55076A
semantic: REAL_DATA_ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT
```

canary 在本机 RTX 4060 上执行，因此只证明真实数据、真实模型和产物链路可运行，不能用于估计 RTX 3090 正式训练时长。154.7 MB checkpoint 超过 GitHub 单文件限制，因此原文件不作为单一 blob 提交，而是发布为 90,000,000 与 64,740,481 bytes 两个原始分卷。分卷逐一绑定 SHA，并已流式验证按序拼接后的 bytes 与 SHA 完全恢复原 checkpoint。远端仍不把“尚未执行重组的目录”冒充直接完整的 canonical generation。

恢复命令：

```powershell
uv run python artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\08_reports\sctsr_v4_implementation_audit_20260813\tools\checkpoint_parts.py reassemble `
  --manifest artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\08_reports\sctsr_v4_implementation_audit_20260813\real_data_canary\r8\checkpoint_parts\CHECKPOINT_PARTS_MANIFEST.json `
  --output <目标路径>\rolling_epoch_0121.generation_1.pt
```

## 6. TDD 历史审计

逐提交 TDD 审计绑定原始 Codex rollout 的不可变行前缀，结果为：

```text
commit_count: 34
behavior_commit_count: 31
non_behavior_commit_count: 3
failing_first_pair_count: 33
raw_event_count: 146
audit_sha256: A42155B08CE6E0E985BE2ACFC1E52562B0B7E4FB28465BB402C7F416E790182C
reviewer_identity: PRIMARY_AGENT_HISTORY_NOT_INDEPENDENT_REVIEW
```

每个行为提交绑定实际 red、patch、historical green、同 test ID exact green 和 commit 输出；red 必须达到目标行为断言，不允许用 import/syntax/setup 偶然失败充数。该证据关闭了原先 `SA-260` 至 `SA-263` 的缺口，但明确不声称独立审稿人。

## 7. Appendix-D 自审计

```text
applicable_check_count: 206
pass_count: 205
fail_count: 1
blocked_count: 0
overall_status: SELF_AUDIT_FAIL
audit_digest: C009E010B74C565E154C9608BB690E34630188664ABC82F048FDDE41B72C93C9
validator_status: VALID_AUDIT_WITH_FAILURES
```

唯一失败：

### SA-266

任务书要求：

```text
uv run pytest tests\stage1_dynamic_replay_v3 -q
至少 231 passed
```

当前冻结树真实结果：

```text
183 passed, 1 skipped in 4.76s
exit_code: 0
```

解决方式只能是：

1. 所有者批准任务书规格变更，承认仓库清理后的 183+1 是当前冻结基线；或
2. 恢复一个经过验证、不会复活已废弃未训练方向的 231-test 历史基线。

不能添加空测试、重复测试或篡改日志来获得 231。

## 8. 十台 RTX 3090 训练容量

用户确认正式训练资源为 10 台机器，每台 1 张 NVIDIA GeForce RTX 3090，共 10 GPU。

容量估算直接读取历史 40 次真实 RTX 3090、YOLO11l、200 epochs、batch 128 训练记录：

```text
historical_run_count: 40
minimum: 14.205 h / 200 epochs
median: 15.507 h / 200 epochs
mean: 15.817 h / 200 epochs
nearest-rank p90: 17.271 h / 200 epochs
maximum: 21.874 h / 200 epochs
```

SCTSR 物理工作量不是 22×8 个完整 200-epoch 独立任务。每个 seed 先训练一个 120-epoch parent，再运行 8 个 80-epoch child，因此每 seed 为 760 个 base epoch-equivalents。8 个 discovery seeds 与 14 个 confirmation seeds 依序通过门控，在 10 GPU 上做 dependency-aware list scheduling：

| 口径 | 预计总时长 |
|---|---:|
| 最接近 120,600 图历史均值 | 125.51 h / 5.23 d |
| 40-run median | 136.46 h / 5.69 d |
| 40-run p90 | 151.98 h / 6.33 d |
| 历史 maximum | 192.49 h / 8.02 d |

SCTSR 还新增 6,402,880 次 replay microbatch 调用、全量 occurrence/step 日志、Zstd Parquet、telemetry、checkpoint hashing、原子事务和 closeout。这些开销尚未在真实 RTX 3090 上完整实测，因此容量报告在 p90 base-only 之上加 20% 运维缓冲，得到：

```text
recommended_reservation: 182.38 h
recommended_calendar_reservation: 8 continuous days
human discovery-review pause: excluded
```

20% 是容量缓冲，不是实测加速比、统计置信区间或训练承诺。正式 release 前应在目标 RTX 3090 上各跑一个 NR 与 replay 臂的一 epoch engineering benchmark，再替换这项不确定开销。

若 discovery gate 不通过并按合同停止，p90+20% 约为 70.47 小时，不应继续消耗 confirmation 预算。

## 9. 仍需解除的正式运行阻断

1. 批准 R2 规格变更或提供可满足零重叠精确 quota 的新冻结资产；禁止静默放宽。
2. 裁决 `SA-266` 的 231-vs-183+1 基线冲突。
3. 完成独立代码审查并签署未来 release。
4. 冻结正式 discovery/confirmation seed registry 和 training identity manifest。
5. 在 10 台 RTX 3090 上完成 NR-vs-replay 一 epoch 工程测速。
6. 若启用 A，先提供独立、群组隔离、SHA 冻结的 `val_target`；否则 A 继续 blocked。
7. 在方法、代码、seed、停止规则和统计规则冻结前，继续密封 blind/test。

## 10. 禁止副作用状态

最终机器可读状态为：

```json
{
  "formal_training_started": false,
  "engineering_gate_generated": false,
  "assignments_generated": false,
  "pilot_release_generated": false,
  "blind_holdout_opened": false,
  "selector_trained": false,
  "method_effectiveness_claimed": false,
  "val_target_available": false,
  "synthetic_registered_as_scientific": false
}
```

## 11. 专家审查入口

建议按以下顺序阅读：

1. `FINAL_IMPLEMENTATION_AUDIT_REPORT.md`
2. `reports/IMPLEMENTATION_SELF_AUDIT_VALIDATION.json`
3. `reports/IMPLEMENTATION_SELF_AUDIT.json`
4. `reports/FORMAL_R2_INFEASIBILITY.json`
5. `reports/REPOSITORY_STATE_AUDIT.json`
6. `real_data_canary/r8/REAL_DATA_ENGINEERING_CANARY_RECEIPT.json`
7. `reports/rtx3090_capacity/RTX3090_CAPACITY_AUDIT.md`
8. `tdd_history/TDD_HISTORY_AUDIT_RECEIPT.json`
9. `reports/MANUAL_LINE_REVIEW.json`
10. `reviewed_e9b6df6/REVIEWED_FILE_SNAPSHOT_MANIFEST.json`
11. `COMMAND_INDEX.json`
12. `EVIDENCE_MANIFEST.json`

## 12. 科学边界

本轮可以推出：SCTSR v4 的合同、固定 base-step runtime、失败封闭、证据采集、真实数据工程路径、恢复、评价和审计机制已经实现，并通过当前可执行的测试与 canary。

本轮不能推出：T、R1、R2、timing、stop、fallback、Q/R/A/D 或 SCTSR 对 FN=0..95 safety frontier 有正效用，也不能推出其跨 unseen training seed 稳定。只有未来严格匹配、预注册、跨 seed 的真实 replay 干预才是 utility evidence。
