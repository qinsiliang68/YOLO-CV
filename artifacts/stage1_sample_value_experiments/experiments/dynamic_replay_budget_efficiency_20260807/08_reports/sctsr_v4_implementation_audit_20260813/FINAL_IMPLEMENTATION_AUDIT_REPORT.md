# SCTSR v4 实施与自我审计最终报告

## 1. 判定先行

本轮已经在冻结的 YOLO-CV 基线上完成隔离的 SCTSR v4 代码实施、测试、真实 PyArrow/Zstd synthetic canary、逐行自审和逐条 Appendix-D 机器审计。代码实现源冻结为：

```text
repository: qinsiliang68/YOLO-CV
baseline: a70ba60485dd32c2f8b4268b8f28ea2d3549f42f
branch: codex/sctsr-v4-taskbook
implementation_source_commit: 8675ebfbc25133607348f358da167d14f1a2f0eb
taskbook_blob_sha: b201d021712e9c6614e119d35f0e14bdf405c6be
source_tree_digest: 8547785D014125255BB2249A2884384D20C0D8D322E0D718F712C5D06ED3433B
```

当前严格判定不是正式放行：

```text
implementation: IMPLEMENTED_AND_MECHANISM_TESTED
appendix_d_self_audit: SELF_AUDIT_FAIL
formal_training_authorized: false
formal_training_started: false
scientific_effectiveness_known: false
```

失败封闭的直接原因有三类：

1. Appendix-D 中 206 项有 201 PASS、5 FAIL。失败项为 SA-260、SA-261、SA-262、SA-263、SA-266。
2. 现有冻结资产无法构造任务书定义的精确 R2：172 个精确分层缺少候选，合计短缺 378 个 occurrence。实现正确返回 `R2_QUOTA_INFEASIBLE`，没有放宽匹配。
3. 没有未来签名 release、正式 seed registry 或 val_target；blind/test 仍密封，A 模块继续 `BLOCKED_BY_VAL_TARGET`。

这些结论不表示 SCTSR 有效或无效。没有进行正式 SCTSR 训练，因此没有任何 Stage1 utility evidence。

## 2. 实施范围

新增实现被隔离在以下代码域：

- `stage1_sctsr_v4/`
- `scripts/stage1_sctsr_v4/`
- `configs/stage1_sctsr_v4/`
- `tests/stage1_sctsr_v4/`
- `integrations/ultralytics/`
- `docs/stage1_sctsr_v4/`

受保护历史域没有被改写：

- `stage1_gapvalue240`
- `stage1_dynamic_replay_v3`
- `YOLOv11/ultralytics`
- 已运行的 v1/v2/v3 queue、release、assignment 和 training evidence

仓库状态审计在实施源提交上为 PASS。tracked worktree 为 clean；大量既有 untracked 文献、审计和历史材料被登记但没有 stage。旧 gate、pilot release 和 assignments 被标记为 legacy detected，不被解释为 active SCTSR v4 状态。

## 3. 已实现的系统合同

### 3.1 百分比预算和八臂

- `ReplayRateSpec` 只接受整数有理数。
- treatment identity pool 为 canonical base 的 25/1000。
- U 为 E121-E200 每 epoch 5/1000。
- F 为 E121-E160 每 epoch 10/1000，E161-E200 为 0。
- 八臂顺序固定为 NR、R1_U、R2_U、T_U、R2_F、T_F、T_TO_R2_AT_160、T_TO_NR_AT_160。
- `CURRENT_LOSS_U` 仅有 HELD 接口，不进入 phase 1。
- 绝对 replay count、浮点 rate、缺失 denominator、不可整除 rate 均失败封闭。

### 3.2 T、R1、R2

- T 绑定 3000 行冻结 stress set，identity digest 为 `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`。
- T 只作为历史符号反转压力集，不是 validated selector。
- R1 从完整 eligible canonical base 做全局随机，并报告自然重叠。
- R2 算法要求与 T 身份零重叠，并精确匹配 label、historical dynamic bucket、OOF fold、`oof_group_id`。
- `oof_group_id` 明确是 filename-bucket surrogate，不冒充真实视频 ID。
- R2 matcher 在匹配前做字段白名单投影，terminal fields 主动访问会抛错。
- 不存在 nearest、relaxed 或隐式 quota fallback。

正式资产尝试证明当前 R2 不可行：172 个 strata、378 个 occurrence shortfall、单 strata 最大 shortfall 为 11。当前正确动作是修改冻结资产或批准科学规格变更，而不是让代码偷偷放宽 R2。

### 3.3 schedule

- 五个 identity groups 互斥并覆盖 pool。
- U 每五个 epoch 每 ID 一次，累计 multiplicity 16。
- F 活跃阶段每五个 epoch 每 ID 两次，累计 multiplicity 16。
- U/F 使用相同 identity digest、总 exposure 和逐 ID multiplicity vector，只改变 epoch distribution。
- E160 stop 与 fallback 被分开记录；fallback 明确切换至 R2-U，stop 明确减少总 dose。
- treatment/comparator 使用同 schedule 的 step-slot skeleton。

### 3.4 common parent、lineage 和恢复

- 每 seed 只允许一个 E1-E120 no-replay common parent。
- checkpoint 绑定 model、EMA、optimizer、scheduler、AMP scaler、Python/NumPy/Torch CPU/CUDA RNG、epoch、global step、seed、lock、source 和 assets identity。
- child 必须通过 parent SHA 和 lineage 启动；裸 checkpoint path 被拒绝。
- logical E1-E120 指向 parent，E121-E200 指向 child。
- parent 产物不可修改；伪造早期 child 产物被拒绝。
- epoch transaction 从 inprogress generation 开始，只在 schema/count/SHA/守恒通过后原子发布。
- kill、OOM、disk full、半写 Parquet、半写 JSON、损坏 receipt 和错误 generation/identity 均走 quarantine。
- resume 只从最后完整 epoch，且不覆盖旧 generation；关键 E120/140/150/160/180/200 checkpoint 保留。

### 3.5 fixed base-step replay runtime

- base Dataset 长度、base batch 数、base order、base augmentation、optimizer steps、scheduler、warmup 和 EMA 轨迹不因 replay 增长。
- replay 作为独立 microbatch 注入既定 base step。
- replay microbatch 不超过实际 base batch 的 25%，包括尾 batch。
- replay CE 为逐样本求和后除以 canonical base batch size 128。
- base loss 仍采用冻结 upstream learner 定义。
- base 和 replay backward 后每 base step 只发生一次 optimizer step。
- AMP unscale、clip、scaler step 和 update 顺序固定且各一次。
- replay forward 后恢复全部 BatchNorm running buffers 和全局 RNG。
- replay augmentation 使用独立 counter domain。
- OOM、隐式梯度累积和 phase-1 world_size 大于 1 均失败封闭。

### 3.6 全量证据

- occurrence ledger：每个 base/replay occurrence 一行。
- optimizer-step ledger：每个 base step 一行。
- exposure ledger：每 epoch 的 planned/actual denominator、numerator、unique、repeat、cumulative 和 steps。
- selection ledger：保存候选全集、选择结果和原因，不只保存 selected IDs。
- 大表使用真实 PyArrow Zstd Parquet，并按 run/epoch 分区。
- 分区 receipt 绑定 schema、row count、bytes 和 SHA-256。
- telemetry 记录 process、system、GPU、CUDA、disk 和 IO；不可用 provider 使用 reason code，不填假 0。
- prediction artifact 绑定 split、manifest、checkpoint、sample-label identity 和每行 raw probability。
- evaluation 生成 FN budget 0-95 的 96 个 tie-safe frontier 点，并分别保存 TN_at_FN95 和 FN_at_TN68253 的阈值。
- discovery/confirmation seed schema 分离，支持 paired completeness、exact sign-flip、Holm、win rate、worst seed 和 dual-end degradation。

### 3.7 Q/R/A/D 和 phase 2

- Q/R/A/D 只允许 gate、stratum 或 factorial 语义。
- weighted total score 被拒绝。
- confidence、loss、RHO、gradient、forgetting、AUM 和 coverage 不可登记为 utility。
- val_target 当前不存在，A enable 必须返回 `BLOCKED_BY_VAL_TARGET` 且不得生成 arm、assignment 或 gradient artifact。
- short-branch、predictor 和 selector 默认 disabled。
- phase 1 gate 未通过时 predictor training 被拒绝。
- 没有实现或启用 RL selector。

## 4. 测试与复现结果

所有命令均在 Python 3.11.14 环境执行，原始 stdout/stderr、exit code、bytes 和 SHA-256 保存在 `COMMAND_INDEX.json` 与 `commands/`。

| 范围 | 结果 |
|---|---:|
| 完整 v4 | 331 passed |
| 旧 v3 regression | 183 passed, 1 skipped |
| contract/rate/schema | 22 passed |
| assets/pools/R2 | 25 passed |
| schedule | 16 passed |
| parent/lineage | 31 passed |
| fixed-step/YOLO | 31 passed |
| evidence/ledgers | 58 passed |
| telemetry | 3 passed |
| evaluation/statistics | 49 passed |
| Q/R/A/D/phase2 | 21 passed |
| transaction/recovery | 16 passed |
| CLI/side effects | 32 passed |
| formal inputs/runtime | 17 passed |
| audit infrastructure | 21 passed |

完整 v4 没有 skip/xfail。旧 v3 命令本身 exit 0，但任务书要求“至少 231 passed”，因此 SA-266 必须 FAIL。不能用 exit 0 隐藏数量不符。

## 5. synthetic canary

最终两次完整 canary 均使用真实 PyArrow/Zstd 路径并通过 `validate_run`：

- 每次覆盖八臂。
- 每臂 16 optimizer steps。
- 每臂 160 prediction rows。
- 每臂生成 96 frontier points。
- 每次 179 个登记 artifact。
- 每次 760 个 logical artifact index entries。
- 每次 6 个故障注入。
- source tree digest 相同。
- parent checkpoint SHA 相同：`21F85E13356EF1A168699DC715A5EAD9750AE19A1236035CA02721DC34C621CC`。
- 7 项稳定语义比较全部通过。

两个运行目录含绝对路径、时间戳和 quarantine 名称，因此整棵目录不要求逐字节相同；稳定合同文件和语义 digest 才是确定性判据。两次 canary 均标记：

```text
SYNTHETIC_NOT_SCIENTIFIC_RESULT
scientific_result=false
formal_training_started=false
method_effectiveness_claimed=false
```

## 6. 逐行自审

冻结源的 reviewed snapshot 包含 160 个 source/test/config 文件，manifest digest 为：

```text
B6270057AA5E0E497E9B2ABFB513CFFB5019CA94FB5DCEA1B54F8BE26552F1F2
```

逐行自审覆盖 SA-280 至 SA-289 共 10 项、24 个精确行锚，包括：

- optimizer 调用边界；
- replay reduction 和 denominator；
- AMP/unscale/clip/step/update；
- BatchNorm buffers；
- Python/NumPy/Torch CPU/CUDA RNG；
- R2 白名单投影；
- 异常路径；
- val_op/test 隔离；
- completion 与 release 隔离；
- public schema registry。

每个行锚均与冻结 snapshot 的 bytes 和 SHA 绑定。reviewer identity 明确为 `SELF_REVIEW_NOT_INDEPENDENT_REVIEW`，不冒充独立专家审查。

## 7. Appendix-D 206 项审计

机器审计输出：

```text
applicable_check_count: 206
pass_count: 201
fail_count: 5
blocked_count: 0
overall_status: SELF_AUDIT_FAIL
audit_digest: 865419C8A29E575AD6ABA70BCC4D6D6718749BEDBA2D596F740EC61B28D41493
validator_status: VALID_AUDIT_WITH_FAILURES
```

失败项：

### SA-260 至 SA-263

初始 rollback units 有 inherited red/green receipts，但后续若干 hardening 修复只有最终 green 命令和会话历史，没有全部固化成“每个行为变更一一对应”的 canonical red/green 文件。因此无法对每个后续修复证明：

- red 一定先于实现；
- red 失败原因一定到达目标 assertion；
- red/green 一定使用同一 pytest node ID；
- 每对 receipt 都有 source-commit、bytes 和 SHA 绑定。

这些历史事实不能事后伪造。解决方式只能是取得并注册真实原始记录，或批准新的审计规格；不能把现在补写的日志冒充当时的 failing-first evidence。

### SA-266

当前旧 v3 suite 为 183 passed、1 skipped，不满足冻结任务书中的 231 passed 下限。仓库清理后删除未训练方向的死代码/测试是合理历史变化，但不能在本任务书审计里擅自改成 PASS。需要批准规格变更或恢复一个真实且不污染历史语义的 231-test 基线。

## 8. 正式运行阻断

在以下阻断全部解除前，不得生成 assignment、engineering gate、pilot release 或正式 seed，也不得启动正式训练：

1. 决定如何处理正式 R2 的 172 个 shortage strata；禁止静默放宽。
2. 解决 SA-266 的旧 v3 数量基线规格冲突。
3. 对 SA-260 至 SA-263 做真实证据补全或正式规格裁决。
4. 进行独立代码审查并签署未来 release。
5. 冻结正式 seed registry 和 training identity manifest。
6. 若启用 A，先取得独立、群组隔离、SHA 冻结的 val_target；否则 A 必须继续 blocked。
7. blind/test 在方法、代码、seed、停止规则和统计规则冻结前继续密封。

## 9. 当前副作用事实

机器可读 repository-state audit 记录：

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

旧历史 gate/release/assignment 实物被分别登记为 legacy detected。它们没有被删除、覆盖或复活。

## 10. 审查入口

建议按以下顺序阅读：

1. `FINAL_IMPLEMENTATION_AUDIT_REPORT.md`
2. `reports/IMPLEMENTATION_SELF_AUDIT_VALIDATION.json`
3. `reports/IMPLEMENTATION_SELF_AUDIT.json`
4. `reports/FORMAL_R2_INFEASIBILITY.json`
5. `reports/REPOSITORY_STATE_AUDIT.json`
6. `reports/MANUAL_LINE_REVIEW.json`
7. `reviewed/REVIEWED_FILE_SNAPSHOT_MANIFEST.json`
8. `COMMAND_INDEX.json`
9. `commands/*/stdout.log` 和 `commands/*/stderr.log`

## 11. 科学边界

本轮能推出：SCTSR v4 的代码合同、失败封闭、证据收集、恢复和 synthetic mechanism path 已被实现并通过大量测试。

本轮不能推出：T、R1、R2、timing、stop、fallback、Q/R/A/D 或 SCTSR 对 FN=0-95 safety frontier 有正效用，也不能推出它们跨 unseen training seed 稳定。只有未来严格匹配的正式配对 replay 干预才是 utility evidence。
