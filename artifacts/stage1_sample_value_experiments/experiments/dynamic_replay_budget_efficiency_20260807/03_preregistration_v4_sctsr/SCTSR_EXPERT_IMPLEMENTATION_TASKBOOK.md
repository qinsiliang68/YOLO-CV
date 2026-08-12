# SCTSR v4 专家代码实施任务书

文档标识：<code>STAGE1_SCTSR_V4_EXPERT_IMPLEMENTATION_TASKBOOK_20260812</code>  
研究名称：State-Conditional Tail-Safe Replay（状态条件尾部安全回流，SCTSR）  
仓库：<code>C:\GitHub\YOLO-CV</code>  
Canonical 路径：<code>C:\GitHub\YOLO-CV\artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\03_preregistration_v4_sctsr\SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md</code>  
专家交付性质：代码施工合同，不是训练授权，不是方法有效性声明，不是科学结果  
当前科学状态：<code>SPECIFICATION_ONLY_NOT_IMPLEMENTED_NOT_RUN</code>  
当前授权范围：单元测试、集成测试、合成 canary、静态验证  
当前明确禁止：正式训练、真实数据 canary、assignment、engineering gate、pilot release、blind holdout/test 访问

---

## 0. 专家必须先读的执行声明

本任务不是继续扩写旧的 RHO-only 矩阵，也不是删除或否定历史 40/120/240-run 结果。本任务要求在仓库中建立一条隔离、可回滚、可审计的 <code>SCTSR v4</code> 实现线，用合格的旧基础设施，替换已经不能回答当前科学问题的实验语义。

专家必须遵守以下九条总则：

1. 不得修改、重命名、覆盖或删除 <code>stage1_gapvalue240</code>、<code>stage1_dynamic_replay_v3</code>、旧 v1/v2/v3 队列、旧 release、旧 assignment、旧训练结果和旧审计证据。
2. 不得把旧 v3 的 <code>231 passed</code> 解释为 SCTSR 已实现；该结果只证明当前 v3 基线可复现。
3. 不得把历史 GapCritical 集合、RHO、loss、confidence、gradient、forgetting、AUM、coverage 或任何 Q/R/A/D 信号称为 utility。
4. 只有同一 common parent、同一 training seed 下，真实 replay treatment 与严格匹配 R2、R1 或 no-replay 的配对干预差，才是 utility evidence。
5. 第一阶段只实现并验证 timing、dose、stop 和 fallback 的因果基础设施。不得训练 selector。
6. 没有独立、群组隔离且 SHA 冻结的 <code>val_target</code> 时，A/gradient-alignment 正式配置必须返回 <code>BLOCKED_BY_VAL_TARGET</code> 并以非零退出码终止。
7. 预算配置只允许有理百分比；绝对样本数只能是由冻结分母推导出的审计字段，不能成为方法配置、arm 名或科学定义。
8. 基础 DataLoader、基础样本顺序、基础 augmentation、基础 batch 数、optimizer step、warmup 和 EMA 更新数必须与 NR 完全相同。Epoch scheduler 只按基础 epoch 边界前进，绝不按 replay forward 或 replay occurrence 前进。Replay 不能增加 optimizer step。
9. 本次专家交付的最终机器状态必须明确写出：
   <code>formal_training_started=false</code>、
   <code>engineering_gate_generated=false</code>、
   <code>assignments_generated=false</code>、
   <code>pilot_release_generated=false</code>、
   <code>blind_holdout_opened=false</code>。

任何与上述规则冲突的“便捷实现”都不允许进入交付。

---

## 1. 研究问题与可证伪对象

### 1.1 核心问题

SCTSR 不把样本价值定义成静态的 <code>V(x)</code>。代码必须支持以下研究对象：

<code>V(S | theta_t, k, schedule, seed)</code>

其中：

- <code>S</code> 是集合级 microset，而不是孤立图片；
- <code>theta_t</code> 是当前 epoch 的完整模型和优化器状态；
- <code>k</code> 是集合内身份此前累计的额外曝光次数；
- <code>schedule</code> 是额外曝光出现的时间和频率；
- <code>seed</code> 是训练初始化、基础顺序、augmentation 和优化路径；
- <code>V</code> 是相对于严格匹配随机 R2 的未来边际效用。

第一阶段并不预测 V。第一阶段只建立能够回答下列问题的干预系统：

1. 固定身份集合在均匀 schedule 下是否优于严格匹配随机？
2. 固定身份集合在前置集中 schedule 下是否优于严格匹配随机？
3. 在身份、累计 occurrence 和每身份 multiplicity 均相同时，只改变时间分布是否改变结果？
4. E160 后继续定向回流、退回 R2 或停止全部 replay 的差异是什么？
5. 任意 replay 相对于 no-replay 是否有一般性收益或一般性伤害？

### 1.2 禁止提前形成的结论

在正式、独立 seed 的配对干预完成前，以下结论均禁止写入代码 receipt、README、报告或 artifact metadata：

- “SCTSR 有效”；
- “历史 T 是高价值样本”；
- “RHO/GapCritical/loss/gradient 能预测 replay utility”；
- “E160 是最优停止点”；
- “前置集中优于均匀”；
- “Q/R/A/D 中某因素有效”；
- “当前 loss 是强基线或弱基线”；
- “固定身份回流比随机更优”。

允许的状态词只有：

- <code>NOT_EVALUATED</code>；
- <code>SPECIFICATION_ONLY</code>；
- <code>IMPLEMENTED_NOT_FORMALLY_RUN</code>；
- <code>SYNTHETIC_NOT_SCIENTIFIC_RESULT</code>；
- <code>BLOCKED_BY_VAL_TARGET</code>；
- <code>REJECTED_BY_VALIDATOR</code>；
- 在未来真实干预后才允许 <code>SUPPORTED</code>、<code>CONTRADICTED</code>、<code>MIXED</code> 或 <code>UNKNOWN_IN_STAGE1</code>。

---

## 2. 已核实仓库事实：实现必须与这些事实对齐

### 2.1 当前代码基线

截至本任务书生成时：

- 分支：<code>codex/stage1-dynamic-oof-replay-v3</code>；
- 调查时 HEAD：<code>d5ebf793e43723a2b942afc82387d7c2ffd3416c</code>；
- 命令 <code>uv run pytest tests\stage1_dynamic_replay_v3 -q</code> 实际为 <code>231 passed</code>；
- 工作树存在与本任务无关的既有修改和未跟踪证据，专家不得清理、reset 或纳入自己的提交；
- <code>stage1_dynamic_replay_v3\matrix.py</code> 仍生成旧的 236-run、RHO-only、C1–C4 HELD 矩阵，不是本任务的科学矩阵；
- <code>stage1_dynamic_replay_v3\draw_plan.py</code> 把 base 和 replay 合并为长度 <code>base + replay</code> 的 draw plan；
- <code>stage1_dynamic_replay_v3\ultralytics_overlay.py</code> 使用 <code>nb = len(train_loader)</code>，并对每个 loader batch 调用一次 optimizer step；
- 因而当前 v3 runtime 会让 replay 增加 batch 和 optimizer step，不能直接用于“固定基础过程、固定 optimizer steps”的 SCTSR 合同。

专家不得用“现有测试通过”掩盖上述语义冲突。正确做法是新建 v4 runtime，并加入失败优先测试证明 v3 型拼接实现会被 v4 合同拒绝。

### 2.2 冻结训练锁和初始权重

以下资产是必须绑定的输入：

| 资产 | 路径 | 字节数 | SHA-256 | 作用 |
| --- | --- | ---: | --- | --- |
| canonical training lock | <code>configs\stage1_gapvalue240\CANONICAL_TRAINING_LOCK_v1.json</code> | 4,623 | <code>7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E</code> | 固定 learner 和超参数 |
| initial checkpoint | <code>yolo11l-cls.pt</code> | 28,553,700 | <code>6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C</code> | 所有 common parent 的共同初始化 |

训练锁中的不可变关键项包括：

- model family：<code>yolo11l</code>；
- epochs：200；
- batch：128；
- workers：4；
- imgsz：224；
- AMP：true；
- deterministic：true；
- patience：0；
- optimizer：auto；
- lr0：0.01；
- lrf：0.01；
- momentum：0.937；
- weight_decay：0.0005；
- augmentation 与 warmup 字段全部沿用锁中值。

专家不得修改 archived learner 或直接改写 <code>YOLOv11</code> 上游源文件。新增行为放在 <code>stage1_sctsr_v4</code> 的窄 overlay/wrapper 中，并由 source tree manifest 绑定实际导入的上游文件。

### 2.3 OOF 分组事实

资产：

- <code>artifacts\stage1_oof_folds_10fold_20260617\metadata.json</code>；
- 字节数 1,076；
- SHA-256 <code>759B7D7E01506694FA508C6F2B040B510458E91056E9192A9F1D0F9101A6F97C</code>；
- 10 folds；
- seed <code>20260606</code>；
- 120,000 行；
- 60,000 positive、60,000 negative；
- 1,156 个 group；
- 当前全部 group 来源是 <code>numeric_filename_bucket</code>；
- bucket size 为 1,000。

因此代码和产物必须把 <code>oof_group_id</code> 标成：

<code>FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID</code>

禁止在字段名、文档、图表或结论中把它称为真实 pipe、inspection 或 video identity。

OOF assignment：

- 路径 <code>artifacts\stage1_oof_folds_10fold_20260617\train_oof_assignments.csv</code>；
- 字节数 51,199,918；
- SHA-256 <code>EE82D19D8B8BC2875842B1DF433CC1B6098D7F88A300B330CA5E45B3642AE0C6</code>。

### 2.4 历史 T 压力集合

第一阶段 treatment identity pool 使用历史压力测试集合的身份，但绝不把它称为已验证 selector。

Canonical 入口：

- <code>artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\generated\selections\RUN_010\selection_manifest.csv</code>；
- 3,000 个 unique IDs；
- 占 canonical base 120,000 的 2.5%；
- 文件字节数 590,747；
- 文件 SHA-256 <code>DFDADC5D75B39A78E0C8995BD46F063C5D969BEA1024E4A332A051D00A172689</code>；
- 角色与样本 ID 的 canonical identity digest：
  <code>85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B</code>。

Identity digest 算法必须固化：

1. 读取 <code>replay_role</code> 和 <code>sample_id</code>；
2. 按二元组字典序排序；
3. 每行编码为 UTF-8：<code>replay_role</code>、TAB、<code>sample_id</code>、LF；
4. 对全部连接字节计算 SHA-256，大写十六进制。

RUN_013、RUN_016 等历史文件可能有不同文件 SHA，但相同 identity digest。v4 只认上述 canonical 入口和 identity digest，不通过“任意一个看起来相同的 CSV”自动替换。

历史结果已观察到同一集合跨 seed 有正向、负向和混合方向。因此它的 v4 角色必须固定为：

<code>HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR</code>

### 2.5 R2 为什么必须重建

历史 R2 与 T 的身份重叠约 91.3%（2,739/3,000），有效身份对比只有约 8.7%。旧 R2 不可沿用。

新的 R2 必须从排除 T 身份后的候选池中采样，并只使用预终端条件精确匹配。已经核实，在当前冻结表中，按 <code>label + dynamic_bucket + oof_fold + oof_group_id quota</code> 做零重叠精确配额是可行的；若未来资产变化导致不可行，代码必须 fail closed，不得自动降级。

可绑定的参考表：

| 资产 | 字节数 | SHA-256 |
| --- | ---: | --- |
| <code>sample_value_table.csv</code> | 31,819,678 | <code>376674FCFD5C8378051FC5D1A588ED415CA8DEFA06A19AAD72505AB49B20B980</code> |
| <code>sample_dynamics_summary.csv</code> | 30,737,786 | <code>84EE044693E9D47295C5A4C6DF05470379FA5CE79B838EDE0CEE3EB1E0DE959F</code> |
| <code>train_oof_assignments.csv</code> | 51,199,918 | <code>EE82D19D8B8BC2875842B1DF433CC1B6098D7F88A300B330CA5E45B3642AE0C6</code> |

注意：这些表中同时存在允许字段和禁止字段。R2 builder 必须使用白名单投影后再获得数据；不能把整行 DataFrame 交给 matching 函数。

### 2.6 数据角色事实

当前仓库存在 <code>train</code>、OOF、<code>val_model</code>、<code>val_cal</code>、<code>val_op</code> 和 <code>test</code> 语义，但没有可用的独立 <code>val_target</code>。

第一阶段 timing/stop/fallback 可以不依赖 A，因此可以实现。A 正式配置必须阻断。不得从 <code>val_op</code>、<code>val_cal</code>、<code>val_model</code> 或 test 中临时切一块冒充 <code>val_target</code>。

---

## 3. 代码边界与目标目录

### 3.1 必须新建

专家必须建立以下隔离目录，不得在旧模块内“顺手改几行”：

    stage1_sctsr_v4/
        __init__.py
        contracts.py
        errors.py
        asset_registry.py
        rate_spec.py
        identity_pool.py
        arm_spec.py
        schedule.py
        random_controls.py
        terminal_field_guard.py
        common_parent.py
        branch_lineage.py
        logical_artifact_index.py
        replay_step_plan.py
        rng_isolation.py
        bn_isolation.py
        fixed_step_runtime.py
        checkpointing.py
        epoch_transaction.py
        recovery.py
        occurrence_ledger.py
        step_ledger.py
        exposure_ledger.py
        telemetry.py
        prediction_artifact.py
        evaluation.py
        statistics.py
        qrad_contract.py
        short_branch_scaffold.py
        completion.py
        source_identity.py

    scripts/stage1_sctsr_v4/
        validate_contract.py
        validate_assets.py
        build_identity_pools.py
        build_schedule.py
        validate_schedule.py
        run_synthetic_canary.py
        run_common_parent.py
        run_branch.py
        evaluate_checkpoint.py
        validate_run.py
        closeout_run.py

    configs/stage1_sctsr_v4/
        contract_v1.json
        asset_registry_v1.json
        arms_phase1_v1.json
        schema_registry_v1.json
        runtime_policy_v1.json
        disabled_phase2_v1.json

    tests/stage1_sctsr_v4/
        conftest.py
        fixtures/
        test_rate_spec.py
        test_contract.py
        test_asset_registry.py
        test_identity_pool.py
        test_random_controls.py
        test_schedule.py
        test_replay_step_plan.py
        test_common_parent.py
        test_branch_lineage.py
        test_logical_artifact_index.py
        test_rng_isolation.py
        test_bn_isolation.py
        test_fixed_step_runtime.py
        test_occurrence_ledger.py
        test_step_ledger.py
        test_exposure_ledger.py
        test_telemetry.py
        test_epoch_transaction.py
        test_recovery.py
        test_prediction_artifact.py
        test_evaluation.py
        test_statistics.py
        test_qrad_contract.py
        test_short_branch_scaffold.py
        test_completion.py
        test_cli_synthetic.py
        test_no_formal_side_effects.py

### 3.2 可以复用但必须通过 v4 适配器调用

以下 v3 思路可以复用，不允许直接假设它们已满足 v4：

- prediction identity；
- tie-safe raw safety frontier；
- epoch transaction；
- runtime checkpoint；
- recovery quarantine；
- source provenance；
- resource monitor；
- completion receipt；
- strict OOM 行为。

建议做法是复制最小必要逻辑到 v4，并保留来源注释与对照测试。不得让 v4 正式运行时隐式 import 一个会被后续 v3 改动改变语义的可变实现。

### 3.3 明确禁止修改

- <code>stage1_gapvalue240\</code>；
- <code>scripts\stage1_gapvalue240\</code>；
- <code>tests\stage1_gapvalue240\</code>；
- <code>stage1_dynamic_replay_v3\</code> 的行为；
- <code>YOLOv11\ultralytics\</code> 上游 learner；
- <code>03_preregistration_v2\</code>、<code>04_run_queue_v2\</code>；
- 任何旧 release、assignment、training output；
- blind holdout/test 资产。

若专家发现必须修改这些路径才能实现 v4，应停止并提交阻断说明，不得自行扩大范围。

---

## 4. 公共类型：字段、验证和序列化必须固定

所有公共类型使用 frozen dataclass 或等价不可变 Pydantic model。序列化必须稳定排序，SHA 使用 canonical UTF-8 JSON：键排序、无多余空白、LF 换行、禁止 NaN/Infinity。

### 4.1 ReplayRateSpec

字段：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| <code>numerator</code> | int | 大于等于 0 |
| <code>denominator</code> | int | 大于 0 |
| <code>semantic</code> | enum | <code>IDENTITY_POOL_RATE</code> 或 <code>PER_EPOCH_REPLAY_RATE</code> |
| <code>denominator_role</code> | enum | 必须为 <code>CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE</code> |

方法：

- <code>reduced()</code>：返回最简分数；
- <code>derive_count(base_denominator: int)</code>：要求乘积整除；
- <code>canonical_token()</code>：例如 <code>5/1000</code>；
- <code>to_basis_points()</code>：仅审计展示，不作为配置真值。

禁止：

- float 配置；
- <code>0.005</code> 直接作为 schema 真值；
- <code>600</code> 作为 replay budget；
- 自动四舍五入；
- floor/ceil；
- 根据 GPU 容量自动改 rate。

固定值：

- identity pool：<code>25/1000</code>；
- U：<code>5/1000</code>；
- F-active：<code>10/1000</code>；
- NR/停止区间：<code>0/1000</code>。

对于 base denominator 120,000，派生值分别是 3,000、600、1,200 和 0。派生值必须进入 ledger，但不能反向成为配置输入。

### 4.2 FixedIdentityPoolSpec

字段：

- <code>pool_id: str</code>；
- <code>pool_role: enum[T_STRESS, R1_GLOBAL_RANDOM, R2_MATCHED_RANDOM, CURRENT_LOSS_HELD]</code>；
- <code>rate: ReplayRateSpec</code>；
- <code>base_manifest_sha256: str</code>；
- <code>source_manifest_path: str</code>；
- <code>source_manifest_sha256: str</code>；
- <code>identity_digest: str</code>；
- <code>identity_digest_algorithm: str</code>；
- <code>unique_count_derived: int</code>；
- <code>label_quota: map[str,int]</code>；
- <code>oof_fold_quota: map[str,int]</code>；
- <code>dynamic_bucket_quota: map[str,int]</code>；
- <code>oof_group_quota: map[str,int]</code>；
- <code>oof_group_semantic: str</code>；
- <code>construction_seed: int | null</code>；
- <code>selection_semantic: str</code>。

验证：

- identity 唯一；
- 所有 identity 属于 canonical base；
- 数量等于百分比派生值；
- T 的 canonical digest 必须匹配；
- R2 与 T 交集必须为 0；
- quota 总和必须等于 unique count；
- surrogate 语义字段必须准确；
- 输入表和生成结果都必须有 SHA。

### 4.3 SctsrArmSpec

字段：

- <code>arm_id</code>；
- <code>display_name</code>；
- <code>phase</code>；
- <code>identity_policy</code>；
- <code>schedule_policy</code>；
- <code>epoch_start</code>；
- <code>epoch_end</code>；
- <code>fallback_epoch</code>；
- <code>fallback_identity_policy</code>；
- <code>requires_common_parent=true</code>；
- <code>optimizer_step_policy=BASE_STEP_LOCK</code>；
- <code>formal_state=HELD</code>；
- <code>scientific_role</code>；
- <code>allowed_comparators</code>。

Validator 必须拒绝：

- 未登记 arm；
- 隐含 absolute count；
- E1–E120 replay；
- 未绑定 parent；
- replay 额外 step；
- <code>best.pt</code>；
- current-loss 混入 phase 1；
- A 配置在无 val_target 时启用。

### 4.4 CommonParentSpec

字段：

- <code>parent_id</code>；
- <code>training_seed</code>；
- <code>epoch_start=1</code>；
- <code>epoch_end=120</code>；
- <code>arm_id=COMMON_PARENT_NR</code>；
- <code>canonical_training_lock_sha256</code>；
- <code>initial_checkpoint_sha256</code>；
- <code>base_manifest_sha256</code>；
- <code>source_tree_digest</code>；
- <code>expected_base_steps_per_epoch=938</code>；
- <code>replay_rate=0/1000</code>；
- <code>checkpoint_schema_version</code>。

### 4.5 ReplayStepPlan

字段：

- <code>run_id</code>；
- <code>arm_id</code>；
- <code>training_seed</code>；
- <code>epoch</code>；
- <code>base_step_count</code>；
- <code>planned_replay_occurrences</code>；
- <code>step_slot_seed</code>；
- <code>identity_order_seed</code>；
- <code>per_step_replay_counts: list[int]</code>；
- <code>per_step_identity_slices: list[list[str]]</code>；
- <code>max_fraction=1/4</code>；
- <code>plan_digest</code>。

不允许仅保存 seed 而不保存最终计划。最终每 step 计划必须物化并哈希。

### 4.6 BranchLineage

字段：

- <code>logical_run_id</code>；
- <code>parent_id</code>；
- <code>parent_checkpoint_path</code>；
- <code>parent_checkpoint_sha256</code>；
- <code>parent_checkpoint_epoch=120</code>；
- <code>training_seed</code>；
- <code>arm_id</code>；
- <code>child_source_tree_digest</code>；
- <code>child_contract_digest</code>；
- <code>created_at_utc</code>；
- <code>lineage_digest</code>。

任何一个字段不匹配都必须拒绝启动 child。

### 4.7 ExposureLedgerRow

字段：

- run/arm/seed/epoch；
- denominator role；
- denominator planned/actual；
- rate numerator/denominator；
- replay numerator planned/actual；
- unique planned/actual；
- repeat occurrences in epoch；
- cumulative occurrences；
- cumulative unique IDs；
- minimum/maximum/mean per-ID multiplicity；
- optimizer steps planned/actual；
- base samples planned/actual；
- schedule digest；
- identity pool digest；
- status 和 failure code。

### 4.8 LogicalArtifactIndex

一个 logical child 的 E1–E120 不复制为 child 产物。索引字段：

- logical run ID；
- logical epoch；
- physical owner type：<code>PARENT</code> 或 <code>CHILD</code>；
- physical run ID；
- artifact relative path；
- artifact SHA；
- checkpoint SHA；
- source tree digest；
- lineage digest。

规则：

- E1–E120 必须指向 parent；
- E121–E200 必须指向 child；
- child 目录中出现伪造的 E1–E120 原生产物时 validator 失败；
- parent 物理文件必须只读；
- index 不得通过复制让多个文件拥有模糊身份。

---

## 5. 百分比预算与固定 schedule

### 5.1 冻结分母

唯一合法分母：

<code>每个 epoch canonical base optimizer-visible exposure</code>

当前冻结分母为 120,000。该数字必须从已哈希的 base manifest 推导，并与 asset registry 一致；不能在 schedule 实现里硬编码成方法真值。

### 5.2 固定身份池和五组分区

T pool 为 base 的 2.5%。对每个固定 identity pool，必须确定性分为五个等大 group：

- <code>G0</code> 到 <code>G4</code>；
- 每组为 base 的 0.5%；
- 当前派生每组 600 IDs；
- 五组互斥，联合等于完整 2.5% pool。

分组算法：

1. 对每个 sample ID 计算
   <code>SHA256(pool_digest + NUL + sample_id)</code>；
2. 按 digest、sample ID 二级排序；
3. round-robin 分配到 G0–G4；
4. 验证每组百分比乘分母可整除；
5. 输出 <code>identity_group_membership.parquet</code> 和 digest。

禁止按原排名连续切组，因为那会把 rank 范围与 schedule 混杂。

### 5.3 U schedule

E121–E200，每 epoch replay rate 为 0.5%。

令 <code>j = (epoch - 121) mod 5</code>，该 epoch 使用 <code>Gj</code>。

性质：

- 每个 5-epoch block，pool 内每个 ID 恰好出现 1 次；
- 80 个 epoch 共 16 个 block；
- 每个 ID 累计 replay multiplicity 为 16；
- 总 occurrence 是 base 的 40%；
- 当前派生为 48,000，但该绝对值只出现在验证 ledger。

### 5.4 F schedule

E121–E160，每 epoch replay rate 为 1.0%；E161–E200 为 0。

令 <code>j = (epoch - 121) mod 5</code>，active epoch 使用：

- <code>Gj</code>；
- <code>G((j + 4) mod 5)</code>。

即每 epoch 两个不同 group，不在同一 epoch 重复同一 ID。

性质：

- 每个 5-epoch block，每个 ID 恰好出现 2 次；
- E121–E160 共 8 个 block；
- 每个 ID 累计 replay multiplicity 为 16；
- 累计 occurrence 与 U 完全相同；
- 身份 pool 与 U 完全相同；
- 唯一变化是时间分布；
- E161–E200 为零 replay。

### 5.5 stop 与 fallback

- <code>T_TO_R2_AT_160</code>：
  - E121–E160 使用 T-U；
  - E161–E200 使用 R2-U；
  - 后半段不是“无 replay”，而是“停止定向，退回匹配随机”；
  - 总 replay occurrence 保持为 U 的 40%；
  - 该 arm 是 replacement policy，不是纯 T timing effect。

- <code>T_TO_NR_AT_160</code>：
  - E121–E160 使用 T-U；
  - E161–E200 为零 replay；
  - 总 replay occurrence 只有 U 的一半；
  - 这是有意改变剂量的“停止全部 replay”策略；
  - 不得与 T-U 直接解释为纯身份或纯 timing 因果。

### 5.6 schedule validator 的硬断言

必须验证：

- U 与 F 的 identity digest 相同；
- U 与 F 的累计 occurrence 相同；
- U 与 F 的每 ID multiplicity 全部相同；
- U 与 F 只有 epoch distribution 不同；
- 同 schedule 的 T/R2 step-slot skeleton 相同；
- fallback E160 前与 T-U 完全相同；
- stop E160 前与 T-U 完全相同；
- NR 全 epoch replay rate 为零；
- E1–E120 所有 arm 都没有 replay；
- 所有派生 count 都由有理 rate 与哈希分母计算；
- 不允许任何 rounding。

---

## 6. 第一阶段八臂矩阵

固定八臂如下，名称和语义不得自行扩展：

| arm_id | E121–E160 | E161–E200 | 主要作用 |
| --- | --- | --- | --- |
| <code>NR</code> | no replay | no replay | 基础过程 |
| <code>R1_U</code> | global random U | global random U | 身份中立随机 replay |
| <code>R2_U</code> | matched random U | matched random U | T-U 的严格身份对照 |
| <code>T_U</code> | T stress set U | T stress set U | 固定 T、均匀曝光 |
| <code>R2_F</code> | matched random F | no replay | T-F 的严格身份对照 |
| <code>T_F</code> | T stress set F | no replay | 固定 T、前置集中 |
| <code>T_TO_R2_AT_160</code> | T-U | R2-U | 停止定向并 fallback |
| <code>T_TO_NR_AT_160</code> | T-U | no replay | 停止全部 replay |

<code>CURRENT_LOSS_U</code> 只实现接口、构造 validator 和 synthetic 测试，状态固定为 <code>HELD_NOT_IN_PHASE1</code>。它不得进入第一阶段 arm registry、assignment 或运行矩阵。

### 6.1 预注册直接比较

| contrast_id | treatment | comparator | 可回答问题 | 不可回答问题 |
| --- | --- | --- | --- | --- |
| C01 | T_U | R2_U | 均匀 schedule 下 T 身份是否优于匹配随机 | 不能证明 T 固有有效 |
| C02 | T_F | R2_F | 前置 schedule 下 T 身份是否优于匹配随机 | 不能单独证明 timing |
| C03 | T_F | T_U | 固定 T 身份、累计剂量和 multiplicity 后的 timing 差 | 不能推广到任意 selector |
| C04 | R2_F | R2_U | 匹配随机身份下的 timing 差 | 不能证明 T 的价值 |
| C05 | T_TO_R2_AT_160 | T_U | E160 后 fallback 策略差 | 不是纯 stop effect |
| C06 | T_TO_NR_AT_160 | T_TO_R2_AT_160 | 同一前缀后，晚期任意 replay 对 no replay 的差 | 后半剂量有意不同 |
| C07 | R2_U | NR | matched random replay 的一般曝光效应 | 不能证明选择有效 |
| C08 | R1_U | NR | global random replay 的一般曝光效应 | 不能证明匹配有效 |

所有结果必须以 contrast_id 报告，不能挑选方便解释的跨臂差异。

---

## 7. T、R1、R2 和 current-loss 的构造要求

### 7.1 T

T 构造器只负责：

- 读取 canonical manifest；
- 验证文件 SHA；
- 验证 2.5% rate；
- 验证 sample ID、label、OOF fold、dynamic bucket；
- 验证 identity digest；
- 输出固定 pool。

不得重新按 GapCritical 排序，不得根据本轮 seed 或 checkpoint 刷新。

### 7.2 R1 global random

R1 候选全集是 canonical base 中符合 replay role 的全部合法训练身份，排除：

- 不在 train/base 的身份；
- label/asset 损坏；
- blind/validation/test 身份。

R1 的抽样 universe 必须包含满足上述条件的完整 canonical base；不得因为某个身份属于 T 就将其排除，否则它不再是 global random。R1 与 T 的自然随机重叠允许存在，但必须报告重叠数量、比例和 identity digest，不能把 R1 用作 T 的零重叠身份对照。T 的严格身份对照只能是 R2。

R1 只匹配：

- identity pool rate；
- schedule；
- total occurrence；
- per-ID multiplicity。

R1 不匹配 T 的 dynamic bucket、fold 或 group quota。使用冻结 <code>selection_seed</code> 的 counter-based hash 抽样，并保存完整候选 universe digest 和选择前后行数。

### 7.3 R2 method-matched random

R2 必须与 T 零身份重叠，并精确匹配以下预终端字段：

- <code>y_true / label</code>；
- <code>historical_dynamic_bucket</code>；
- <code>oof_fold</code>；
- <code>oof_group_id</code> quota；
- identity pool rate；
- U/F schedule；
- per-ID multiplicity。

允许进入 R2 builder 的列白名单：

- sample_id；
- y_true；
- replay_role；
- historical_dynamic_bucket；
- oof_fold；
- oof_group_id；
- group_source；
- base_manifest_membership；
- source manifest identity。

禁止进入 builder 进程内 matching frame 的字段：

- rank；
- GapCritical；
- loss；
- current loss；
- confidence；
- mean probability；
- probability standard deviation；
- correct rate；
- RHO；
- gradient；
- forgetting；
- AUM；
- feature distance；
- future epoch outcome；
- val_model/val_cal/val_op/test 指标；
- 任何由 T 的终端选择公式产生的字段。

实现方式：

1. 在读表层创建 <code>TerminalFieldGuard</code>；
2. 先按明确白名单 project；
3. 再把白名单 frame 传给 matcher；
4. matcher 对每个精确 stratum 计算需要数和可用数；
5. 任一 stratum 不足，抛出 <code>R2_QUOTA_INFEASIBLE</code>；
6. 不允许退化为 fold-only、bucket-only 或 nearest quota；
7. 每个可行 stratum 内用 counter hash 随机选择；
8. 输出完整 quota comparison、候选数、排除数、抽样 seed、结果 digest。

必须有测试证明：即使禁止字段与 sample_id 一起存在于原 CSV，matcher 也无法访问它们。测试应使用会在属性访问时抛错的 sentinel column/accessor，而不是只检查最终输出没有这些列。

### 7.4 current-loss

只实现 disabled builder 接口：

- 输入必须是 common parent 或注册 checkpoint 的当前训练身份 loss；
- 必须有明确 lag；
- 禁止同 epoch 先看 loss 再回流同一个 augmentation；
- 输出必须带 <code>CANDIDATE_SIGNAL_NOT_UTILITY</code>；
- phase 1 registry 遇到该 arm 必须失败；
- 无正式 selector gate 时 runner 必须拒绝。

---

## 8. Common parent 与分叉语义

### 8.1 Parent

每个 training seed 只训练一次 E1–E120 no-replay parent。所有八个 child 使用该 seed 的同一 parent checkpoint。

Parent runner 必须：

- 从冻结初始权重开始；
- 绑定 canonical training lock；
- E1–E120 完全无 replay；
- 保存 E120 的完整状态；
- 完成 parent closeout 后把 parent 标记只读；
- 生成 parent receipt 和 SHA；
- 不创建任何 child。

### 8.2 E120 checkpoint 必填状态

Checkpoint 至少包含：

- model state；
- EMA state 和 EMA update count；
- optimizer state；
- scheduler state；
- AMP GradScaler state；
- Python random state；
- NumPy RNG state；
- Torch CPU RNG state；
- 每个可见 CUDA device RNG state；
- epoch=120；
- global step；
- base sampler/dataloader generation；
- canonical training lock SHA；
- initial weight SHA；
- base manifest SHA；
- training seed；
- source tree digest；
- runtime config digest；
- input asset registry digest；
- checkpoint payload digest。

禁止只保存 <code>last.pt</code> 路径而不验证 payload。禁止使用 <code>best.pt</code>。

### 8.3 Child 启动

Child 启动顺序：

1. 加载 BranchLineage；
2. 重新计算 parent checkpoint SHA；
3. 验证 parent epoch、seed、source、lock、base manifest；
4. 验证 parent receipt canonical completion；
5. 在只读模式加载 parent；
6. 恢复全部 RNG/optimizer/scheduler/EMA/scaler；
7. 创建 child 的 E121 transaction；
8. 不得写 parent 目录；
9. 不得复制 E1–E120 trainer 目录；
10. logical artifact index 指向 parent。

如果两个 arm 的 parent checkpoint 字节不同，则 paired seed 失效，后续结果不得进入配对分析。

---

## 9. 固定 base-step 的 replay gradient injection

这是本实现最关键的工程变更。

### 9.1 Base DataLoader 绝对不含 replay

Base DataLoader 必须只包含 canonical base：

- 每 epoch 120,000 optimizer-visible base occurrences；
- batch 128；
- 938 base batches（937 个完整 batch和 1 个尾 batch）；
- 与 NR 相同的 base sample order；
- 与 NR 相同的 augmentation seed；
- 与 NR 相同的 worker seed；
- 与 NR 相同的 batch boundaries；
- 与 NR 相同的 optimizer step count。

不得使用 v3 的 <code>base + replay</code> Dataset 长度。不得把 replay 拼入 loader。

### 9.2 Replay step-slot 分配

对每个 epoch：

1. 令 <code>N</code> 为 replay occurrence 派生数；
2. 令 <code>B=938</code> 为 base step 数；
3. 计算 <code>q, r = divmod(N, B)</code>；
4. 每 step 先分配 q 个；
5. 用 counter hash 从 B 个 step 中选择 r 个，各加 1；
6. 选择过程只依赖 training seed、epoch、schedule family，不依赖 treatment identity；
7. 同 schedule 的 T 与 R2 必须得到相同 <code>per_step_replay_counts</code>；
8. 将当 epoch identity pool 做独立确定性 permutation；
9. 按 step count 顺序切片分配；
10. 保存整个计划与 digest。

当前派生示例：

- U：600 occurrences，600 个 step 各 1 个，338 个 step 为 0；
- F-active：1,200 occurrences，262 个 step 各 2 个，676 个 step 各 1 个；
- NR/F-inactive：全部 0。

Replay microbatch 上限：

<code>floor(actual_base_batch_size × 1/4)</code>

尾 batch 使用实际 base batch size 计算 cap。任何 step 超 cap 都必须在开跑前失败，不得运行时拆成多个 optimizer step。

### 9.3 Loss 和 backward

Base loss 完全沿用冻结 learner：

<code>L_base = upstream classification loss</code>

Replay contribution：

<code>L_replay = sum_i CE(logits_i, y_i) / canonical_base_batch_size</code>

其中 <code>canonical_base_batch_size=128</code>，即使最后一个 base batch 较小也不改变分母。

单个 base step 的逻辑：

    zero_grad 已在上一步完成
    base_batch = next(base_loader)
    base_logits = model(base_batch)
    L_base = upstream_loss(base_logits, base_labels)
    scaler.scale(L_base).backward()

    replay_ids = replay_step_plan[base_step]
    if replay_ids 非空:
        保存 replay 前全局 RNG
        保存所有 BatchNorm running_mean/running_var/num_batches_tracked
        使用独立 replay RNG 和独立 augmentation 生成 replay_microbatch
        replay_logits = model(replay_microbatch)
        per_sample_ce = cross_entropy(replay_logits, replay_labels, reduction="none")
        L_replay = per_sample_ce.sum() / 128
        scaler.scale(L_replay).backward()
        恢复 BatchNorm buffers
        恢复 replay 前全局 RNG
    else:
        L_replay = 0

    只调用一次 upstream optimizer_step
    global_step += 1
    warmup/global-step/EMA 只随该 base step 前进一次
    epoch scheduler 只在冻结的基础 epoch 边界调用，不受 replay 数量影响

要求：

- 两次 backward 累加到同一梯度；
- clip 在合并梯度 unscale 后执行一次；
- scaler.step、scaler.update 各一次；
- optimizer.zero_grad 一次；
- EMA update 一次；
- Replay forward 不得改变下一 base batch 的 RNG；
- Replay forward 后 BN running buffers digest 必须恢复到 replay 前值；
- replay 梯度必须保留；不能通过 <code>torch.no_grad</code> 或错误 detach 消失；
- base loss 数值和 base forward 路径不得因 arm 改写；
- 不得启用隐式 gradient accumulation；
- 第一版只允许单进程单 GPU；<code>world_size != 1</code> 必须返回 <code>DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1</code>。

### 9.4 为什么恢复 BN 而不恢复参数梯度

Replay 的目的就是给参数增加梯度，因此参数梯度保留。BN running buffers 是 forward 的状态副作用；如果保留，会让 replay 通过额外 forward 改变 base 模型轨迹，混入未计量的 BN 曝光。实现必须枚举所有 <code>_BatchNorm</code> 模块的：

- running_mean；
- running_var；
- num_batches_tracked。

保存和恢复都在同一 device/dtype 上完成。测试必须验证：

- replay 后 buffer 逐字节相同；
- 参数 gradient 与 no-replay 不同；
- 下一 base forward 的 RNG/augmentation 与 NR 相同。

### 9.5 OOM 和异常

发生 OOM：

- 不自动减 batch；
- 不减少 replay count；
- 不更换 precision；
- 不改变 imgsz/workers；
- 不拆 step；
- 不继续下一 batch；
- 标记当前 epoch transaction 为 partial；
- 写异常 receipt；
- quarantine；
- 释放可释放缓存只用于安全退出；
- resume 只能从最后一个完整 epoch checkpoint。

---

## 10. RNG、augmentation 和基础轨迹相等性

### 10.1 RNG 域

必须至少分成以下 counter domains：

- <code>base_order</code>；
- <code>base_augmentation</code>；
- <code>replay_identity_order</code>；
- <code>replay_step_slots</code>；
- <code>replay_augmentation</code>；
- <code>R1_selection</code>；
- <code>R2_stratum_selection</code>；
- <code>synthetic_fixture</code>。

每个 domain seed 从以下 canonical payload 派生：

<code>SHA256(domain + NUL + training_seed + NUL + epoch + NUL + optional_sample_or_step)</code>

取固定字节转换为无符号整数。不得使用 Python 内置 <code>hash()</code>。

### 10.2 必须记录的 digest

- epoch 开始 RNG digest；
- 每个 base batch 前 RNG digest；
- replay RNG fork digest；
- replay 后恢复 digest；
- epoch 结束 RNG digest；
- base sample order digest；
- base augmentation trace digest；
- step-slot digest；
- replay identity order digest。

paired arms 的 base order 和 base augmentation digest 必须逐 epoch相同，否则 run validation 失败。

---

## 11. 全量数据采集合同

总体原则：

- 大表：Zstd Parquet；
- 小型合同、manifest、receipt：canonical JSON；
- 所有 Parquet 按 <code>run_id/epoch</code> 分区；
- 每个分区有 row count、schema version、字节数、SHA；
- 任何表不能只保留 epoch 平均值而丢失原始 occurrence；
- null 只允许出现在 schema 明确声明 nullable 的字段，同时必须有对应 reason code；
- 不得使用空字符串、<code>unknown</code> 或未登记 sentinel。

### 11.1 occurrence_ledger

每个 base occurrence 和 replay occurrence 一行。

必填字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| run_id | string | logical child 或 parent |
| parent_id | string | parent 自身填自己的 ID |
| arm_id | string | 登记 arm |
| training_seed | int64 | 配对 seed |
| epoch | int16 | 1–200 |
| base_batch_index | int32 | 0–937 |
| global_step_before | int64 | step 前 |
| occurrence_role | enum | BASE 或 REPLAY |
| occurrence_index_in_step | int16 | base/replay 内序号 |
| sample_id | string | canonical identity |
| y_true | int8 | 0/1 |
| replay_role | string | base 时用 NOT_APPLICABLE_BASE |
| identity_pool_id | string | base 时用 NOT_APPLICABLE_BASE |
| identity_group | string | G0–G4 或 NOT_APPLICABLE_BASE |
| selection_policy | string | BASE_CANONICAL、T_STRESS、R1、R2 等 |
| selection_reason_code | string | 登记 reason |
| oof_fold | int8 | 0–9 |
| oof_group_id | string | surrogate |
| oof_group_semantic | string | 固定 surrogate 声明 |
| historical_dynamic_bucket | string | 冻结值 |
| augmentation_seed | uint64 | 独立 domain |
| augmentation_trace_digest | fixed string | 参数和输出 identity 摘要 |
| replay_count_before | int32 | base 为 0 并标 reason |
| replay_count_after | int32 | 本 occurrence 后 |
| last_replay_epoch | int16 nullable | 从未 replay 可 null |
| last_replay_epoch_reason | string | NEVER_REPLAYED 或 NOT_APPLICABLE_BASE |
| epochs_since_last_replay | int16 nullable | 合法 null 有 reason |
| logit_normal | float32 | 原始 logit |
| logit_defect | float32 | 原始 logit |
| p_defect_raw | float32 | 未校准概率 |
| ce_unreduced | float32 | 逐样本 CE |
| margin_defect_minus_normal | float32 | 方向固定 |
| predicted_label_argmax | int8 | 0/1 |
| correct_argmax | bool | 逐 occurrence |
| oof_reference_probability | float32 nullable | 仅存在 OOF 时 |
| oof_reference_reason | string | PRESENT 或 REGISTERED_NOT_AVAILABLE |
| rho_candidate_signal | float32 nullable | 仅诊断 |
| rho_reason | string | CANDIDATE_SIGNAL_NOT_UTILITY 或未报告原因 |
| row_generation | int32 | transaction generation |

Replay occurrence 还必须记录：

- <code>planned_replay_epoch</code>；
- <code>planned_step_slot</code>；
- <code>cumulative_replay_count_before/after</code>；
- <code>pool_multiplicity_target</code>；
- <code>schedule_family</code>；
- <code>fallback_state</code>。

### 11.2 optimizer_step_ledger

每个 base step 一行：

- run/parent/arm/seed/epoch；
- base_batch_index；
- global_step_before/after；
- base_batch_size；
- replay_microbatch_size；
- replay rate numerator/denominator；
- base_loss；
- replay_loss；
- combined_loss_for_reporting；
- base_loss_items；
- parameter grad norm before clip；
- parameter grad norm after clip；
- clip max norm；
- optimizer step count delta，必须为 1；
- LR：每个 param group；
- momentum/betas：每个 param group；
- weight decay：每个 param group；
- AMP scale before/after；
- overflow/step_skipped；
- EMA updates before/after；
- scheduler state digest；
- warmup progress；
- BN digest before replay；
- BN digest after replay restore；
- RNG digest before replay；
- RNG digest after replay restore；
- base augmentation digest；
- replay augmentation digest；
- dataloader wait seconds；
- base forward seconds；
- replay forward seconds；
- backward seconds；
- optimizer seconds；
- write buffer bytes；
- status。

如果 GradScaler 因 overflow 跳过 optimizer update，必须如实记录。paired arms 的 skipped step 不一致时，该 epoch 不满足配对轨迹，不能悄悄当作等步。

### 11.3 epoch_exposure_ledger

每 epoch 一行：

- planned/actual base denominator；
- rate 分数；
- planned/actual replay numerator；
- unique replay IDs；
- repeat occurrences；
- cumulative occurrence；
- per-ID multiplicity min/max/mean/quantiles；
- base optimizer steps planned/actual；
- EMA updates delta；
- scheduler epoch transitions delta；
- base order digest；
- base augmentation digest；
- replay schedule digest；
- identity pool digest；
- occurrence partition SHA；
- step ledger partition SHA；
- checkpoint SHA；
- write seconds；
- dataloader wait seconds；
- training seconds；
- evaluation seconds；
- disk bytes written；
- transaction generation；
- validation status。

### 11.4 selection_ledger

T/R1/R2 构造必须保存候选全集和选择结果，不能只保存入选 ID。

字段：

- candidate sample ID；
- eligibility；
- exclusion reason；
- allowed strata；
- stratum quota required/available；
- selection counter hash；
- selected bool；
- selected pool；
- terminal field guard digest；
- source row asset SHA；
- duplicate/overlap status。

禁止把 terminal fields 复制进 R2 selection ledger。只可记录 <code>TERMINAL_FIELDS_NOT_LOADED</code> 和 guard digest。

### 11.5 resource_telemetry

采样周期固定 1 秒。每行：

- timestamp UTC 和 monotonic seconds；
- run/arm/seed/epoch；
- process PID；
- process CPU percent；
- process RSS/VMS；
- process read bytes/write bytes；
- process read count/write count；
- system CPU percent；
- system memory total/available/used/percent；
- GPU index/UUID/name；
- GPU utilization；
- GPU memory used/total；
- GPU temperature；
- GPU power；
- CUDA allocated/reserved/max allocated/max reserved；
- 运行卷 disk total/free/used；
- artifact 卷 disk total/free/used；
- telemetry provider status；
- provider error code。

Telemetry provider 失败不能默默填 0。硬件字段不可用时必须填 null 并给登记 reason；关键字段全不可用时 run 不可 closeout。

### 11.6 prediction artifact

每个评价 split/checkpoint 一份 Parquet：

- run/arm/seed；
- split role；
- split manifest path 和 SHA；
- sample_id；
- y_true；
- logit_normal；
- logit_defect；
- p_defect_raw；
- checkpoint epoch；
- checkpoint SHA；
- model/EMA 选择；
- source tree digest；
- prediction generation；
- sample-label identity digest；
- row count。

不得只保存已排序概率；必须保留完整 sample-label 一一对应。

### 11.7 frontier artifact

FN budget 0–95 每个整数一行：

- fn_budget；
- actual_fn；
- tn；
- fp；
- tp；
- threshold；
- threshold_rule <code>p_defect_raw &gt;= threshold</code>；
- tie_size；
- reachable；
- defect_count；
- normal_count；
- normalized_tn；
- checkpoint SHA；
- prediction artifact SHA。

另存小型 summary：

- raw frontier normalized AUC；
- TN_at_FN95；
- FN_at_TN68253；
- threshold_at_FN95；
- threshold_at_TN68253；
- target TN reachable；
- 两个阈值是否相同，只作事实字段；
- tie diagnostics。

严禁把两个阈值拼成同一个 confusion matrix。

---

## 12. 产物目录与原子事务

每个 synthetic 或未来正式 run 使用：

    <experiment_root>/
        00_contract/
        01_assets/
        02_parent/
        03_branch/
        04_ledgers/
            occurrence/run_id=<id>/epoch=<eeee>/
            optimizer_step/run_id=<id>/epoch=<eeee>/
            exposure/
            selection/
            telemetry/run_id=<id>/epoch=<eeee>/
        05_checkpoints/
        06_predictions/
        07_evaluation/
        08_receipts/
        09_quarantine/
        ARTIFACT_INDEX.json
        RUN_MANIFEST.json

### 12.1 每 epoch 事务

写入流程：

1. 创建唯一 <code>epoch_XXXX.generation_N.inprogress</code>；
2. 只向 inprogress 写；
3. flush 并关闭所有 Parquet/JSON；
4. 验证 schema、row count、曝光守恒和 step count；
5. 计算每个文件 SHA；
6. 写 generation manifest；
7. 原子 rename 为 complete；
8. 写 append-only receipt；
9. 更新 logical artifact index；
10. 更新 rolling recovery pointer。

任何步骤失败：

- 不更新 canonical pointer；
- inprogress 移入 quarantine；
- 记录失败原因和原路径；
- 不覆盖上一个 generation；
- 不自动继续下一 epoch。

### 12.2 checkpoint 保留

必须保留：

- E120 parent；
- E140；
- E150；
- E160；
- E180；
- E200；
- 当前 rolling last-complete；
- 恢复所需的上一个 rolling checkpoint。

固定评价 checkpoint 是 E200。其他 checkpoint 只是预注册轨迹锚点，不得用于选方法或选停止点。

### 12.3 Resume

Resume 只允许：

- 同 logical run；
- 同 parent SHA；
- 同 arm；
- 同 training seed；
- 同 source tree；
- 同 contract；
- 同 asset registry；
- 同 generation chain；
- 从最后完整 epoch；
- 当前 partial 已 quarantine。

错误 RNG、错误 source、错误 parent、损坏 receipt、缺失分区、磁盘不足都必须拒绝 resume。

---

## 13. 评价实现

### 13.1 Tie-safe raw safety frontier

定义正类为 defect，预测规则为：

<code>p_defect_raw &gt;= threshold</code>。

同概率样本必须作为完整 tie group 同时跨阈值，禁止按 sample ID 拆 tie 来人为获得更好点。

对每个 FN budget <code>b=0..95</code>：

1. 枚举 tie-group boundary；
2. 保留 <code>FN &lt;= b</code> 的可行点；
3. 选择 TN 最大点；
4. TN 相同时按冻结 tie-breaker：threshold 更高优先，再 actual FN 更低优先；
5. 记录 actual FN、TN、threshold 和 tie size。

Primary：

<code>nAUC = (1/95) × sum_{b=0}^{94} 0.5 × (TN_b/N_normal + TN_{b+1}/N_normal)</code>

注意：该归一化只除以 FN 轴长度 95 和 normal count，不把横轴再缩放或重复除以 95。

如果 <code>max_fn=0</code> 的测试夹具，则直接使用该点的 normalized TN。

### 13.2 双端锚点

- <code>TN_at_FN95</code>：FN budget 95 的 frontier 点 TN；
- <code>FN_at_TN68253</code>：在 <code>TN &gt;= 68,253</code> 的阈值中最小 FN；
- 两者分别记录阈值；
- target TN 不可达时返回 <code>reachable=false</code> 和 null FN，并在比较中按预注册失败规则处理；
- 禁止用线性插值伪造可达阈值。

### 13.3 数据角色

- <code>val_model</code>：只允许 discovery 诊断，不可进入正式 endpoint；
- <code>val_cal</code>：如需要单一部署阈值，只用于冻结该阈值；
- <code>val_op</code>：只评价冻结方法、冻结 E200 checkpoint；
- <code>test</code>：全部代码、阈值、方法和停止规则冻结前保持密封；
- <code>val_target</code>：只为未来 A 方向使用，当前不存在。

<code>val_op</code> 不得选择 method、stop epoch、checkpoint 或 threshold。

---

## 14. 第一阶段统计接口

### 14.1 探索与确认严格分离

代码支持两个不相交 seed phase：

- timing discovery：8 个完全未见 training seeds；
- confirmatory：14 个新的完全未见 training seeds。

专家本次不生成正式 seed assignment。实现 seed registry schema 和 validator，正式 seed 值由未来 release authority 在训练授权前一次性冻结。当前状态写 <code>FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE</code>，不是空字段。

历史训练 seed、选择 seed和 discovery/confirmation seed 必须去重。所有八臂在同一 phase 内按 training seed 完全配对。

### 14.2 Discovery 规则

Discovery 只决定：

- 是否存在足够信号值得进入确认；
- 是否否证 timing/identity/stop 某机制；
- 是否触发预注册备选路线。

不得声称方法有效。

进入下一阶段的最低稳定性：

- 对预注册的关键 contrast，至少 7/8 primary delta 同方向；
- safety 不得出现明确系统性双端恶化；
- 报告全部 8 seed、worst seed、胜率和双端恶化；
- 不得删除“异常 seed”；
- 不得根据 val_op 结果重新选择 E160。

### 14.3 Confirmation 规则

未来确认仍需：

- 14 个全新 seed；
- 至少 12/14 primary delta 为正；
- worst-seed primary delta 非负；
- 同一 seed 的 TN_at_FN95 更差且 FN_at_TN68253 更差的次数为 0；
- exact paired sign-flip test；
- Holm familywise alpha 0.05；
- 缺一对即该比较 fail closed；
- 不允许 efficacy early stop。

第一阶段 contrast family 的具体 Holm family 必须在正式 release 前由统计 JSON 冻结；代码不得运行后修改 family。实现需支持 C01–C08 的完整原始 p 值、排序、调整阈值和判定记录。

### 14.4 报告字段

每个 contrast：

- treatment/comparator；
- paired seed IDs；
- 每 seed treatment endpoint；
- 每 seed comparator endpoint；
- 每 seed delta；
- mean、median；
- positive/zero/negative count；
- win rate；
- worst seed 和 worst delta；
- TN_at_FN95 delta；
- FN_at_TN68253 delta；
- dual-end degradation；
- exact sign-flip p；
- Holm rank、threshold、decision；
- missing pair status；
- scientific state。

---

## 15. Q/R/A/D 合同和第二阶段 disabled scaffolding

### 15.1 固定概念

- Q：reliability；
- R：residual/reducible learnability；
- A：对独立 FN95 局部目标的方向；
- D：集合条件下的覆盖与冗余。

只允许：

- 顺序 gate；
- 分层；
- 预注册析因。

禁止：

- <code>wQ × Q + wR × R + wA × A + wD × D</code>；
- 任意可调权重总分；
- 将 candidate signal 标记为 utility；
- 用同一 R2 同时不透明地吸收 R/A/D 所有匹配语义。

### 15.2 A 的 fail-closed 接口

实现 <code>TargetAlignmentConfig</code> 和 validator，但正式 enable 条件必须全部满足：

- 独立 <code>val_target</code>；
- 与 train/OOF/val_model/val_cal/val_op/test identity-disjoint；
- group-disjoint；
- manifest SHA 冻结；
- 角色登记；
- 禁止 test；
- 禁止 val_op；
- 局部 FN95 目标公式冻结；
- checkpoint lag 冻结。

当前任何 enable 请求必须：

- 非零退出；
- error code <code>BLOCKED_BY_VAL_TARGET</code>；
- 不生成 arm；
- 不生成 assignment；
- 不做 gradient forward。

### 15.3 短分支 scaffold

只实现数据结构和 synthetic path：

- common-parent anchor：E120、E140、E160；
- horizon：5 或 10 epochs；
- microset rate：有理百分比；
- treatment microset；
- 严格匹配 R2 microset；
- outcome：未来固定 checkpoint 的 T-vs-R2 endpoint delta；
- label 是向量和 gate，不是加权总分；
- predictor feature table 可容纳 Q/R/A/D、state、累计 exposure；
- predictor 只允许低容量模型接口；
- 默认 <code>enabled=false</code>；
- phase 1 gate 未通过时训练 predictor 必须失败。

第一版不得实现强化学习 selector。

---

## 16. CLI 详细合同

所有 CLI：

- 使用 <code>uv run python</code>；
- 默认只读或 synthetic；
- 明确输出 JSON receipt；
- 非法状态非零退出；
- 不因 validation PASS 自动进入下一阶段；
- 不扫描并复活旧 queue；
- 不写 blind/test；
- 支持 <code>--output</code> 的显式路径；
- 默认禁止 formal。

### 16.1 validate_contract.py

命令形态：

    uv run python scripts/stage1_sctsr_v4/validate_contract.py ^
      --contract configs/stage1_sctsr_v4/contract_v1.json ^
      --arms configs/stage1_sctsr_v4/arms_phase1_v1.json ^
      --schemas configs/stage1_sctsr_v4/schema_registry_v1.json ^
      --output <receipt.json>

检查：

- 百分比有理数；
- 八臂精确集合；
- current-loss HELD；
- A blocked；
- no weighted score；
- formal flags false；
- fixed checkpoint；
- test sealed。

### 16.2 validate_assets.py

输入 asset registry，逐项检查 path、size、SHA、row count、identity digest、split role 和互斥性。不得修改资产。

### 16.3 build_identity_pools.py

模式只允许：

- <code>--pool T_STRESS</code>；
- <code>--pool R1_GLOBAL_RANDOM</code>；
- <code>--pool R2_MATCHED_RANDOM</code>；
- <code>--pool CURRENT_LOSS_HELD --validate-only</code>。

R2 必须传 T digest 和 terminal-field guard。输出 pool manifest、membership Parquet、quota audit 和 receipt。

### 16.4 build_schedule.py / validate_schedule.py

生成 U/F/stop/fallback 的 epoch plans 和 replay step plans。validate 必须同时加载 T 与 R2，从跨 arm 角度验证 parity，不能只验证单个 CSV。

### 16.5 run_synthetic_canary.py

只使用小型 synthetic dataset、tiny model、synthetic assets。产物根必须包含：

<code>SYNTHETIC_NOT_SCIENTIFIC_RESULT</code>

Canary 要覆盖：

- parent；
- 八臂 schedule；
- replay gradient；
- RNG/BN 隔离；
- checkpoint/resume；
- evaluation；
- completion。

不得读取真实 val_op/test。

### 16.6 run_common_parent.py

必须有参数：

- contract；
- asset registry；
- training seed registry；
- output root；
- execution mode；
- release authorization。

默认 <code>execution_mode=synthetic</code>。当请求 formal 时，必须验证未来签名 release manifest；当前仓库没有该 manifest，所以必须拒绝。

### 16.7 run_branch.py

除 parent 参数外必须传：

- arm ID；
- BranchLineage；
- identity pool；
- schedule；
- parent checkpoint；
- artifact index。

禁止无 lineage 直接用 checkpoint 路径启动。

### 16.8 evaluate_checkpoint.py

只接受登记 checkpoint epoch。正式 endpoint 默认且只能是 E200。E120/140/150/160/180 只能用 <code>trajectory</code> 模式，receipt 必须声明 <code>NOT_FOR_METHOD_SELECTION</code>。

### 16.9 validate_run.py / closeout_run.py

validate_run 是只读全量 audit。closeout 只有 validate_run PASS 后才发布 canonical completion。closeout 不能发布 release 或 assignment。

---

## 17. 测试清单：必须失败优先

每个功能先写失败测试，保存 red receipt，再实现，保存 green receipt。测试命名必须能对应下列要求。

### 17.1 百分比和合同

- 拒绝 float rate；
- 拒绝 absolute replay count；
- 拒绝缺失 denominator role；
- 拒绝不可整除百分比；
- 验证 25/1000、5/1000、10/1000；
- 拒绝 phase 1 第九个 arm；
- 拒绝 current-loss 进入 phase 1；
- 拒绝 weighted Q/R/A/D；
- 拒绝 <code>best.pt</code>；
- 拒绝 test access；
- 拒绝 val_op selection。

### 17.2 identity pool

- T 文件 SHA 错误失败；
- T identity digest 错误失败；
- pool 数量不等于 rate 派生失败；
- duplicate ID 失败；
- 非 base ID 失败；
- 五组联合/互斥/等率；
- 分组与 rank 无关；
- R1 从完整 eligible canonical base 抽样，并报告与 T 的自然重叠；
- R2 与 T 零重叠；
- R2 精确 label/bucket/fold/group quota；
- quota 不可行时失败；
- 禁止字段 accessor 被触发时测试立即失败；
- R2 不允许 fallback matching。

### 17.3 schedule

- U 每 ID 16 次；
- F 每 ID 16 次；
- U/F 累计 occurrence 相同；
- U/F identity digest 相同；
- U/F 只时间分布不同；
- F active epoch 同 ID 不重复；
- E161–E200 F 为 0；
- fallback E160 前与 T-U 一致；
- stop E160 前与 T-U 一致；
- T/R2 同 schedule step slots 一致；
- NR 全零。

### 17.4 fixed-step runtime

- NR 与 replay arm base order 相同；
- base augmentation 相同；
- base batch count 938；
- optimizer steps 938；
- scheduler/EMA 更新数相同；
- replay 有梯度；
- replay backward 后每 base batch只 optimizer step 一次；
- replay 不进入 Dataset length；
- replay microbatch 不超过 25%；
- 尾 batch cap 正确；
- replay CE 使用 sum/128；
- base loss 保持 upstream；
- RNG replay 前后相同；
- BN buffers replay 前后相同；
- parameter gradients因 replay 改变；
- OOM 不自动减 batch；
- world_size 大于 1 拒绝。

### 17.5 parent 和 lineage

- 同 seed/配置 parent checkpoint deterministic；
- 不同 arm 引用同一 parent SHA；
- child 不修改 parent；
- 错 parent SHA 拒绝；
- 错 seed 拒绝；
- 错 source tree 拒绝；
- 错 asset registry 拒绝；
- parent epoch 不是 120 拒绝；
- logical E1–E120 指向 parent；
- child 伪造历史 epoch 文件拒绝。

### 17.6 ledger 和 telemetry

- occurrence conservation；
- planned/actual numerator；
- unique/repeat/multiplicity；
- step count conservation；
- EMA/scheduler count；
- Parquet schema strict；
- Zstd compression；
- 分区 SHA；
- telemetry 1 秒 cadence；
- process IO；
- disk free；
- GPU/CUDA fields；
- provider failure 不填假 0。

### 17.7 evaluation

- FN 0–95 恰好 96 点；
- tie group 不可拆；
- normalized AUC；
- TN_at_FN95；
- FN_at_TN68253；
- 两个独立阈值；
- unreachable target TN；
- prediction identity reorder/missing/duplicate 失败；
- checkpoint 非 E200 的正式评价失败；
- best.pt 失败。

### 17.8 recovery

- kill；
- OOM；
- disk full；
- 半写 Parquet；
- 半写 JSON；
- receipt 损坏；
- SHA 不匹配；
- RNG 不匹配；
- generation 不匹配；
- source tree 不匹配；
- parent 不匹配；
- partial quarantine；
- 从最后完整 epoch resume；
- 不覆盖旧 generation。

### 17.9 no-side-effect

测试 CLI 后必须断言：

- 没有正式训练目录；
- 没有 assignment；
- 没有 engineering gate；
- 没有 pilot release；
- 没有 blind/test access receipt；
- 旧 v2/v3 文件 mtime 和 SHA 未改变；
- synthetic 产物都有 synthetic 标记。

---

## 18. 提交顺序和最小回滚单元

专家不得把全部实现塞进一个大提交。建议严格使用以下八个提交；每个提交只 stage 自己的文件，不得顺手纳入现有脏工作树。

### Commit 1：合同、类型和 schema

包含：

- errors；
- contracts；
- rate spec；
- arm spec；
- config schemas；
- 失败优先测试；
- red/green test receipts。

验收：

- rate 和 arm 合同测试 PASS；
- 尚无 runner。

### Commit 2：身份池、R1/R2 和 schedule

包含：

- asset registry；
- terminal field guard；
- T/R1/R2；
- five-group partition；
- U/F/stop/fallback；
- schedule tests。

验收：

- 零重叠；
- quota exact；
- U/F parity；
- 禁止字段不可访问。

### Commit 3：common parent 和 lineage

包含：

- full-state checkpoint schema；
- parent validation；
- BranchLineage；
- LogicalArtifactIndex；
- tests。

### Commit 4：固定 base-step replay injection

包含：

- ReplayStepPlan；
- RNG isolation；
- BN isolation；
- fixed-step runtime；
- upstream overlay adapter；
- gradient/step tests。

这是最高风险提交；不得与 telemetry 或 evaluation 混合。

### Commit 5：全量 ledger、telemetry 和恢复

包含：

- occurrence/step/exposure；
- telemetry；
- epoch transaction；
- recovery/quarantine；
- partition manifest；
- failure tests。

### Commit 6：预测、评价、统计和 completion

包含：

- prediction identity；
- raw frontier；
- thresholds；
- paired statistics；
- completion audit；
- tests。

### Commit 7：held Q/R/A/D 和短分支 scaffold

包含：

- Q/R/A/D no-weight validator；
- A blocked interface；
- short-branch disabled schema；
- low-capacity predictor data interface；
- 不能训练 selector 的测试。

### Commit 8：CLI、synthetic canary 和运行手册

包含：

- scripts；
- synthetic fixtures；
- CLI integration tests；
- source tree manifest；
- 全套 validation receipt；
- README。

每个提交前：

- <code>git diff --check</code>；
- 定向测试；
- 记录命令、exit code、日志 bytes/SHA；
- 只 stage 此提交路径；
- 不 push，除非仓库 owner 另行授权。

---

## 19. 验收命令

专家交付时至少执行并登记：

    uv run pytest tests\stage1_sctsr_v4 -q

    uv run pytest tests\stage1_dynamic_replay_v3 -q

    uv run python scripts\stage1_sctsr_v4\validate_contract.py ...

    uv run python scripts\stage1_sctsr_v4\validate_assets.py ...

    uv run python scripts\stage1_sctsr_v4\validate_schedule.py ...

    uv run python scripts\stage1_sctsr_v4\run_synthetic_canary.py ...

    uv run python scripts\stage1_sctsr_v4\validate_run.py ...

    git diff --check

要求：

- v3 基线不得低于当前 231 passed；
- v4 全套必须全部通过；
- 任一 skipped/xfailed 必须有逐项理由，核心合同测试不允许 skip；
- 测试输出保存原始日志；
- 每条命令保存 exit code、日志字节数、SHA；
- synthetic canary receipt 明确不是科学结果。

---

## 20. 机器可读 completion audit

专家必须交付 <code>SCTSR_V4_IMPLEMENTATION_COMPLETION_AUDIT.json</code>。至少包含：

- schema version；
- implementation status；
- source tree digest；
- commit list；
- changed file ledger；
- frozen asset checks；
- contract validation；
- rate validation；
- T identity validation；
- R1 validation；
- R2 zero-overlap/quota validation；
- schedule parity；
- parent checkpoint completeness；
- lineage tests；
- logical artifact index tests；
- fixed base step tests；
- replay gradient test；
- RNG isolation；
- BN isolation；
- exposure conservation；
- telemetry；
- prediction identity；
- frontier；
- statistics；
- recovery；
- synthetic canary；
- v3 regression；
- v4 test suite；
- known blockers；
- prohibited side-effect flags。

整体 PASS 的含义只允许是：

<code>IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION</code>

必须另外保存以下布尔值：

- <code>formal_training_started=false</code>；
- <code>engineering_gate_generated=false</code>；
- <code>assignments_generated=false</code>；
- <code>pilot_release_generated=false</code>；
- <code>blind_holdout_opened=false</code>；
- <code>selector_trained=false</code>；
- <code>method_effectiveness_claimed=false</code>。

如果旧目录里检测到历史 gate/release/assignment，必须分开记录：

- <code>legacy_engineering_gate_detected=true</code>；
- <code>legacy_pilot_release_detected=true</code>；
- <code>legacy_assignments_detected=true</code>；
- <code>active_v4_engineering_gate_generated=false</code>；
- <code>active_v4_pilot_release_generated=false</code>；
- <code>active_v4_assignments_generated=false</code>。

不得通过忽略旧历史文件来伪称“仓库从未生成过 gate”。

---

## 21. 专家交付包清单

专家应交付：

1. v4 source；
2. scripts；
3. configs；
4. tests；
5. synthetic fixtures；
6. README/运行手册；
7. 每个提交的说明；
8. red/green receipts；
9. source tree manifest；
10. asset validation receipt；
11. schedule parity receipt；
12. synthetic canary 完整 artifact index；
13. test command ledger；
14. completion audit；
15. 已知限制和剩余阻断；
16. 明确的 no-formal-side-effect 声明。

交付报告每个“已修复”必须指向：

- 失败优先测试名；
- red 日志路径和 SHA；
- green 日志路径和 SHA；
- 实现文件与行号；
- 剩余风险。

只引用 Markdown、只说“测试过”、只给文件数或只给总代码行数都不算验证。

---

## 22. 当前剩余科学与发布阻断

即使专家实现全部通过，以下事项仍然阻断正式训练：

1. 本任务书需要独立代码审查；
2. 正式 seed registry 尚未由 release authority 冻结；
3. 正式 training release 尚未生成；
4. 独立 <code>val_target</code> 尚不存在，因此 A 仍阻断；
5. BudgetedReplay 报告声明的三个源码载体仍缺失，专家仓库逐行审计未完成；
6. 旧 v2/v3 gate、assignment 和 release 是历史证据，不能复活；
7. blind holdout/test 仍密封；
8. SCTSR、历史 T 和所有 candidate signal 均未被正式干预验证。

这些阻断必须写进 completion audit，不能因代码测试通过而消失。

---

## 23. 失败代码注册表

至少实现以下稳定 error codes：

- <code>ABSOLUTE_BUDGET_FORBIDDEN</code>
- <code>FLOAT_RATE_FORBIDDEN</code>
- <code>RATE_NOT_INTEGRAL</code>
- <code>DENOMINATOR_IDENTITY_MISMATCH</code>
- <code>IDENTITY_DIGEST_MISMATCH</code>
- <code>T_POOL_ROLE_MISMATCH</code>
- <code>R1_UNIVERSE_NOT_GLOBAL</code>
- <code>R2_OVERLAPS_T</code>
- <code>R2_QUOTA_INFEASIBLE</code>
- <code>TERMINAL_FIELD_ACCESS_FORBIDDEN</code>
- <code>SCHEDULE_EXPOSURE_MISMATCH</code>
- <code>SCHEDULE_MULTIPLICITY_MISMATCH</code>
- <code>BASE_ORDER_MISMATCH</code>
- <code>BASE_AUGMENTATION_MISMATCH</code>
- <code>BASE_STEP_COUNT_MISMATCH</code>
- <code>REPLAY_MICROBATCH_CAP_EXCEEDED</code>
- <code>REPLAY_ADDED_OPTIMIZER_STEP</code>
- <code>BN_BUFFER_NOT_RESTORED</code>
- <code>RNG_NOT_RESTORED</code>
- <code>PARENT_CHECKPOINT_INCOMPLETE</code>
- <code>PARENT_SHA_MISMATCH</code>
- <code>BRANCH_LINEAGE_MISMATCH</code>
- <code>CHILD_MUTATED_PARENT</code>
- <code>LOGICAL_ARTIFACT_IDENTITY_MISMATCH</code>
- <code>BEST_PT_FORBIDDEN</code>
- <code>VAL_OP_SELECTION_FORBIDDEN</code>
- <code>TEST_ACCESS_FORBIDDEN</code>
- <code>BLOCKED_BY_VAL_TARGET</code>
- <code>WEIGHTED_QRAD_FORBIDDEN</code>
- <code>CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN</code>
- <code>DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1</code>
- <code>OOM_FIXED_CONTRACT_ABORT</code>
- <code>DISK_SPACE_PRECHECK_FAILED</code>
- <code>ATOMIC_TRANSACTION_INCOMPLETE</code>
- <code>RESUME_GENERATION_MISMATCH</code>
- <code>SYNTHETIC_RESULT_MISLABELLED</code>
- <code>FORMAL_RELEASE_NOT_AUTHORIZED</code>
- <code>LEGACY_ASSIGNMENT_REUSE_FORBIDDEN</code>

Error JSON 包含：

- code；
- message；
- failing field；
- observed；
- expected；
- artifact path；
- run/arm/seed/epoch；
- recoverable；
- required action。

---

## 24. 专家施工时的判断边界

专家可以自行决定：

- 内部函数拆分；
- dataclass 或 Pydantic 的选择；
- Parquet writer 的已有依赖适配；
- 测试 fixture 的小型数值；
- 性能优化，只要不改变合同。

专家不得自行决定：

- 改 arm；
- 改 E120/E160/E200；
- 改 2.5%/0.5%/1.0%；
- 改 U/F multiplicity；
- 放宽 R2；
- 改 loss denominator；
- 增加 optimizer step；
- 用 best.pt；
- 从 val_op 选策略；
- 开 test；
- 启用 A；
- 加权 Q/R/A/D；
- 训练 selector；
- 生成正式队列或 release；
- 启动真实训练。

遇到后者必须停下，提交 <code>SPECIFICATION_CHANGE_REQUEST</code>，由 owner 另行决定。

---

## 25. 最终验收定义

本任务完成的唯一正确表述：

> SCTSR v4 的隔离实现、失败封闭合同、固定 base-step replay 注入、全量证据记录、恢复机制、评价与 disabled 后续接口已经通过代码验收；尚未启动正式训练，尚未评价任何方法有效性。

不得写成：

> SCTSR 已证明有效。

也不得写成：

> 因为 synthetic canary 通过，所以可直接发布训练。

代码验收通过后，下一步是独立审查和正式 release 决策，不是自动执行训练。

---

## 附录 A：最低 source tree manifest

Source tree manifest 必须覆盖：

- 全部 <code>stage1_sctsr_v4</code>；
- 全部 v4 scripts/configs/tests；
- 实际 import 的 YOLOv11 文件；
- 实际 import 的 v3 适配来源；
- dependency lock；
- Python/torch/CUDA/driver/Ultralytics 版本；
- git HEAD；
- dirty state；
- 未跟踪但可 import 文件；
- 每个文件相对路径、字节数、SHA。

正式 release 必须 clean。Synthetic 开发期可 dirty，但 receipt 必须如实记录，不能说 clean。

## 附录 B：最小 synthetic canary 断言

Synthetic 数据至少：

- base denominator 能被 25/1000、5/1000、10/1000 整除；
- 有 2 类；
- 有 5 个 identity groups；
- 有多个 OOF fold 和 surrogate group；
- T/R2 零重叠；
- 有概率 ties；
- 有 target TN 可达和不可达两个评价 fixture；
- 有可注入 OOM、kill、disk-full、corrupt-receipt 故障点。

Canary 不要求复现 YOLO 精度，但必须真实执行 forward/backward/optimizer/EMA/checkpoint。

## 附录 C：实现验收时不得遗漏的原始事实

- 当前 v3 基线是 231 passed；
- 当前 v3 仍是旧 236-run RHO-only matrix；
- v3 loader 语义会让 replay 增加 batch/step；
- 当前没有独立 val_target；
- T 是 2.5% 压力集合，identity digest 已冻结；
- 历史 R2 有约 91.3% 重叠，不能复用；
- 新 R2 必须零重叠并只匹配预终端条件；
- OOF group 是 filename bucket surrogate；
- learner、200 epochs、batch 128 和初始权重固定；
- U/F 用百分比定义并确保身份、累计 exposure、每身份 multiplicity 相同；
- replay 必须注入 base step，不增加 optimizer step；
- E200 固定评价，best.pt 禁止；
- test 保持密封；
- synthetic 不是科学结果；
- implementation PASS 不是 training authorization。

---

## 附录 D：专家交付前强制自我审计与测试需求清单

本附录必须在专家交付前逐项执行。专家必须把它复制为机器可读 checklist；每一项只能是 <code>PASS</code>、<code>FAIL</code> 或 <code>BLOCKED_WITH_REASON</code>。只有本任务明确允许长期阻断的项目（例如正式训练、正式 seed、val_target 和 blind test）可以使用 <code>BLOCKED_WITH_REASON</code>；实现、单元测试、synthetic canary、证据完整性项目不得用阻断状态替代失败。

每一项都必须提供：

- <code>check_id</code>；
- <code>status</code>；
- <code>requirement</code>；
- <code>evidence_paths</code>；
- <code>reproduction_command</code>；
- <code>exit_code</code>；
- <code>stdout_log_path</code>；
- <code>stdout_log_bytes</code>；
- <code>stdout_log_sha256</code>；
- <code>stderr_log_path</code>；
- <code>stderr_log_bytes</code>；
- <code>stderr_log_sha256</code>；
- <code>observed_result</code>；
- <code>expected_result</code>；
- <code>reviewed_source_files</code>；
- <code>reviewed_test_files</code>；
- <code>remaining_risk</code>；
- <code>required_action_if_not_pass</code>。

空字段、只有 Markdown 声明、只写“人工检查通过”、只给测试数量、无日志 SHA 或无复现命令，均不能计为 PASS。

### D.1 范围和仓库卫生自审

- [ ] SA-001：本次所有实现只位于登记的 v4 source/scripts/configs/tests 和 v4 artifact 目录。
- [ ] SA-002：<code>stage1_gapvalue240</code> 的 tracked source SHA 与实施前快照相同。
- [ ] SA-003：<code>stage1_dynamic_replay_v3</code> 没有被 v4 提交修改。
- [ ] SA-004：<code>YOLOv11\ultralytics</code> 上游 learner 没有被修改。
- [ ] SA-005：旧 v1/v2/v3 queue、release、assignment、training evidence 的 SHA/mtime 没有被改变。
- [ ] SA-006：没有把已有工作树中的无关修改 stage 或 commit。
- [ ] SA-007：没有新建含义不清的 <code>tmp</code>、<code>final</code>、<code>fixed</code>、<code>new</code> 或根目录散落文件。
- [ ] SA-008：所有生成型资产有 owner、source、consumer、lifecycle、manifest 和 verification。
- [ ] SA-009：source truth、human truth、generated output、runtime log 和 synthetic result 没有混放。
- [ ] SA-010：<code>git diff --check</code> exit 0。
- [ ] SA-011：source tree manifest 覆盖全部可 import 文件，包括未跟踪文件检测。
- [ ] SA-012：专家报告如实记录 dirty/clean，不隐藏开发期脏状态。

### D.2 合同和百分比自审

- [ ] SA-020：ReplayRateSpec 只接受整数有理数。
- [ ] SA-021：absolute replay count 输入被失败测试拒绝。
- [ ] SA-022：float rate 输入被失败测试拒绝。
- [ ] SA-023：缺失 canonical denominator identity 被拒绝。
- [ ] SA-024：不可整除 rate 被拒绝，不做 rounding。
- [ ] SA-025：2.5%、0.5%、1.0% 分别编码为 25/1000、5/1000、10/1000。
- [ ] SA-026：120,000 只作为资产推导值，不是 schedule 方法真值。
- [ ] SA-027：八臂 registry 的集合和顺序与任务书完全一致。
- [ ] SA-028：CURRENT_LOSS_U 是 HELD，未进入第一阶段。
- [ ] SA-029：所有 arm E1–E120 replay rate 为 0。
- [ ] SA-030：正式运行默认关闭。

### D.3 T/R1/R2 身份自审

- [ ] SA-040：T canonical 文件 path/bytes/SHA 均匹配。
- [ ] SA-041：T identity digest 算法有固定 golden test。
- [ ] SA-042：T identity digest 等于任务书冻结值。
- [ ] SA-043：T role 为 stress set，不是 validated selector。
- [ ] SA-044：T unique rate 恰为 canonical base 的 25/1000。
- [ ] SA-045：R1 从完整 eligible canonical base 做 global random，不人为排除 T。
- [ ] SA-046：R1 与 T 的自然重叠被报告，不冒充 zero-overlap comparator。
- [ ] SA-047：R2 与 T 交集为 0。
- [ ] SA-048：R2 label quota 精确匹配。
- [ ] SA-049：R2 historical dynamic bucket quota 精确匹配。
- [ ] SA-050：R2 OOF fold quota 精确匹配。
- [ ] SA-051：R2 oof_group_id quota 精确匹配。
- [ ] SA-052：oof_group_id 全部标记为 filename-bucket surrogate。
- [ ] SA-053：R2 任一 quota 不可行时 fail closed。
- [ ] SA-054：R2 没有 nearest/relaxed fallback。
- [ ] SA-055：R2 matcher 无法访问所有登记 terminal fields。
- [ ] SA-056：terminal field guard 有主动访问即抛错的测试，不只是输出列检查。
- [ ] SA-057：T/R1/R2 universe、排除行、选择行和 digest 全部保存。

### D.4 schedule 和曝光守恒自审

- [ ] SA-060：五个 identity groups 互斥且联合等于完整 pool。
- [ ] SA-061：分组算法不读取原 rank。
- [ ] SA-062：U 每五 epoch 每 ID 恰好一次。
- [ ] SA-063：F active 每五 epoch 每 ID 恰好两次。
- [ ] SA-064：U 每 ID 累计 multiplicity 为 16。
- [ ] SA-065：F 每 ID 累计 multiplicity 为 16。
- [ ] SA-066：U/F identity digest 相同。
- [ ] SA-067：U/F 累计 replay occurrence 相同。
- [ ] SA-068：U/F 每 ID multiplicity vector 逐项相同。
- [ ] SA-069：U/F 唯一差异是 epoch distribution。
- [ ] SA-070：F E161–E200 为 0。
- [ ] SA-071：T_TO_R2_AT_160 在 E121–E160 与 T_U 逐 epoch 相同。
- [ ] SA-072：T_TO_NR_AT_160 在 E121–E160 与 T_U 逐 epoch 相同。
- [ ] SA-073：fallback 后使用 R2-U，不被错误标为 no replay。
- [ ] SA-074：stop 后 exposure 减少被如实记录，不伪称 dose-matched。
- [ ] SA-075：同 schedule 的 treatment/comparator step-slot skeleton 完全相同。
- [ ] SA-076：计划和实际 occurrence、unique、repeat、multiplicity 全部守恒。

### D.5 common parent 和 lineage 自审

- [ ] SA-080：每 training seed 只生成一个 E1–E120 no-replay parent。
- [ ] SA-081：同 seed 八个 child 使用字节相同 parent checkpoint。
- [ ] SA-082：checkpoint 含 model/EMA/optimizer/scheduler/scaler 和全部 RNG。
- [ ] SA-083：checkpoint 含 epoch/global step/seed/lock/source/assets identity。
- [ ] SA-084：错误 parent SHA 的失败测试先红后绿。
- [ ] SA-085：错误 seed/source/asset/epoch 均拒绝。
- [ ] SA-086：child 启动不能只给裸 checkpoint path。
- [ ] SA-087：child 运行前后 parent SHA 不变。
- [ ] SA-088：logical E1–E120 全部指向 parent。
- [ ] SA-089：logical E121–E200 全部指向 child。
- [ ] SA-090：child 目录伪造 E1–E120 产物被拒绝。
- [ ] SA-091：parent/child lineage digest 可从原始字段重复计算。

### D.6 fixed-step replay 自审

- [ ] SA-100：base Dataset 长度不随 replay 改变。
- [ ] SA-101：所有 arm 每 epoch base batches 都是 938。
- [ ] SA-102：所有 arm 每 epoch optimizer steps 都与 NR 相同。
- [ ] SA-103：scheduler epoch transitions 与 NR 相同，且 replay forward 不推进 scheduler。
- [ ] SA-104：EMA updates 与 NR 相同。
- [ ] SA-105：warmup progress 与 NR 相同。
- [ ] SA-106：同 seed paired arms 的 base sample order digest 逐 epoch相同。
- [ ] SA-107：同 seed paired arms 的 base augmentation digest 逐 epoch相同。
- [ ] SA-108：每 base step 最多调用一次 optimizer_step。
- [ ] SA-109：replay backward 真实改变 parameter gradient。
- [ ] SA-110：replay CE 是逐样本 sum 除以 128。
- [ ] SA-111：base loss 仍是 upstream learner 的原定义。
- [ ] SA-112：replay microbatch 不超过实际 base batch 的 25%。
- [ ] SA-113：尾 batch cap 正确。
- [ ] SA-114：clip/unscale/scaler.step/scaler.update 各只发生一次。
- [ ] SA-115：replay forward 后 BN running buffers 逐字节恢复。
- [ ] SA-116：replay forward 后全局 RNG 逐字节恢复。
- [ ] SA-117：replay augmentation 使用独立 counter domain。
- [ ] SA-118：OOM 不减 batch、不减 replay、不拆 step、不继续训练。
- [ ] SA-119：隐式梯度累积被拒绝。
- [ ] SA-120：world_size 大于 1 在 phase 1 被拒绝。

### D.7 全量证据 schema 自审

- [ ] SA-130：每个 base/replay occurrence 都有一行。
- [ ] SA-131：每个 optimizer step 都有一行。
- [ ] SA-132：每个 epoch 都有 exposure summary。
- [ ] SA-133：occurrence 包含 identity/role/augmentation/logits/probability/CE/margin/correctness。
- [ ] SA-134：replay occurrence 包含累计次数和距上次曝光。
- [ ] SA-135：candidate signal 明确标为非 utility。
- [ ] SA-136：step ledger 包含 base/replay loss、grad、clip、LR、AMP、EMA、BN、RNG 和耗时。
- [ ] SA-137：epoch ledger 包含 planned/actual denominator/numerator/unique/repeat/cumulative/steps。
- [ ] SA-138：所有大表为 Zstd Parquet。
- [ ] SA-139：所有 Parquet 按 run/epoch 分区。
- [ ] SA-140：每个分区有 schema、row count、bytes 和 SHA。
- [ ] SA-141：JSON 只承担小型合同、identity、receipt 和 summary。
- [ ] SA-142：合法 null 都有 reason code。
- [ ] SA-143：没有空字符串、未登记 unknown 或同上式占位。
- [ ] SA-144：selection ledger 保存候选全集，不只保存 selected IDs。

### D.8 telemetry 自审

- [ ] SA-150：采样周期固定 1 秒并有 cadence 容差测试。
- [ ] SA-151：记录 process CPU/RSS/VMS。
- [ ] SA-152：记录 process read/write bytes 和 counts。
- [ ] SA-153：记录 system memory。
- [ ] SA-154：记录 GPU utilization/memory/temperature/power。
- [ ] SA-155：记录 CUDA allocated/reserved/max。
- [ ] SA-156：记录运行卷和 artifact 卷 free space。
- [ ] SA-157：provider 失败不填假 0。
- [ ] SA-158：关键 telemetry 全不可用时不能 canonical closeout。
- [ ] SA-159：telemetry partition 与 epoch receipt SHA 绑定。

### D.9 评价和统计自审

- [ ] SA-170：正式 endpoint 只接受 E200。
- [ ] SA-171：best.pt 在 CLI、config 和 runtime 三层均拒绝。
- [ ] SA-172：prediction manifest 与 sample-label digest 一致。
- [ ] SA-173：缺行、多行、重复行、错 label、错 split 均失败。
- [ ] SA-174：frontier 恰好包含 FN budget 0–95 的 96 行。
- [ ] SA-175：tie group 不可拆。
- [ ] SA-176：normalized AUC 公式有手算 golden fixture。
- [ ] SA-177：TN_at_FN95 正确。
- [ ] SA-178：FN_at_TN68253 正确。
- [ ] SA-179：两个阈值分别保存。
- [ ] SA-180：target TN 不可达时 fail-closed 语义正确。
- [ ] SA-181：val_op 不能选 method/checkpoint/stop/threshold。
- [ ] SA-182：test access 在所有入口失败。
- [ ] SA-183：8-seed discovery 和 14-seed confirmation schema 分离。
- [ ] SA-184：historical/discovery/confirmation seeds 去重 validator 存在。
- [ ] SA-185：paired missing member 导致比较失败。
- [ ] SA-186：exact paired sign-flip 有 exhaustive small-n golden test。
- [ ] SA-187：Holm 排序、阈值和 decision 有 golden test。
- [ ] SA-188：win rate、worst seed、dual-end degradation 全部报告。

### D.10 Q/R/A/D 和 disabled phase 2 自审

- [ ] SA-200：Q/R/A/D 只有 gate、stratum 或 factorial 语义。
- [ ] SA-201：任意 weighted total score 被拒绝。
- [ ] SA-202：confidence/loss/RHO/gradient/forgetting/AUM/coverage 不被写成 utility。
- [ ] SA-203：当前无 val_target 的事实进入 asset audit。
- [ ] SA-204：A enable 请求返回 BLOCKED_BY_VAL_TARGET。
- [ ] SA-205：A 阻断时不生成 arm/assignment/gradient artifact。
- [ ] SA-206：短分支 scaffold 默认 disabled。
- [ ] SA-207：phase 1 gate 未通过时 predictor training 被拒绝。
- [ ] SA-208：没有实现或启用强化学习 selector。
- [ ] SA-209：synthetic predictor fixture 不被标为科学证据。

### D.11 原子写、故障和恢复自审

- [ ] SA-220：每 epoch 从 inprogress generation 开始。
- [ ] SA-221：只有 schema/count/SHA/守恒全部通过才原子发布。
- [ ] SA-222：kill 故障注入会 quarantine partial。
- [ ] SA-223：OOM 故障注入会 quarantine partial。
- [ ] SA-224：disk-full 故障注入不会留下 canonical 半文件。
- [ ] SA-225：半写 Parquet 被检测。
- [ ] SA-226：半写 JSON 被检测。
- [ ] SA-227：损坏 receipt 被检测。
- [ ] SA-228：错误 generation 被拒绝。
- [ ] SA-229：错误 RNG/source/parent/asset 被拒绝。
- [ ] SA-230：resume 只从最后完整 epoch。
- [ ] SA-231：resume 不覆盖旧 generation。
- [ ] SA-232：关键 checkpoint E120/140/150/160/180/200 保留。
- [ ] SA-233：rolling checkpoint 策略可恢复且不删除关键 checkpoint。

### D.12 CLI 和副作用自审

- [ ] SA-240：所有 CLI 默认只读或 synthetic。
- [ ] SA-241：正式运行需要不存在的未来 release，因此当前必定拒绝。
- [ ] SA-242：validation PASS 不自动触发 runner。
- [ ] SA-243：runner 不生成 assignment。
- [ ] SA-244：runner 不生成 engineering gate。
- [ ] SA-245：runner 不生成 pilot release。
- [ ] SA-246：runner 不读取 blind/test。
- [ ] SA-247：runner 不扫描和复活旧 v2/v3 queue。
- [ ] SA-248：synthetic artifact 全部带 SYNTHETIC_NOT_SCIENTIFIC_RESULT。
- [ ] SA-249：formal side-effect flags 全为 false。
- [ ] SA-250：legacy_detected 与 active_v4_generated 字段分开。

### D.13 必跑测试和日志审计

- [ ] SA-260：每个行为变更均有失败优先测试。
- [ ] SA-261：每个修复保留 red 和 green receipt。
- [ ] SA-262：red test 的失败原因正是待实现行为，不是语法或 import 偶然错误。
- [ ] SA-263：green test 对应同一个 test ID。
- [ ] SA-264：<code>uv run pytest tests\stage1_sctsr_v4 -q</code> exit 0。
- [ ] SA-265：v4 核心合同测试无 skip/xfail。
- [ ] SA-266：<code>uv run pytest tests\stage1_dynamic_replay_v3 -q</code> 至少 231 passed。
- [ ] SA-267：contract CLI exit 0。
- [ ] SA-268：asset CLI exit 0。
- [ ] SA-269：schedule CLI exit 0。
- [ ] SA-270：synthetic canary exit 0。
- [ ] SA-271：validate_run exit 0。
- [ ] SA-272：closeout 只发布 IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION。
- [ ] SA-273：全部命令有原始 stdout/stderr、exit code、bytes 和 SHA。
- [ ] SA-274：测试日志没有被手工删节后冒充原始日志。
- [ ] SA-275：测试随机性有固定 seed，重复运行产物 digest 稳定。

### D.14 人工逐行复核

机器测试完成后，专家仍必须人工复核：

- [ ] SA-280：逐行检查 fixed_step_runtime 的 optimizer 调用边界。
- [ ] SA-281：逐行检查 replay loss reduction 和 denominator。
- [ ] SA-282：逐行检查 AMP/unscale/clip/step/update 顺序。
- [ ] SA-283：逐行检查 BN 保存恢复覆盖全部 BatchNorm buffers。
- [ ] SA-284：逐行检查 RNG fork 没有遗漏 Python/NumPy/Torch CPU/CUDA。
- [ ] SA-285：逐行检查 R2 白名单投影发生在 matcher 之前。
- [ ] SA-286：逐行检查任何异常路径都不会自动改变固定训练合同。
- [ ] SA-287：逐行检查 val_op/test 没有进入 selection/config generation。
- [ ] SA-288：逐行检查 completion PASS 没有触发 release。
- [ ] SA-289：逐行检查每个 public schema 与本任务书字段一致。

人工复核结果必须提供文件与行号，不能只列文件名。

### D.15 最终否定性证明

交付前必须运行一个独立 no-side-effect audit，机器检查并报告：

- [ ] SA-300：<code>formal_training_started</code> 为 JSON boolean <code>false</code>。
- [ ] SA-301：<code>engineering_gate_generated</code> 为 JSON boolean <code>false</code>。
- [ ] SA-302：<code>assignments_generated</code> 为 JSON boolean <code>false</code>。
- [ ] SA-303：<code>pilot_release_generated</code> 为 JSON boolean <code>false</code>。
- [ ] SA-304：<code>blind_holdout_opened</code> 为 JSON boolean <code>false</code>。
- [ ] SA-305：<code>selector_trained</code> 为 JSON boolean <code>false</code>。
- [ ] SA-306：<code>method_effectiveness_claimed</code> 为 JSON boolean <code>false</code>。
- [ ] SA-307：<code>val_target_available</code> 为 JSON boolean <code>false</code>，且 A 为 blocked。
- [ ] SA-308：旧历史 gate/release/assignment 只被检测和登记，没有被复活。
- [ ] SA-309：synthetic canary 没有进入 scientific result registry。

### D.16 自我审计最终判定规则

自我审计只能在以下条件全部满足时输出：

<code>SELF_AUDIT_PASS_IMPLEMENTATION_ONLY</code>

条件：

1. SA-001 至 SA-309 中所有适用实施项均为 PASS；
2. 允许阻断项有明确、已登记且与任务书一致的 reason；
3. 没有未解释 skip/xfail；
4. 没有缺失日志、exit code、bytes 或 SHA；
5. 没有正式训练或发布副作用；
6. v3 regression 不低于 231 passed；
7. v4 全套和 synthetic canary 全绿；
8. source tree 和所有证据产物身份可重算；
9. 人工逐行复核完成；
10. completion audit 明确不宣称方法有效。

任一适用项 FAIL 时，自我审计 overall 必须为 <code>FAIL</code>，并列出精确 check ID、字段、证据、剩余风险和修复要求。不得使用通过比例、代码行数、测试总数、文档长度或“主要功能已完成”来覆盖失败项。

专家自我审计不能替代 owner/第三方独立审查。自我审计 PASS 后的下一状态只能是：

<code>READY_FOR_INDEPENDENT_CODE_REVIEW_NOT_READY_FOR_FORMAL_TRAINING</code>
