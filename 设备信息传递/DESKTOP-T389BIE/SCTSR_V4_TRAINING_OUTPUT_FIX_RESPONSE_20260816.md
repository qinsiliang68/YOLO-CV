# SCTSR v4 训练与产物正确性修复回执

## 1. 回执身份与结论

- 接收规格：`设备信息传递/DESKTOP-T389BIE/SCTSR_V4_TRAINING_OUTPUT_FIX_SPEC_20260816.md`
- 规格审查基线：`54e320020c60c8d7a11f59d2fa606ff203fd0d4d`
- 修复分支：`codex/sctsr-v4-training-output-fixes`
- 本轮代码与 runbook 冻结提交：`f5c9801555285e08969a2c7040be394b6bb0a0e6`
- 修复方式：在隔离 clean worktree 内按 RED→GREEN 施工；原始脏工作树和历史训练产物未改动。
- 当前实现判定：`IMPLEMENTED_AND_LOCALLY_VERIFIED_FOR_REREVIEW`
- 当前训练判定：`NO_GO_PENDING_TRAINING_MACHINE_REREVIEW`

这两个判定不能互相替代。前者表示 TC-01 至 TC-10 已落实到实现、负向测试和本地工程 canary；后者表示开发机不能自行替训练机签发正式训练授权。只有训练机重新拉取本分支、复验本回执并由 owner 签发正式 seeds、release、一次性 job token 和共享 claim registry 后，才可改变训练判定。

本轮没有声称 SCTSR、T、R2 或任何 Q/R/A/D 信号有效；所有 canary 都是机械验证，不是科学结果。

## 2. 最小回滚提交链

| 顺序 | Commit | 作用 |
| --- | --- | --- |
| 1 | `6419307` | T/R2 同时按 sample ID 与 image SHA-256 排斥，重新冻结 R2 |
| 2 | `6b49a1d` | loader 与 endpoint 逐图输入字节绑定 |
| 3 | `09af540` | endpoint/index/completion 在 latest fence 事务内终结，写 terminal heartbeat |
| 4 | `69f4f72` | 当前 runtime、adapter import origin 和二分类 head 在训练前复验 |
| 5 | `cef0e03` | logical job 按 experiment/release namespace 隔离，output root 不再拆科研身份 |
| 6 | `752d326` | runbook 与证据文本 LF 身份冻结 |
| 7 | `7b44d7a` | 强制分类视图与 canonical 图片为同一硬链接实体，终端重新扫描整棵 loader 树 |
| 8 | `4748ce9` | 分离 canonical 数据根与 Ultralytics classification view 数据根 |
| 9 | `f5c9801` | 将分离数据根、硬链接约束和 13×3090 运维口径写入 runbook v5 |

提交没有压扁，任何一层都可单独回退和复查。

## 3. TC-01 至 TC-10 修复映射

### TC-01：T 与 R2 的字节级重叠

实现：

- `stage1_sctsr_v4/random_controls.py:134` 的 `build_minimum_oof_group_displacement_selection()` 和 `:256` 的 `build_r2_matched_random()` 在初选与补位阶段都排除 T 的 image SHA。
- `stage1_sctsr_v4/identity_pool.py` 在正式 R2 校验中要求 content map 和 T content set。
- `stage1_sctsr_v4/r2_addendum.py:106` 的 `validate_approved_r2_build()` 同时验证 identity digest、content digest、唯一图像数与字节交集。

关键测试：

- `test_random_controls.py::test_r2_excludes_different_id_with_the_same_image_bytes_as_t`
- `test_identity_pool.py::test_r2_pool_validation_rejects_content_overlap_with_t`
- `test_r2_addendum.py::test_registered_addendum_materializes_the_audited_3000_id_pool`

重新冻结的 R2 事实：

```text
unique sample IDs:             3000
unique image SHA-256:          3000
T/R2 sample-ID overlap:        0
T/R2 image-SHA overlap:        0
identity digest:               A6DAA20A70F02B30D15B7C3E4079EA86903051AEED264F53E0A104A4C1AA80B6
content-map digest:            751742DA58F13A2678CB700D018B95B5FA8E38392F15366B688CCEE97BE81CA7
selected-content digest:       A48B721CA37AD66D65B8C5972C5AE66C328C09194BA3C8C22C19B8FECE40F819
excluded-content digest:       F0F5F857F684E214BA252360B9454E46FBCE0D9A361910E03F3390F1FCEC42B7
oof_group displacement count:  379
oof_group total variation:     0.12633333333333333
```

三条 R2 arms 继续使用同一个冻结 pool；没有为某一治疗臂单独放宽条件。

### TC-02：endpoint 未绑定实际输入字节

实现：

- `stage1_sctsr_v4/prediction_runtime.py:172` 构建 `EndpointInputBinding`，绑定 dataset root、content ledger、split bundle、逐图 bytes/SHA 和聚合 digest。
- `stage1_sctsr_v4/prediction_runtime.py:241` 每次使用前重新验证输入 binding。
- `stage1_sctsr_v4/prediction_runtime.py:381` 只允许完整 binding 相同的 endpoint reuse；不一致的旧 endpoint 被隔离。
- `stage1_sctsr_v4/run_validation.py:168` closeout 重新校验 endpoint 输入与输出证据。

关键测试：

- `test_prediction_runtime.py::test_endpoint_input_binding_rejects_changed_image_bytes_and_binds_dataset_root`
- `test_prediction_runtime.py::test_formal_endpoint_publisher_runs_real_images_and_writes_complete_evidence`
- `test_prediction_evaluation_hardening.py::test_formal_closeout_rejects_missing_or_mutated_endpoint`

示例回执位于 `08_reports/sctsr_v4_training_output_fixes_20260816/commands/endpoint_input_binding_example.log`。它记录完整 binding，并证明绑定后修改图片会以 `DATASET_CONTENT_MISMATCH` 被拒绝。该示例只使用 synthetic schema，未访问正式 val_op/test。

### TC-03：loader staging 的 TOCTOU

实现：

- `stage1_sctsr_v4/dataset_adapter.py:319` 生成 `MATERIALIZED_DATASET_BINDING v3`，逐行记录 canonical path、loader physical path、bytes、SHA 和 `samefile_as_canonical`。
- `stage1_sctsr_v4/dataset_adapter.py:473` 在 step 0、resume、endpoint、completion 边界重新扫描物理树、重新计算内容、重新验证同一文件实体。
- 分类视图内额外 class、额外文件、缺失文件、symlink/reparse 漂移、同名换 inode、同路径换 bytes 都失败封闭。
- `stage1_sctsr_v4/formal_cli.py:724` 从冻结 asset manifest 解析 canonical dataset root；`trainer_overrides.data` 只表示单独的 classification view。两者不再混用。

关键测试：

- `test_dataset_adapter.py::test_materialized_binding_detects_post_setup_byte_replacement`
- `test_dataset_adapter.py::test_materialized_binding_rejects_unregistered_extra_file`
- `test_dataset_adapter.py::test_materialized_binding_requires_hardlink_to_canonical_source`
- `test_dataset_adapter.py::test_materialized_binding_rejects_extra_file_added_after_setup`
- `test_dataset_adapter.py::test_materialized_binding_accepts_separate_hardlink_only_classification_view`

本项有两组额外 RED→GREEN：

```text
752d326: 2 failed, 9 passed  -> 7b44d7a: 11 passed
7b44d7a: 3 failed, 9 passed -> 4748ce9: 12 passed
```

还执行了真实 Sewer-ML 两图硬链接 canary：canonical root 与 classification view 分离，两行 `os.path.samefile=true`；预检后新增文件被 `DATASET_CONTENT_MISMATCH` 拒绝。证据在 `commands/real_hardlink_binding_canary.log`。

训练机必须注意：不要复制图片建立 classification view，必须创建硬链接。若数据盘/文件系统不支持跨卷硬链接，应把 view 建在 canonical 数据同一卷内；不能退回复制文件。

### TC-04：终结产物位于 latest fence 事务之外

实现：

- `stage1_sctsr_v4/formal_execution.py:1197` 的 `execute_fenced_finalization()` 在 logical-job control lock 下保持 `FINALIZING` fence，覆盖 endpoint、receipt、manifest、index、completion 和 terminal heartbeat。
- common parent 与 branch runner 都调用同一终结入口，不再自行拼接完成流程。
- 旧 generation 被 supersede 后，所有 canonical 写入都返回 `LOGICAL_JOB_FENCED`。

关键测试：

- `test_formal_execution_claim.py::test_fenced_finalization_publishes_terminal_complete_and_blocks_resume`
- `test_formal_execution_claim.py::test_resume_requires_stale_prior_lease_and_fences_old_attempt`
- `test_formal_completion_transaction.py::test_branch_cannot_complete_before_endpoint_receipt_and_final_index`
- `test_formal_completion_transaction.py::test_partial_endpoint_is_quarantined_before_finalization_retry`

### TC-05：heartbeat 没有 FAILED/COMPLETE

实现：

- `execute_fenced_finalization()` 成功时写经 terminal receipt 绑定的 `COMPLETE`；异常时原子写 `FAILED` 后重新抛出异常。
- `stage1_sctsr_v4/formal_execution.py:1252` 的 `mark_execution_failed()` 处理训练期异常，且对相同 fence 幂等。
- `FAILED` 可由新的签名 RESUME token 立即接管；`COMPLETE` 永久拒绝 START/RESUME；旧 generation 不能覆盖新 heartbeat。

关键测试：

- `test_formal_execution_claim.py::test_failed_finalization_marks_failed_and_allows_immediate_resume`
- `test_formal_execution_claim.py::test_training_exception_can_mark_active_attempt_failed_idempotently`
- `test_formal_execution_claim.py::test_fenced_finalization_publishes_terminal_complete_and_blocks_resume`

实例证据位于 `commands/heartbeat_examples.log`，包含 `FAILED`、`COMPLETE` 两个 terminal heartbeat，以及 generation 1 在 generation 2 后被 `LOGICAL_JOB_FENCED` 拒绝的结果。

### TC-06：改变 output root 会拆成两个 logical job

实现：

- `stage1_sctsr_v4/formal_execution.py:241` 的 logical-job digest 不再包含 storage root。
- logical key 包含 experiment/release namespace、run role、logical run ID、arm、seed、parent、lineage 与 schedule。
- `stage1_sctsr_v4/formal_execution.py:741` 首个 START 固定 authorized output-root digest；同科研身份换目录不能再次 START。
- registry/token/fence/heartbeat 升级到 v2 namespace，旧 v1 与 v2 不混用。

关键测试：

- `test_formal_execution_claim.py::test_logical_job_identity_is_independent_of_storage_root_but_first_start_binds_storage`
- `test_formal_execution_claim.py::test_logical_job_identity_is_namespaced_by_experiment_and_release`
- `test_formal_execution_claim.py::test_distinct_tokens_for_same_logical_job_cannot_both_claim`

本轮按用户要求没有增加 hostname/GPU UUID 级复杂锁；同一 logical job 的唯一性由共享 v2 registry 和 fence chain 保证，机器分配仍可随机进行。

### TC-07：source manifest 没有现场重探测 runtime

实现：

- `stage1_sctsr_v4/source_identity.py:37` 的 `probe_runtime_environment()` 每次读取当前 Python、Torch、CUDA、GPU、driver 和注册依赖身份。
- `stage1_sctsr_v4/source_identity.py:304` 的 `validate_source_tree_manifest()` 现场重探测并字段级比较，不能只重新 hash manifest 旧值。
- authorization、resume 与 closeout 都重新触发该检查。
- uv isolated 临时父目录被规范化，不把同内容的随机环境路径误判为漂移。

关键测试：

- `test_source_identity.py::test_source_manifest_validation_reprobes_live_runtime_and_rejects_drift`
- `test_source_identity.py::test_source_identity_ignores_ephemeral_interpreter_parent_directory`
- `test_source_identity.py::test_clean_source_manifest_revalidates_current_git_head_and_worktree`

### TC-08：SCTSR trainer adapter import origin 未验证

实现：

- `stage1_sctsr_v4/training_system.py:31` 将 `integrations/ultralytics/sctsr_classification_trainer.py` 纳入 upstream binding。
- trainer setup 后检查 `module.__file__` 的 resolved path、bytes 与 SHA，要求严格等于当前 repository root 下冻结 adapter。
- 预加载的同名假模块、shadow path 和单字节修改都会在构造训练器前失败。

关键测试：

- `test_training_system.py::test_sctsr_adapter_import_origin_is_exactly_bound`
- `test_training_system.py::test_upstream_binding_records_git_and_file_hashes`

### TC-09：二分类映射与两输出 head 校验过晚

实现：

- `stage1_sctsr_v4/formal_cli.py:756` 的 `validate_binary_classification_contract()` 在任何 optimizer step 前同时检查：
  - train/val 直接 class 目录恰为 `no_target`、`target_defect`；
  - dataset `classes` 和 `class_to_idx`；
  - model names 与 `nc=2`；
  - 分类 head 输出维度为 2；
  - loader label 仅为 0/1 且两类均非空。
- binary-contract digest 写入 prepared trainer binding，并在后续正式身份中重验。

关键测试：

- `test_binary_classification_contract.py::test_binary_contract_accepts_exact_two_class_training_system`
- `test_binary_classification_contract.py::test_binary_contract_rejects_every_two_class_mismatch`

参数化负例覆盖额外 class、交换 class index、错误 model names、三输出 head 与非法 label。

### TC-10：证据文本行尾不稳定

实现：

- 根 `.gitattributes` 固定 `.py/.json/.md/.yaml` 与本轮 report 的 `.log/.txt/.csv` 为 LF。
- `test_portable_text_identity.py::test_every_registered_evidence_text_has_portable_checkout_identity` 遍历登记证据并检查 Git attribute 与 checkout bytes。
- 本回执的 manifest 从最终 LF checkout bytes 生成，不使用手改 hash。

## 4. Runbook v5 身份

旧 `RUNBOOK_MANIFEST_v4.json` 是不可变历史证据，没有覆盖。当前使用新建的 v5：

```text
path:           docs/stage1_sctsr_v4/RUNBOOK_MANIFEST_v5.json
document count: 22
total bytes:    226572
manifest bytes: 4195
manifest SHA:   CBCCAF21637AE2FE7377C1C0F18369F3E1A1DE86214E4926F1FC6298ECA7D15E
runbook digest: 1A07C339C4E573C311EF1E06B83117988E86D7A92FEE297A7140BEDA61E378C5
validator:      PASS
```

runbook 已明确：

- 13 台 RTX 3090 中 12 台 active、1 台 spare；
- parent 阶段受 seed 数量限制，不能为了占满 GPU 重复跑同一 logical job；
- canonical root 与 classification view 分开；
- classification view 只能使用同卷硬链接；
- 失败封闭、resume、endpoint、E200/EMA/val_op 与正式副作用边界。

## 5. RED→GREEN 证据

通用 TC-01 至 TC-10 合同探针在原审查基线 `54e3200` 上运行：

```text
exit code: 1
result:    10 failed in 1.47s
log SHA:   8B8B9C24A579C5FD60E719F2A311FBD3E0F3932D37475438F1F738BE9BFAE7B2
```

相同探针在 `f5c9801` 上运行：

```text
exit code: 0
result:    10 passed in 1.39s
log SHA:   215F6B952E44802D4FBCB41194F925BD1527B158121EB823AB673018860D7B65
```

TC-03 的两个追加问题还保留了独立失败优先证据：

```text
hardlink/rescan RED:  2 failed, 9 passed; SHA DBF76EFE7EA13806DED821799DD4BA3E4F595EB3E64A42F430420315EE4B5B1B
hardlink/rescan GREEN: 11 passed;          SHA 003136FC9684A692DC79E14F2B791756B643259F2B49FD9631F08CB6BAFC3DD8
separate roots RED:   3 failed, 9 passed; SHA 93D20769BB4BE6733A21F08A0128EFC117D87143AAC3E79B633B86662AF37761
separate roots GREEN: 12 passed;           SHA 96A64FE10A4D766DB7FF3F34D7B58F11C3C77FA5A22820A7BFBC14DC18F96E1F
```

## 6. 最终 clean-checkout 动态验证

验证 checkout：`C:/Users/28898/AppData/Local/Temp/YOLO-CV-sctsr-final-verify-20260816-2`。运行前显式 provision 注册权重：

```text
file:   yolo11l-cls.pt
bytes:  28,553,700
SHA256: 6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C
```

最终命令结果：

| 验证 | 结果 | 日志 SHA-256 |
| --- | --- | --- |
| `uv lock --check` | PASS | `E52C3D1A1C212C31B6E722E674F795D9E5EE75E5C6B8B254CE530A96FF3FA6D9` |
| Python 3.11 v4 | 456 passed / 0 failed | `44F65E3341D4263F802572AD2905FF33AA6CE64798FBA5B398136A78FD81E17E` |
| Python 3.12 v4 | 456 passed / 0 failed | `5CC57A2D1B84ECEE9037E3E4FCC680FEB211426851F438220E3CA6C170D4109E` |
| Python 3.11 compileall | PASS | `814114AFF75DF6C02E12F182B4A682F39505866C336887B6F7FA21AA909422D6` |
| Python 3.12 compileall | PASS | `4425DBCA4DCDAA571F69A4A03459FD482930F831355AF6943A974A8933D47DDC` |
| v3 regression | 181 passed, 3 skipped | `BFBA103DFD55FC7386D3FF53BF889D7EA4CC27114C21E0479A0E03811503CB04` |
| runbook v5 validator | PASS / 22 docs | `E3C9DD93DCEEE1AE853E00F36202CD0FFFCF2821A2DAC2FB41EA62A52D32168E` |
| `git diff --check` | PASS / tracked clean | `5C730A1D24E5FDADF35055297E05011C121E83251FEAE70089CFC9FC3C53AF46` |

## 7. 真实数据工程 canary

在真实本地 Sewer-ML 图片、真实注册 `yolo11l-cls.pt` 和 CUDA 上做了一个有界工程 canary。开发机 GPU 是 RTX 4060，因此结果只证明代码链路，不外推 RTX 3090 吞吐或方法效果。

已验证：

- 4 张真实图片逐图 bytes/SHA；
- 真实 forward；
- base backward 与 replay backward；
- replay 后 RNG 和 BatchNorm running buffer 恢复；
- 恰好 1 次 optimizer step 与 1 次 EMA update；
- Zstd Parquet occurrence/prediction/frontier；
- checkpoint save/reload；
- 96 个 tie-safe frontier 点；
- partial generation quarantine；
- 损坏 checkpoint 被拒绝；
- canary manifest 15/15 文件重新读取匹配。

主要产物身份：

```text
checkpoint bytes:       103,328,511
checkpoint SHA:         E64E016C976F332815AF118A4EF9FF305918A4E30D154DF9455BB28D7514355D
prediction SHA:         64DAA9D130A8950CC5D080971340E920FA64BA3C23E38F3B6133462D419300EC
frontier SHA:           430F4C58CAEC0572824EFC780A560A9D7D44CD5257F201B0DCD8999051FEAC9E
generation digest:      901D5122CA26B59F6ECA54B27E45E6E247ACDCF4830FA88765BF5B61F5778C86
receipt-chain digest:   C2831D6F9A8A3101A54330D0D1D6BE63BF6D555005336E71024F85104C8AAA59
canary report SHA:      514F10298A9703791680227756E07EC9695C51FBA79BAA2FE2FCF8AB55646C84
canary receipt SHA:     BC9412BAAB150E3D41933F09C4751E45C818A50E82F62C76BE50E90A5E538257
command log SHA:        C43934375A71F4C63DFA7D147043FCA5E09D6746CEC1FA5B7B32A827C651AB3E
```

103 MB checkpoint 没有提交进 Git；其 bytes/SHA 和原始 manifest 已绑定，完整 15 文件集合在本机重新读取通过。小型 Parquet、JSON receipt、index、summary 与 manifest 进入审查证据目录。

## 8. 训练机重新审查与部署顺序

训练机不得从旧 dynamic worker 或旧 296-job queue 启动。重新审查应按以下顺序：

1. 拉取 `codex/sctsr-v4-training-output-fixes` 并核对本回执最终 commit。
2. 在 clean checkout 显式放置并校验注册权重 bytes/SHA。
3. 运行 `uv lock --check`、Python 3.11/3.12 v4 suites、compileall、v3 regression。
4. 在每台 3090 机器生成同卷 hardlink-only classification view，验证 `MATERIALIZED_DATASET_BINDING v3`；禁止复制图片代替硬链接。
5. 现场重新探测 Python/Torch/CUDA/GPU/driver/import origin。
6. 训练机审查 TC-01 至 TC-10 与证据 manifest；如通过，另行签发正式 seed registry、owner-signed release、每个 logical job 的一次性 token 和共享 v2 claim registry。
7. 先跑 common parent E1–E120，再从同一 E120 完整 checkpoint 分八臂。不能用旧 `dynamic_campaign_train_worker.py --job-id`。
8. 所有正式 child 固定 E200、EMA、val_op；禁止 `best.pt`，禁止 test/blind。

机器可随机分配 logical jobs；不需要新增复杂 GPU 锁。但所有机器必须访问同一个共享 claim registry，且同一 logical job 只能存在一个 latest fence chain。

## 9. 明确的剩余边界

- A/gradient-alignment 仍因缺少独立冻结 `val_target` 而 `BLOCKED_BY_VAL_TARGET`；这不阻止第一阶段 timing/stop/fallback 八臂，但禁止开启 A 正式配置。
- 尚未签发正式 seeds、release、one-use job tokens 与共享 v2 registry；这些是训练授权输入，不在开发机代码修复中伪造。
- 本轮没有在 3090 上估计正式总工期；RTX 4060 单步 canary 不能用来替代 3090 全量 benchmark。
- 本轮没有启动 200 epoch、没有跑任何正式 seed、没有生成 assignment/gate/pilot，也没有访问 val_op/test 数据内容。

## 10. 副作用与科学声明

```text
formal_training_authorized=false
formal_training_started=false
engineering_gate_generated=false
assignments_generated=false
pilot_release_generated=false
blind_holdout_opened=false
test_accessed=false
method_effectiveness_claimed=false
```

本回执请求训练机做独立复审；在复审通过及 owner 正式签发前，保持：

```text
sctsr_current_training_output_verdict=NO_GO_PENDING_TRAINING_MACHINE_REREVIEW
```

## 11. 证据入口

主证据目录：

`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_training_output_fixes_20260816/`

优先读取：

- `VERIFICATION_SUMMARY.json`
- `COMMAND_INDEX.json`
- `CANARY_RECHECK.json`
- `EVIDENCE_MANIFEST.json`
- `commands/*.log`
- `canary/ENGINEERING_CANARY_REPORT.json`

`EVIDENCE_MANIFEST.json` 是最终文件级 bytes/SHA 索引；任何条目不匹配都不能把本回执判为通过。
