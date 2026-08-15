# SCTSR v4 训练与产物正确性修复规格

## 交付身份与当前结论

- 审查对象：`qinsiliang68/YOLO-CV`
- 精确审查提交：`54e320020c60c8d7a11f59d2fa606ff203fd0d4d`
- 前置 R2 提交：`5ce6fa726f6d2f6200aae33b99593efd778c18d3`
- 审查设备：`DESKTOP-T389BIE`
- 日期：`2026-08-16`（Asia/Shanghai）
- 当前结论：机械训练路径可以执行单步 CUDA 更新，但正式训练与正式产物仍为 **NO-GO**。

本文件只处理会改变训练输入、梯度过程、checkpoint、预测、完成状态或恢复正确性的缺陷。机器分配策略不在本轮验收范围内。不要只改报告状态、放宽校验或补一条说明来关闭问题；每项必须有失败复现、代码修复和回归测试。

## 合并前硬门槛

修复提交只有同时满足以下条件才可请求重新审查：

1. 下列 TC-01 至 TC-09 全部有先失败、后通过的负向测试；TC-10 和 runbook 身份问题也一并关闭。
2. T 与 R2 的实际图片 SHA-256 交集为 0，不只是 sample ID 交集为 0。
3. endpoint receipt 能独立证明实际 dataset root、内容账本和逐图输入字节。
4. 旧 execution attempt 被新 fence 取代后，不能再写 endpoint、最终 index、最终 receipt 或 completion marker。
5. 失败 attempt 原子进入 `FAILED`，成功 attempt 与 canonical completion 同一事务进入 `COMPLETE`。
6. 同一个逻辑科研任务即使换 output root，也只能存在一个 logical-job fence 链。
7. 正式授权与 closeout 都重新探测当前 Python/Torch/CUDA/GPU/driver/import origin，不能只校验旧 manifest 内自带的值。
8. epoch 1 前证明 train/val 都严格是 `{0:no_target, 1:target_defect}`，模型输出头严格为 2。
9. 干净 checkout 在显式提供并校验注册权重后，可复现完整测试；不能依赖某台机器未跟踪的偶然文件。

## 建议提交拆分

请按依赖拆成 5 个可审查提交，不要把所有变更压成一个大提交：

1. `fix(sctsr-v4): enforce content-disjoint T and R2`
2. `fix(sctsr-v4): bind loader and endpoint input bytes`
3. `fix(sctsr-v4): fence finalization and terminal lease state`
4. `fix(sctsr-v4): verify live runtime adapter and binary head`
5. `docs(sctsr-v4): refreeze runbook and portable evidence`

每个提交应附对应的 RED/GREEN 命令和输出；不要在同一个提交里先改测试以适配旧错误行为，再把该行为宣布为正确。

## TC-01 — T 与已批准 R2 存在 1 个字节级重叠

### 已复现事实

```text
T:  3000 IDs / 3000 unique image SHA
R2: 3000 IDs / 3000 unique image SHA
sample-ID overlap: 0
image-SHA overlap: 1

SHA-256:
92DCCBF1AE74191DB0C0CB6B8B5681A321460A0C9F08C16D5239AA68990CFA86

T:  Det/images/normal_train/00175370.png
R2: Det/images/normal_train/00859781.png
```

当前 `random_controls.py` 只按 sample ID 排除 T；同字节别名仍可进入 R2。当前批准的 R2 digest `957346D5...94B` 因而不能继续沿用。

### 必须修改

- `stage1_sctsr_v4/random_controls.py`
  - 修改 `build_minimum_oof_group_displacement_selection()` 和 `build_r2_matched_random()`，显式接收冻结 content ledger 映射 `sample_id -> image_sha256`。
  - 构造 `t_image_sha256_set`，候选必须同时满足 `sample_id not in T` 和 `image_sha256 not in t_image_sha256_set`。
  - counter-hash 补位也必须经过相同内容排除，不能只在第一轮筛选时排除。
  - `PoolBuildAudit` 新增并持久化：`t_unique_image_sha256_count`、`selected_unique_image_sha256_count`、`overlap_with_t_image_sha256_count`、`selected_content_digest`。
- `stage1_sctsr_v4/r2_addendum.py`
  - `validate_approved_r2_build()` 必须要求 `overlap_with_t_image_sha256_count == 0`。
  - 校验 `selected_content_digest` 与冻结批准值，而不仅是 identity digest。
- `stage1_sctsr_v4/identity_pool.py`
  - `IdentityPool.validate()` 增加可选 `content_sha_by_id` 与 `t_content_sha256` 参数；R2 正式路径缺少这些输入时直接失败。
- 重新生成 R2 pool、selection/quota/displacement 账本、schedule、asset registry、source/runbook/evidence manifests，并更新 `R2_APPROVED_IDENTITY_DIGEST`。不能手工只改常量。

### 必须新增的测试

- 在 `tests/stage1_sctsr_v4/test_random_controls.py` 构造不同 sample ID、相同 image SHA 的别名；断言别名永远不进入 R2，且确定性补位后仍为 3000 条。
- 在 `tests/stage1_sctsr_v4/test_r2_addendum.py` 把 `overlap_with_t_image_sha256_count` 改为 1，必须以 `R2_OVERLAPS_T` 失败。
- 增加真实冻结资产测试：重建 T/R2 后断言 sample ID 交集和 image SHA 交集都为 0，并断言新 identity/content digest 与冻结值一致。
- 继续断言 label/dynamic/fold 精确配额、最小 oof_group 位移规则、3000 唯一条目和三个 R2 arms 共用同一 pool。

## TC-02 — Endpoint receipt 没有绑定实际输入字节和 dataset root

### 当前错误链

- `prediction_runtime.py:58-148` 的 `load_registered_image_records()` 只校验注册路径、标签、包含关系和文件存在，不对实际图片重算内容 SHA。
- `PredictionArtifactBinding` 未记录 dataset root、content-ledger digest 或 endpoint input digest。
- `prediction_runtime.py:200-238` 的 `build_formal_endpoint_receipt()` 只覆盖输出文件。
- `prediction_runtime.py:241-304` 的 endpoint reuse 不接收当前输入绑定；旧 endpoint 校验通过后，代码会把调用方当前 dataset root 填进返回值，造成错误归因。

### 必须修改

- `RegisteredImageRecord` 增加 `image_bytes`、`image_sha256` 和 canonical content-ledger row digest。
- `load_registered_image_records()` 必须加载注册 content ledger，对每个实际 `image_path` 执行 byte count 和 SHA-256 校验；任何同路径不同字节必须在 inference 前失败。
- 建立一个明确结构 `EndpointInputBinding`，至少包含：
  - 规范化、已解析的 dataset root 及其 digest；
  - content ledger 文件 SHA-256 与 ledger digest；
  - effective `val_op` split bundle SHA-256；
  - 排序后的 `(sample_id, y_true, image_bytes, image_sha256)` 聚合 digest；
  - 输入行数与总字节数。
- 将该 binding 写入并交叉校验：
  - `PredictionArtifactBinding`；
  - `prediction_summary.json`；
  - `FORMAL_ENDPOINT_RECEIPT.json`；
  - `RUN_MANIFEST.json` 或最终 closeout receipt；
  - `run_validation.validate_formal_endpoint_evidence()`。
- `prepare_formal_endpoint_publication()` 必须接收 `expected_input_binding`。只有旧 endpoint receipt 中的完整输入绑定与当前重新计算值完全一致时才允许 `REUSE_VALID_ENDPOINT`；否则隔离旧 endpoint 并重建。
- 不要把绝对路径字符串当作唯一内容证明。路径 digest 与内容 digest必须同时存在。

### 必须新增的测试

- 建两个目录结构和 sample ID 完全相同、其中一张图片字节不同的数据根；旧 endpoint 在第二个根下必须拒绝 reuse。
- endpoint 建好后修改一张输入图，`validate_formal_endpoint_evidence()` 和 closeout 都必须失败。
- 同字节复制到另一个批准根时，测试应明确项目政策：若 root 也必须固定则拒绝；若允许迁移，则必须由签名迁移 receipt 授权，不能静默接受。
- 破坏 content ledger SHA、split bundle SHA、aggregate input digest 中任一项，必须失败。

## TC-03 — Loader 实际 staging 在 preflight 后存在 TOCTOU

### 当前错误链

- `formal_cli.py:690-729` 在 trainer setup 后调用 `validate_materialized_dataset_bytes()`，这一刻能证明 train/val 文件正确。
- `dataset_adapter.py:317-375` 返回的只有聚合摘要，没有保留每个 loader 实际物理路径供 closeout 重验。
- 训练期间和最终完成前没有重新枚举、重算 loader 实际路径；closeout 主要重验 canonical dataset root，不等于重验训练真正消费的 staging。

### 必须修改

- 生成 canonical `MATERIALIZED_DATASET_BINDING`（建议 Parquet 行表 + JSON summary），每行至少包含：
  - role、sample ID、label；
  - loader 实际 resolved path；
  - canonical source resolved path；
  - bytes、SHA-256；
  - Windows file identity（volume serial/file ID，或可证明硬链接相同的等价字段）；
  - `os.path.samefile(loader_path, canonical_path)` 结果（要求硬链接时必须为 true）。
- `validate_prepared_trainer_datasets()` 返回并持久化该 manifest 的 path/SHA/digest，写入 `prepared_trainer_binding` 和 formal input snapshot。
- 在以下边界调用同一个 revalidator：
  - 第一个 optimizer step 前；
  - RESUME 恢复完成、重新进入 epoch 前；
  - endpoint inference 前；
  - canonical completion 前。
- revalidator 必须重新从 `trainer.train_loader.dataset` / `trainer.test_loader.dataset` 枚举实际路径，不能只读旧 manifest 自己。
- staging root 建成后设为不可写，并拒绝额外 class 目录、额外文件、缺失文件、符号链接或 junction 漂移。只设置只读属性不是内容证明，终端完整 rehash 仍必须执行。

### 必须新增的测试

- preflight 后、epoch 1 前修改 staging 文件，训练必须在 optimizer step 0 前失败。
- preflight 后替换 staging 文件 inode/file ID 但保持同名同大小，必须失败。
- resume 前替换一个 hardlink，必须失败。
- E200 后、completion 前修改 staging 文件，必须阻止 completion marker。
- 加一个多余 class 目录或多余图片，必须在 epoch 1 前失败。

## TC-04 — Endpoint、最终 index 和 completion 位于 latest-fence 事务之外

### 当前错误链

- epoch publication 已使用 `execution_fence_guard()`。
- common-parent 在 `formal_training.py:1078-1107` 写 `PARENT_RECEIPT`、最终 manifests/index 和 completion 时没有持有该 guard。
- branch 在 `scripts/stage1_sctsr_v4/run_branch.py:278-312` 执行 endpoint、更新 branch receipt、重建 index 和发布 completion 时也没有持有该 guard。
- 已复现：generation 1 被 generation 2 fence 后，旧进程仍可发布 `FORMAL_PARENT_COMPLETE`。

### 必须修改

- 增加一个唯一正式终结入口，例如 `finalize_under_execution_fence()`；其锁范围必须覆盖：
  - 再次验证 current fence；
  - endpoint 生成或合法 reuse；
  - pending run receipt 更新；
  - `RUN_MANIFEST.json`；
  - generation/logical/exhaustive artifact indexes；
  - `FORMAL_COMPLETION_RECEIPT.json`；
  - heartbeat `COMPLETE` 转换。
- common-parent 和 branch 都必须走该入口。不要让调用者自行按多个函数拼装正式完成流程。
- 所有 canonical 写入前必须校验相同的 `execution_id`、`logical_job_digest`、`fence_generation`、`fence_digest`。
- 已被 supersede 的 attempt 只能写入自己的 quarantine/failure evidence，不能触碰 canonical root 的 final artifacts。
- 若 endpoint inference 时间较长，应在持有 logical-job control lock 的终结事务内运行，或设计等价的不可抢占 finalizing fence；不能只在 endpoint 前检查一次、结束后再检查一次而允许中间被接管。

### 必须新增的测试

- generation 1 完成最后 epoch 后暂停；generation 2 RESUME claim；恢复 generation 1。断言 generation 1 对 endpoint、manifest、index、completion 的每次写入都返回 `LOGICAL_JOB_FENCED`。
- 在 endpoint、index、completion 三个阶段分别故障注入并 supersede，旧 attempt 均不能发布 canonical 文件。
- 成功路径断言 completion receipt 与 terminal heartbeat 在同一 fence generation 下。

## TC-05 — Heartbeat 只写 ACTIVE，从不进入 FAILED/COMPLETE

### 当前错误链

`formal_execution.py:343-382` 能读取三种状态，但生产写入只有 claim 时的 `ACTIVE`（约 768-776）和 renew 时的 `ACTIVE`（约 958-971）。CLI failure receipt 不更新 lease；成功 completion 也不更新 lease。

### 必须修改

- 新增单一函数 `transition_execution_lease(binding, expected_job_bindings, terminal_status, terminal_receipt)`：
  - 只允许 `FAILED` 或 `COMPLETE`；
  - 在 logical-job control lock 内验证 current fence；
  - 绑定 terminal receipt path、bytes、SHA-256、status、execution ID 和 fence generation；
  - 原子替换 heartbeat 并 fsync；
  - terminal 状态不可从旧 generation 覆盖新 generation。
- 成功路径：先在同一 fence 事务生成并验证 canonical completion，再写 `COMPLETE` heartbeat，然后释放锁。
- 失败路径：只有当前 attempt 仍拥有 latest fence 时才写 `FAILED`；若已被 supersede，只记录本 attempt 的 failure receipt，不得覆盖新 heartbeat。
- `claim_formal_execution()`：
  - `COMPLETE` 必须永久拒绝 START/RESUME；
  - `FAILED` 可由新的、签名的 RESUME token 立即接管，无需等 6 小时；
  - `ACTIVE` 仍按租约规则处理；
  - 同时检查授权 output root 下是否已有有效 canonical completion，避免 heartbeat 丢失后重复运行。

### 必须新增的测试

- action 抛异常后 heartbeat 立即为 `FAILED`，新 RESUME 可立即 claim。
- 正常结束后 heartbeat 为 `COMPLETE`，6 小时以后仍不可 RESUME。
- 旧 attempt 在新 fence 后抛异常，不得把新 heartbeat 改成 `FAILED`。
- completion receipt 缺失或校验失败时绝不能写 `COMPLETE`。

## TC-06 — 改 output root 可拆成两个 logical job

### 当前错误链

`formal_execution.py:32-41` 把 `output_root_digest` 放入 `LOGICAL_JOB_INVARIANT_FIELDS`。因此相同 run role、logical run ID、arm、seed、parent、lineage、schedule 只换目录，会得到不同 `logical_job_digest`；两个 token 都能 `CLAIMED`。

### 必须修改

- 将 logical scientific identity 与 storage location 分离：
  - logical key 至少由 experiment/release identity、run role、logical run ID、arm、training seed、parent checkpoint、lineage 和 schedule 构成；
  - `output_root_digest` 继续作为签名 job binding 的属性，但不得参与 logical-job digest。
- 首个 START fence 固定 `authorized_output_root_digest`；同 logical key 的后续 START/RESUME 必须匹配该值，否则以专门错误拒绝。若未来需要搬迁，另做 owner-signed migration receipt，不要用换目录隐式重开任务。
- 升级 execution token、claim、fence、heartbeat schema 版本。旧 v1 registry 与新 schema 不得混用；建立新 registry 或提供显式、只读迁移工具。
- logical key 应包含 release/experiment namespace，避免两个独立实验恰好使用相同 run ID 时互相冲突。

### 必须新增的测试

- 两个 token 除 output root 外完全相同：第二个必须被识别为同一 logical job 并拒绝，不能产生第二个 digest。
- 两个不同 experiment/release 使用相同 run ID：必须得到不同 logical key。
- 合法 RESUME 使用原 output root：继续同一 fence chain，generation 单调增加。

## TC-07 — Source manifest 校验没有重新探测当前 runtime

### 当前错误链

`source_identity.py:37-133` 在 build 时探测 Python/Torch/CUDA/GPU/driver。`validate_source_tree_manifest()` 在约 363-379 行只对 manifest 里已经存储的 `runtime_environment` 重算 digest，没有调用新的 live probe。已复现验证阶段 live probe 调用次数为 0。

### 必须修改

- 将 `_runtime_environment()` 改为可测试的公开/内部稳定接口，例如 `probe_runtime_environment()`。
- `validate_source_tree_manifest()` 每次必须现场调用 probe，并把 fresh result 与冻结 manifest 做字段级比较；不能把旧值重新 hash 后当作现场证明。
- `prepare_formal_authorization()`、RESUME authorization 和 final closeout 都必须触发 fresh probe。
- runtime identity 至少绑定：
  - Python implementation/version；
  - Torch version、CUDA build/availability；
  - 选定 GPU index/name/compute capability/VRAM；
  - NVIDIA driver；
  - Ultralytics 与 SCTSR adapter 的 import origin 和内容 SHA；
  - 对正式训练有影响的已注册依赖版本。
- 对 `uv run --isolated` 的随机临时绝对路径做规范化，比较稳定身份和内容，不要因为临时父目录变化产生假失败。
- 将 fresh runtime digest 写入 authorization receipt、RUN_MANIFEST 和 completion receipt，使 closeout 可证明开始与结束环境一致。

### 必须新增的测试

- build manifest 后 monkeypatch live probe 的 Torch/CUDA/GPU/driver 任一字段，validate 必须失败且断言 probe 被调用。
- build 与 validate 的 uv 临时父目录不同但解释器版本/内容相同，应通过。
- authorization 通过后、closeout 前改变 live runtime，completion 必须失败。

## TC-08 — `sctsr_classification_trainer` import origin 未验证

### 当前错误链

- `training_system.bind_upstream()` 只绑定 6 个 Ultralytics 文件。
- `formal_cli.py:1022-1038` 修改 `sys.path` 后直接 `importlib.import_module("sctsr_classification_trainer")`，没有检查 `module.__file__` 或 SHA。

### 必须修改

- 将 `integrations/ultralytics/sctsr_classification_trainer.py` 加入 `UpstreamBinding` 或新增明确的 adapter binding；冻结 relative path、bytes、SHA-256。
- import 后立即：
  - 解析 `Path(module.__file__).resolve()`；
  - 要求它严格等于当前 repository root 下的注册 adapter 文件；
  - 重算 SHA-256 并与 source manifest/binding 比较；
  - 将 origin/SHA 写入 `prepared_trainer_binding` 与 runtime identity。
- 已经在 `sys.modules` 中预加载的同名错误模块也必须被上述检查拒绝，不能假设新插入 `sys.path` 会覆盖缓存。

### 必须新增的测试

- 在更高优先级路径放置同名 adapter，必须在 trainer setup 前失败。
- 预先向 `sys.modules` 注入同名假模块，必须失败。
- 正确 adapter 内容改 1 byte，必须失败。

## TC-09 — 二分类映射和两输出 head 校验发生得太晚

### 当前错误链

`prediction_runtime._binary_class_indices()` 在 endpoint 阶段才要求 `{0:no_target, 1:target_defect}`。trainer preflight 没有在 epoch 1 前同时证明 class directories、`class_to_idx`、dataset names、model names、`nc` 和输出 head 都严格为 2。一个额外空 class 目录可能让 Ultralytics 建出三输出模型，而样本计数仍看似正确。

### 必须修改

- 新增 `validate_binary_classification_contract(trainer, data_root)`，在 `trainer._setup_train()` 后、任何 optimizer step 前调用。
- 必须同时检查：
  - train 与 val 根目录的直接 class 子目录集合严格等于 `{"no_target", "target_defect"}`；
  - train/val dataset 的 `classes == ["no_target", "target_defect"]`；
  - `class_to_idx == {"no_target": 0, "target_defect": 1}`；
  - `trainer.model.names == {0: "no_target", 1: "target_defect"}`；
  - dataset/model `nc == 2`；
  - 最终 classification head 的 `out_features == 2`，或用无梯度 shape probe 证明输出最后一维为 2；
  - 所有 loader label 只含 0/1，且两类均非空。
- 把这份 binding 写入 `prepared_trainer_binding`、RUN_MANIFEST 和 completion validation。

### 必须新增的测试

- 添加空目录 `zzz_extra_class`，必须在 epoch 1 前失败。
- 交换 class_to_idx、改 model names、把 head 改成 3 输出，分别必须失败。
- 正常两类数据必须通过，并在 receipt 中出现同一 binary-contract digest。

## TC-10 — 新增 evidence log 的行尾身份不稳定

### 必须修改

- 在 `.gitattributes` 为所有被 evidence manifest 收录的文本类型设定明确 `text eol=lf`，至少覆盖：
  - `docs/stage1_sctsr_v4/tdd_receipts/**/*.log`；
  - `artifacts/**/08_reports/sctsr_v4_*/commands/**/*.log`；
  - 同目录被登记的 `.txt/.csv/.json/.md/.yaml`。
- 扩展 `tests/stage1_sctsr_v4/test_portable_text_identity.py`：遍历每一个 `EVIDENCE_MANIFEST.json`，枚举其全部文本条目，要求 Git attribute 为 LF 且 checkout bytes 中不存在 CRLF。
- 在属性规则生效后的干净 checkout 重新生成 evidence manifest，不能手工替换 hash。

## Runbook manifest 必须同步修复

当前 `docs/stage1_sctsr_v4/RUNBOOK_MANIFEST_v4.json` 自己会以 `RUNBOOK_IDENTITY_MISMATCH` 拒绝当前 HEAD。六个已有文档字节/SHA 已变化：

```text
ASSET_IDENTITY_LEDGER.md
EXPERIMENT_INTENT.md
FAIRNESS_CONTRACT.md
IMPLEMENTATION_GUIDE.md
KNOWN_BLOCKERS.md
READ_ME_FIRST.md
```

manifest 还漏掉至少以下四个 addendum：

```text
SCTSR_DATA_CONTENT_AND_T_REPAIR_ADDENDUM_20260815.md
SCTSR_LOGICAL_JOB_FENCING_ADDENDUM_20260815.md
SCTSR_ATOMIC_COMPLETION_ADDENDUM_20260815.md
SCTSR_SIMPLE_RANDOM_DEPLOYMENT_ADDENDUM_20260815.md
```

修复方式：从最终 post-checkout LF bytes 重新构建 manifest，然后运行真实 validator。不要把 validator 改成忽略 mismatch。

## 建议的测试文件落点

优先扩展已有文件，避免再造一套平行测试框架：

```text
tests/stage1_sctsr_v4/test_random_controls.py
tests/stage1_sctsr_v4/test_r2_addendum.py
tests/stage1_sctsr_v4/test_formal_input_snapshot.py
tests/stage1_sctsr_v4/test_formal_input_bindings.py
tests/stage1_sctsr_v4/test_prediction_runtime.py
tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py
tests/stage1_sctsr_v4/test_formal_execution_claim.py
tests/stage1_sctsr_v4/test_formal_completion_transaction.py
tests/stage1_sctsr_v4/test_source_identity.py
tests/stage1_sctsr_v4/test_formal_training.py
tests/stage1_sctsr_v4/test_portable_text_identity.py
```

建议新增两个集中故障注入文件：

```text
tests/stage1_sctsr_v4/test_formal_finalization_fencing.py
tests/stage1_sctsr_v4/test_endpoint_input_provenance.py
```

## 最终验收命令与产物

提交修复时请附真实命令、退出码、版本和日志 SHA。最低验收集：

```powershell
uv run --locked --python 3.11 pytest -q tests/stage1_sctsr_v4
uv run --locked --python 3.12 pytest -q tests/stage1_sctsr_v4
uv run --locked --python 3.11 python -m compileall -q stage1_sctsr_v4 scripts/stage1_sctsr_v4 integrations/ultralytics
uv lock --check
```

此外必须运行：

1. 干净 checkout/源码 archive 测试，使用显式 provisioning 命令放入 `yolo11l-cls.pt`，先校验：

```text
bytes: 28,553,700
SHA-256: 6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C
```

2. 一个有界真实数据工程 canary，至少覆盖 setup、一次真实 forward/backward/optimizer update、checkpoint save/reload、故障后 RESUME、E200 形式的 endpoint prediction、frontier、receipt、artifact index 和 completion validation。
3. 对 canary 输出逐文件重新读取，核对所有 manifest/receipt 引用的 path、bytes、SHA 与实际文件一致。
4. 上述所有负向测试必须证明错误被拒绝；不能只报告正向 suite passed。

## 重新审查时需要返回的信息

开发机修复后，请在同一共享目录新增回执，至少包含：

- 修复 commit SHA；
- TC-01..TC-10 对应的代码文件与测试名；
- 每个 RED 测试修复前的失败证据；
- 修复后的完整测试命令、退出码、passed/failed/skipped 数；
- 新 R2 identity digest、content digest、T/R2 image-SHA overlap count；
- endpoint input binding 示例；
- FAILED/COMPLETE heartbeat 示例；
- stale attempt 被 fence 拒绝的测试输出；
- clean-checkout 权重 provisioning 与 hash 校验方式；
- 明确声明是否执行过正式训练、是否打开过 blind/test。

在收到并复核这些材料之前，保持：

```text
formal_training_authorized: false
formal_training_started: false
sctsr_current_training_output_verdict: NO_GO
```
