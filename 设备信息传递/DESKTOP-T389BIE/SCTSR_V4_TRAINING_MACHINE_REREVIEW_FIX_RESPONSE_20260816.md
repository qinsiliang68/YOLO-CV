# SCTSR v4 训练机复审缺口修复回复

## 1. 结论先行

训练机在 `eb3efece2c5ed01d918dc1701759473e5bc9ad42` 提出的两个实质缺口均已按 RED→GREEN 修复，并拆成可独立回滚的代码提交：

1. `36aa756c38701a22bcbff47b181282b139e97f9f`：claim 后统一异常终态化；
2. `44e2340b0591847aeafc322cc3335be5b7a18ea7`：精确 materialized role-tree 与全祖先 reparse 检查；
3. `c9dc0b9044e9188e602ab4df97dc3906127211f5`：直接加载两个正式 runner 的故障注入测试；
4. `a6baad0dce9cbd427570aa267614a1566434b25f`：冻结对应运维语义与 runbook manifest v6。

当前开发机判定只能提升为：

```text
READY_FOR_TRAINING_MACHINE_REREVIEW
FORMAL_TRAINING_AUTHORIZED=false
METHOD_EFFECTIVENESS_CLAIMED=false
```

原因是新代码尚未在 physical64 / RTX 3090 上重复真实图、真实权重 CUDA canary。旧 canary 的 PASS 是 `1f0f128...` 的工程证据，不能自动转移到新提交。训练机复验通过后，训练机可独立决定是否把状态改为 `READY_FOR_FORMAL_AUTHORIZATION_REVIEW`；这仍不等同于签发正式 seeds/release/token，也不代表 SCTSR 有效。

## 2. 修复基线与证据接入

- 修复基线：`1f0f12827dc6f79e107459fcb3135b99eaaa1423`
- 训练机复审原提交：`eb3efece2c5ed01d918dc1701759473e5bc9ad42`
- 训练机原始复审证据以 cherry-pick 提交 `8196829` 原样接入本分支；没有修改其 verdict、receipt 或 canary 数字。
- 实现冻结提交：`c9dc0b9044e9188e602ab4df97dc3906127211f5`
- runbook 冻结提交：`a6baad0dce9cbd427570aa267614a1566434b25f`
- 修复分支：`codex/sctsr-v4-training-output-fixes`
- 原始脏工作树、旧 120/240-run 代码与产物未改动。

## 3. 对 P1 的逐项回复：claim 后 ACTIVE 残留

### 3.1 代码结构

`stage1_sctsr_v4/formal_execution.py:1299` 新增 `execute_claimed_phase(...)`。语义为：

- operation 正常返回时不改写结果；
- claim 后任意 `BaseException` 均调用 `mark_execution_failed(...)`；
- terminalization 成功后重新抛出原始异常，不吞异常、不改异常类型；
- 内层 fenced finalizer 已写 FAILED 时，利用现有幂等语义，不产生第二条冲突终态；
- 若 newer generation 或 terminal COMPLETE 阻止旧 fence 写终态，拒绝覆盖较新状态，并把 terminalization 异常作为 cause，原始异常仍为主异常。

两个正式 runner 都从 claim 成功后的第一步进入这一边界：

- `scripts/stage1_sctsr_v4/run_common_parent.py:20,192,237`
- `scripts/stage1_sctsr_v4/run_branch.py:40,244,386`

边界内包含：

1. `prepare_formal_resume_context(...)`；
2. resume preview/preparation TOCTOU 比较；
3. `build_prepared_trainer(...)`；
4. prepared training call；
5. branch 的 `result.pop("_finalization_context")`；
6. branch endpoint、artifact index、completion 的 fenced finalization。

因此不存在“claim 已成功，但 try 尚未开始”的窗口。

### 3.2 失败优先测试

第一轮共享 helper 测试在未实现时得到：

```text
11 failed, 45 passed in 14.58s
```

随后为避免“只给 runner_role 打标签但没有加载 runner”的弱测试，又增加真实 runner 模块加载测试。该测试在 top-level runner boundary 尚不存在时先得到：

```text
5 failed, 21 deselected in 3.00s
```

实现后得到：

```text
5 passed, 21 deselected in 2.11s
```

五个用例分别覆盖：

- common parent / resume preparation；
- common parent / trainer setup；
- branch / resume preparation；
- branch / trainer setup；
- branch / missing finalization context。

每个用例都用真实 claim registry 断言：

- 原始异常对象被保留；
- heartbeat.status 为 `FAILED`；
- terminal receipt 存在，且 SHA-256 等于 heartbeat 登记值；
- logical_job_digest 不变；
- 不等待 lease expiry 即可由合法 RESUME token claim；
- 新 fence_generation 等于旧 generation + 1。

## 4. 对 P2 的逐项回复：sibling class/role 与 junction 逃逸

### 4.1 binding schema 升级

binding 已从 `stage1.sctsr.materialized_dataset_binding.v3` 升级为：

```text
stage1.sctsr.materialized_dataset_binding.v4
```

新增并冻结：

- `materialized_data_root`：classification view 根；
- `materialized_role_root`：本 loader 的精确 `train` 或 `val` 根；
- `allowed_materialized_role_roots`：classification view 顶层允许的完整 role 集合；
- `materialized_top_level_exact`：是否执行顶层 exact-set 检查。

正式调用位于 `stage1_sctsr_v4/formal_cli.py:724-742`：

- train loader 绑定 `<classification_view>/train`；
- val_model loader 绑定 `<classification_view>/val`；
- 两者共同登记唯一允许的顶层集合 `{train, val}`。

### 4.2 不再使用叶子 class scan roots

`stage1_sctsr_v4/dataset_adapter.py` 现在执行三层检查：

1. `:333 _validate_non_reparse_chain(...)`
   - 使用未 `resolve()` 的绝对路径；
   - 从物理文件逐级 `lstat` 到绑定 role root；
   - 每一级都检查 symlink 与 Windows reparse attribute。
2. `:356 _validate_materialized_role_set(...)`
   - classification view 顶层必须恰好等于登记的 role roots；
   - sibling role、普通文件、junction/reparse 均失败。
3. `:405 _validate_exact_role_tree(...)`
   - allowed files 精确等于 loader 选中物理路径；
   - allowed directories 精确等于这些文件到 role root 的祖先闭包；
   - sibling class、额外嵌套目录、额外文件或 unsupported entry 均失败；
   - 遍历使用 `os.scandir(..., follow_symlinks=False)` 语义，在入栈前拒绝链接，不会先进入 junction 再检查。

同一 v4 binding 在 setup、训练边界、resume、endpoint 与 completion 复验路径继续由 `revalidate_materialized_dataset_binding(...)` 使用，不另建一套宽松规则。

### 4.3 负向与正向测试

`tests/stage1_sctsr_v4/test_dataset_adapter.py` 新增并通过：

- setup 前 `train/injected_class/extra.png` 拒绝；
- setup 前 classification view 顶层 sibling role 拒绝；
- setup 后 sibling class 拒绝；
- setup 后 sibling role 拒绝；
- role/class/中间祖先被识别为 reparse 时，在递归前拒绝；
- 合法 `train` 与 `val` 同卷 canonical hardlink 树均通过 setup 与 revalidation；
- 既有 byte replacement、inode/hardlink、同叶子 extra-file 测试继续通过。

定向结果：

```text
17 passed in 1.68s
```

## 5. 完整回归结果

最终实现冻结后重新串行/隔离执行：

| 检查 | 结果 |
| --- | --- |
| `uv lock --check` | PASS，77 packages resolved |
| Python 3.11 v4 | `468 passed in 180.93s` |
| Python 3.12 v4 | `468 passed in 178.16s` |
| Python 3.11 v3 regression | `181 passed, 3 skipped in 6.04s` |
| Python 3.11 compileall | PASS |
| Python 3.12 compileall | PASS |
| `git diff --check` | PASS |
| `stage1_gapvalue240` vs base | unchanged |
| `stage1_dynamic_replay_v3` vs base | unchanged |
| `YOLOv11` vs base | unchanged |
| runbook-intent 定向测试 | `26 passed` |

测试总数从中间轮次的 469 调整为最终 468，是因为将原先 6 个共享-helper 参数标签用例收紧为 5 个与真实 runner/stage 一一对应的用例，不是功能回退。

## 6. runbook 同步

以下文档已同步：

- `docs/stage1_sctsr_v4/MACHINE_RUNBOOK.md`
- `docs/stage1_sctsr_v4/FAILURE_AND_RECOVERY.md`
- `docs/stage1_sctsr_v4/RUNBOOK_MANIFEST_v6.json`
- `docs/stage1_sctsr_v4/RUNBOOK_MANIFEST_BUILD_RECEIPT_v6.json`

v6 明确要求：

- classification view 顶层只允许冻结的 `train/val`；
- role 内只允许选中文件及其祖先闭包；
- 不跟随链接地检查每级祖先；
- claim 后 resume/trainer/training/context/finalization 任一异常立即 FAILED；
- 只有新 RESUME token 可 generation+1 接管。

旧 runbook manifests 保留为历史身份，没有覆盖或删除。

## 7. 训练机必须执行的下一步

训练机请从远端分支 `codex/sctsr-v4-training-output-fixes` 拉取本回复所在最终 HEAD，并按顺序执行：

```powershell
uv lock --check

uv run --isolated --python 3.11 --extra dev `
  pytest tests/stage1_sctsr_v4/test_formal_execution_claim.py `
  -k claimed_runner_phase_failure -q

uv run --isolated --python 3.11 --extra dev `
  pytest tests/stage1_sctsr_v4/test_dataset_adapter.py -q

uv run --isolated --python 3.11 --extra dev `
  pytest tests/stage1_sctsr_v4 -q
```

然后必须在 physical64 使用真实 Sewer-ML 训练角色图片、注册 `yolo11l-cls.pt` 和 RTX 3090 重跑 engineering canary，并重新登记：

- source commit/archive SHA；
- 真实图片逐文件 bytes/SHA；
- optimizer step=1；
- EMA update=1；
- model digest changed；
- checkpoint save/reload；
- corruption/quarantine/resume；
- 新 binding v4 内容；
- 产物 manifest bytes/SHA；
- 无残留进程；
- formal side effects 全为 false。

训练机还应直接复现：

1. claim 后 resume setup 抛错；
2. claim 后 trainer setup 抛错；
3. branch 结果缺 finalization context；
4. setup 前/后 sibling class；
5. setup 前/后 sibling role；
6. Windows junction/reparse ancestor。

如果这些全部通过，训练机可把这两个 finding 标为 `RESOLVED`，并将工程状态提升到 `READY_FOR_FORMAL_AUTHORIZATION_REVIEW`。任何一项失败仍为 `NO_GO_FIX_REQUIRED`。

## 8. 明确未做事项

本轮开发机没有：

- 启动 formal training；
- 生成或签发正式 seed；
- 生成 assignment、engineering gate 或 pilot release；
- 签发 formal release 或 one-use job token；
- 访问 val_op 以选择方法/checkpoint；
- 访问 test/blind holdout；
- 声称 SCTSR、T、R2 或任何 Q/R/A/D 信号有效。

机器可读结果位于：

`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_training_machine_rereview_fixes_20260816/`

其中包括 `FINDINGS.json`、`COMMAND_INDEX.json`、`REVIEW_VERDICT.json`、命令摘要与 SHA manifest。
