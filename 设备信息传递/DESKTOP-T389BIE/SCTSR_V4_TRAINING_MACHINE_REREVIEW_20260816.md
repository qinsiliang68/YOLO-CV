# SCTSR v4 训练机独立复审：仍为 NO-GO，需补两处 fail-closed 缺口

- 日期：2026-08-16（Asia/Shanghai）
- 复审对象：`codex/sctsr-v4-training-output-fixes`
- 固定提交：`1f0f12827dc6f79e107459fcb3135b99eaaa1423`
- 比较基线：`54e320020c60c8d7a11f59d2fa606ff203fd0d4d`

```text
REREVIEW_STATUS=NO_GO_FIX_REQUIRED
FORMAL_TRAINING_STARTED=false
ASSIGNMENT_GENERATED=false
GATE_GENERATED=false
RELEASE_GENERATED=false
TEST_ACCESSED=false
BLIND_HOLDOUT_OPENED=false
METHOD_EFFECTIVENESS_CLAIMED=false
```

本轮确认 TC-01～TC-10 的主体修复大部分已经落到代码，提交身份、R2 内容去重、证据哈希、端点输入绑定、逻辑作业命名空间、CUDA 工程 canary 入口等均有实质实现。但独立复审发现下列两个可复现缺口。它们不是文档措辞问题，必须修改代码并增加负向测试后再复审；在此之前不得启动正式训练、签发 assignment/gate/release 或打开 test/blind。

## 已独立确认的范围

- 远端分支 HEAD 与本地固定 checkout 均为 `1f0f12827dc6f79e107459fcb3135b99eaaa1423`。
- 修复分支相对基线为 11 commits、94 files、ahead 11 / behind 0。
- `EVIDENCE_MANIFEST.json` 注册 38 个证据文件；独立按 bytes + SHA-256 重算为 38/38 一致，包括 4 个 Windows 长路径文件。
- `SCTSR_V4_TRAINING_OUTPUT_FIX_RESPONSE_20260816.md`：19,963 bytes，SHA-256 `8AA1448AB4DD35266797AB0E8126E37D1C0C94F7AF29B3D90AA6D83D75E45CA7`。
- `EVIDENCE_MANIFEST.json`：6,062 bytes，SHA-256 `4687EC6161A14A71079B8119D9E39CC9599B09E2ED4093C96496F3BCE3F47AAF`。
- 官方 `yolo11l-cls.pt`：28,553,700 bytes，SHA-256 `6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C`。
- Python 3.11 的 v4 独立合约复核累计 455 个不同测试通过。另一个全仓历史可移植性测试因 sparse checkout 未物化全部历史 manifest 而未直接执行；当前 v4 证据包已通过独立 Git blob、checkout bytes 和 SHA-256 交叉核验，不能把该环境性未执行项写成产品失败，也不能把它冒充为 pytest 通过。
- 训练机 CUDA 复核状态见本文末尾“训练机工程 canary”。

## P1：claim 成功后，setup/resume 异常不会立即写 FAILED 终态

### 当前代码路径

`scripts/stage1_sctsr_v4/run_branch.py`：

- 第 217～225 行先执行 `claim_formal_execution(...)`；
- 第 226～236 行执行 `prepare_formal_resume_context(...)` 和 resume TOCTOU 校验；
- 第 237～246 行执行 `build_prepared_trainer(...)`；
- 第 247 行才进入 `try`；
- 第 268～269 行的 `mark_execution_failed(...)` 因此只覆盖 `run_prepared_branch(...)`；
- 第 271 行的 `result.pop("_finalization_context")` 也在该 `try` 之外。

`scripts/stage1_sctsr_v4/run_common_parent.py` 同样存在该窗口：claim 在第 165～173 行，resume preparation 在第 174～184 行，trainer build 在第 185～192 行，`try` 到第 193 行才开始。

### 可触发结果

以下任一真实 fail-closed 异常发生时，logical job 已经被 claim，但 CLI 不调用 `mark_execution_failed(...)`：

1. resume 状态在 preview 与 fenced preparation 之间变化；
2. trainer 初始化时发现权重、二分类 head、labels、dataset binding、adapter/runtime bytes 不匹配；
3. branch runner 返回结果缺少 `_finalization_context`。

此时 heartbeat 会继续保持 `ACTIVE`，直到 lease 过期；新的 START/RESUME 不能立即接管。现有单元测试只证明 `mark_execution_failed` 自身幂等，以及 finalizer 内部异常会 FAILED，没有覆盖“claim 后、training try 前”的 runner 控制流。`tests/stage1_sctsr_v4/test_formal_resume.py:286-293` 甚至只用字符串位置证明 `preview < claim < mutate`，没有验证 mutate/setup 失败后的终态。

### 必须怎样改

不要只扩大训练函数附近的 `try` 一两行。两个 runner 都必须保证：从 claim 成功开始，到进入已经自带 terminalization 的 fenced finalizer 或返回已终结结果为止，任何 `BaseException` 都会幂等调用：

```python
mark_execution_failed(
    execution_claim,
    expected_job_bindings=execution_job,
    error=exc,
)
```

推荐把该语义收敛成一个可测 helper，例如 `execute_claimed_phase(claim, expected_job_bindings, operation)`；runner 在 claim 后把 resume preparation、TOCTOU check、trainer build、training call，以及 branch 的 `_finalization_context` 提取全部放入该 helper。`execute_fenced_finalization(...)` 已有自己的终态处理，可继续作为唯一 finalization fence；外层若可能重复标记，必须依赖并验证现有幂等语义，不能吞掉原始异常。

### 必须新增的负向测试

至少覆盖两个 runner（common parent 与 branch）和以下三类注入：

1. claim 成功后 monkeypatch `prepare_formal_resume_context` 抛出异常；
2. claim 成功后 monkeypatch `build_prepared_trainer` 抛出异常；
3. branch 的 prepared run 返回不含 `_finalization_context` 的结果。

每个测试不能只断言 CLI 非零，还必须读取 claim registry 并断言：

```text
heartbeat.status == FAILED
terminal receipt exists and matches heartbeat.terminal_receipt_sha256
the original logical_job_digest is unchanged
a valid RESUME token can claim immediately, without waiting for lease expiry
new fence_generation == old fence_generation + 1
```

验收命令应单独列出该测试文件，并保留完整 stdout/stderr、退出码、源提交和证据哈希。

## P2：materialized loader 只扫描叶子 class 目录，额外 sibling class 可逃逸

### 当前代码路径

`stage1_sctsr_v4/dataset_adapter.py`：

- 第 356～389 行用 `scan_roots.add(path.parent)` 记录每个选中样本的叶子 class 目录；
- 第 431～440 行的 setup 扫描只遍历这些叶子目录；
- 第 450 行把这些叶子目录写入 `materialized_roots`；
- 第 502～519 行的终态 revalidation 仍只遍历相同叶子目录。

因此，若合法样本在：

```text
classification_view/train/no_target/a.png
```

setup 后新增：

```text
classification_view/train/injected_class/extra.png
```

`injected_class` 不属于任何 `path.parent` scan root，当前 `revalidate_materialized_dataset_binding(binding)` 会返回 `PASS`。独立最小复现的实际结果为：

```text
ACCEPTED {'status': 'PASS', 'role': 'train', 'row_count': 1,
          'binding_digest': '50AE...'}
```

现有 `test_materialized_binding_rejects_unregistered_extra_file` 和 `test_materialized_binding_rejects_extra_file_added_after_setup` 都把 extra file 放在已选样本的同一叶子 class 目录，所以没有覆盖 sibling class。

另有一个同源边界错误：第 421～430 行检查 materialized path 的祖先时，以 `canonical_root` 作为停止边界。canonical root 与 classification view 已经分离后，第一次循环通常就在选中文件处停止，未检查 materialized class/role/root 祖先上的 junction/reparse point。

### 必须怎样改

不要继续把叶子 class 目录当作“整棵 loader tree”。需要显式绑定并保存以下边界：

```text
materialized_data_root = classification_view
train_role_root        = classification_view/train
val_role_root          = classification_view/val
```

建议给 `validate_materialized_dataset_bytes(...)` 增加明确的 `materialized_role_root` 参数；`formal_cli.validate_prepared_trainer_datasets(...)` 分别传 `dataset_root / "train"` 和 `dataset_root / "val"`。binding schema 记录 role root，不再把 `path.parent` 集合作为完整扫描边界。

对每个 role root 做 exact-tree 校验：

1. role root 必须位于 classification view 内、同卷、非 symlink/reparse；
2. allowed files 必须精确等于 loader 已选物理路径集合；
3. allowed directories 必须精确等于这些文件到 role root 的祖先闭包；
4. role root 下任何其他文件、class 目录或嵌套目录均失败；
5. 从每个文件到 role root 的每一级祖先都用 `lstat`/Windows reparse attribute 检查，停止边界必须是 materialized role root，不能是 canonical root；
6. classification view 顶层还要绑定允许的 role root 集合，拒绝额外 sibling role；
7. setup、pre-step、checkpoint/epoch closeout 和 formal endpoint publication 前必须调用同一 revalidation 语义。

遍历 junction 时不要先递归进入再检查。应使用不跟随链接的目录遍历方式，在入栈前检查 symlink/reparse attribute，避免 `Path.rglob` 在 Windows junction 行为差异下越界。

### 必须新增的负向测试

在已有测试之外至少增加：

1. setup 前存在 `train/injected_class/extra.png`，setup 必须失败；
2. setup 后新增 `train/injected_class/extra.png`，revalidation 必须失败；
3. setup 后新增 classification view 顶层 sibling role，revalidation 必须失败；
4. 把 train role root、class dir 或中间祖先替换成 directory junction/reparse point，必须在递归前失败；
5. 合法的 `train/{no_target,target}` 与 `val/{no_target,target}` 同卷硬链接树仍通过；
6. 以上负向测试同时覆盖 setup 与 endpoint publication 前的真实调用链，不能只测孤立 helper。

## 修复后的最低回传材料

开发机下一轮回复必须包含：

- 新 commit SHA 与相对 `1f0f128...` 的 diff；
- 两处修复的精确文件/行和 binding schema 是否升级；
- 上述 runner terminalization 负向测试日志；
- sibling class / sibling role / ancestor reparse 负向测试日志；
- Python 3.11 与 3.12 全量 v4 pytest；
- v3 回归；
- 在训练机重新执行真实图片、真实权重、CUDA 单步 engineering canary 的 receipt 和轻量 artifact manifest；
- 所有新增证据的 bytes + SHA-256 manifest；
- 明确声明没有启动正式训练、没有签发 assignment/gate/release、没有访问 test/blind。

只有两处缺口都修复、负向测试先红后绿、训练机复核通过后，状态才可以从 `NO_GO_FIX_REQUIRED` 改为 `READY_FOR_FORMAL_AUTHORIZATION_REVIEW`。这仍不等同于 SCTSR 有效，也不自动授权正式训练。

## 训练机工程 canary

- 目标节点：`physical64 / 192.168.100.9 / DESKTOP-DGQTJ9O`
- GPU：NVIDIA GeForce RTX 3090，24,576 MiB，UUID `GPU-396ff5c3-1c32-1614-80fe-f2822833da4b`
- 源码位置：`C:\Users\ASUS\Desktop\ssh\AI\sctsr_rereview_1f0f128`（真实 C 盘）
- 数据位置：`C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset`（真实 C 盘）
- 输出位置：`D:\ssh\AI\artifacts\sctsr_rereview_1f0f128_canary_20260816`
- 精确源码包：1,885,693 bytes，SHA-256 `EEABEF90AC3AF924DB23C49ED7E828109939BCE37F602ACC0D2486BBB93D056A`

```text
CANARY_STATUS=PASS
CANARY_ROLE=ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT
FORMAL_TRAINING_STARTED=false
TEST_ACCESSED=false
BLIND_HOLDOUT_OPENED=false
```

训练机执行命令使用项目私有 C 盘 venv，并由 `uv run --locked --python 3.11 --extra dev` 在运行前校准；共享 venv 未被修改。实际结果：

- receipt：9,563 bytes，SHA-256 `2DF902B61ACBC0E621153A8C159A228D5911E65D414653F2905ACBB17484ADE8`；CLI 与 result 均为 `PASS`。
- report：9,221 bytes，SHA-256 `4782692C7F47927A7AFAE2FDDEA4BF7562D2F4A63EE02C4B487098B280ED5F27`。
- Python `3.11.15`，torch `2.11.0+cu128`，CUDA build `12.8`，NVIDIA driver `531.29`。
- 设备为 `cuda:0 / NVIDIA GeForce RTX 3090 / compute capability 8.6 / 25,769,279,488 bytes`。
- frozen trainer 源码来自本次 C 盘源码包内的 `YOLOv11/ultralytics/models/yolo/classify/train.py`，不是 installed Ultralytics。
- 4 张真实训练角色图片逐文件 bytes/SHA 校验通过；真实 `yolo11l-cls.pt` 校验通过。
- forward、base backward、replay backward 均通过；optimizer step `1`，EMA update `1`，model state digest 前后不同。
- checkpoint：103,328,511 bytes，SHA-256 `078EA237D29F588A9D31A8EEAA82F9AE2739640F78B84F0A3636C6DA61DD228D`；save/reload 均通过。
- corrupt checkpoint 被拒绝，partial generation 被 quarantine；pointer 与 last complete epoch 均为 121。
- artifact manifest：2,980 bytes，SHA-256 `83DA12A29D650ACF0F4D0BFB56CF2ED1751A5211A474B44F8E2D477305C71462`，`file_count=15`，manifest digest `4BDE30A609A7C3C897F64BEE588C11CEC844E9F875BACF5285E684708563F321`。
- 独立重算 manifest 内 15 个文件的 bytes/SHA：15/15 一致；extra=0、missing=0；加 manifest 自身共 16 个文件。
- receipt 明确记录 `formal_training_started=false`、`assignments_generated=false`、`engineering_gate_generated=false`、`pilot_release_generated=false`、`test_accessed=false`、`blind_holdout_opened=false`、`method_effectiveness_claimed=false`。
- canary 后无匹配的 uv/python/yolo 残留进程；GPU 回到 399 MiB、0% utilization、34°C。

该节点非交互环境没有可用 Git checkout 元数据，因此 receipt 的 `source_tree_git_head` / `source_tree_git_dirty` 为 `null`。本轮没有把它伪装成 commit 证明；源码身份由控制机从固定提交 `1f0f128...` 生成的 1,885,693-byte Git archive 及其 SHA-256 `EEABEF...D056A` 锚定，训练机接收后重新核对 bytes/SHA。canary 的 source-tree/runtime digest 为 `515B8F4888D5C0B17A84F05407627E373CB71FAD634E0EFC8C911F07E3E275D7`。

这个 PASS 只证明训练机上的 GPU 工程路径可执行，不能抵消前述 P1/P2 fail-closed 缺口，也不改变总判定 `NO_GO_FIX_REQUIRED`。

已收集到 Git 的轻量原始证据位于 `设备信息传递/DESKTOP-T389BIE/sctsr_v4_training_machine_rereview_20260816_evidence/`。目录包含 receipt、report、artifact manifest 和 `EVIDENCE_INDEX.json`；103 MB checkpoint 等大文件只保留在上述训练机 D 盘路径。
