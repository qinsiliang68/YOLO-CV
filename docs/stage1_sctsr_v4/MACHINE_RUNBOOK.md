# SCTSR v4 单训练机执行手册

## 1. 本机 AI/operator 开始前必须回答

在执行任何 formal 命令前，执行者必须能准确回答：

1. 这是 common parent 还是哪个 branch arm？
2. training seed、logical run ID 和 output root 是什么？
3. parent/initial checkpoint 的完整 SHA 是什么？
4. 本 job 唯一允许变化的 treatment 是什么？
5. 为什么 T 不是已验证 selector？
6. R2 是否精确绑定 owner-approved policy、selection seed、pool digest 和同池 arm 映射？
7. 为什么 exit 0 不等于完成？
8. E200/EMA/val_op 为什么不能用于选择方法？
9. 发生 OOM、disk full、kill 或 receipt 损坏时，为什么不能直接 `--resume`？
10. 本机是否在共享 canonical execution-claim registry 上 claim？

当前部署可使用 13 台 3090，其中最多 12 台并行、1 台作为故障替补。任务可由 operator
随机分配，不要求复杂 GPU 调度器；正确性只依赖每个逻辑 job 的 one-use token、全部机器
共享的 v2 claim registry、单进程指定一个 numeric CUDA device，以及同一 logical key
只能存在一条 fence chain。不得为提高利用率复制 registry 或换 output root 重开任务。

任何答案不清楚，停止并阅读 `EXPERIMENT_INTENT.md`、`FAIRNESS_CONTRACT.md` 和
`FAILURE_AND_RECOVERY.md`。不得以“命令能跑”代替理解。

## 2. 每个 job 的参数单

Release authority 必须给出以下全部值，禁止留空/`latest`/通配符：

```text
CHECKOUT_ROOT=<absolute clean checkout>
CANONICAL_DATASET_ROOT=<absolute immutable Sewer-ML root derived from registered manifests>
CLASSIFICATION_DATA_ROOT=<absolute hardlink-only train/val view; exact trainer_overrides.data>
PYTHON_VERSION=3.11 or 3.12
CUDA_DEVICE=<one integer>
RUN_ROLE=COMMON_PARENT or BRANCH
TRAINING_SEED=<integer>
LOGICAL_RUN_ID=<exact ID>
ARM_ID=<COMMON_PARENT_NR or one of 8 arms>
OUTPUT_ROOT=<new absolute directory; must not exist>
RECEIPT_ROOT=<outside OUTPUT_ROOT>
SOURCE_TREE_MANIFEST=<exact file>
FORMAL_IDENTITY=<exact file>
TRAINER_OVERRIDES=<exact file>
IDENTITY_MANIFEST=<exact 120k CSV>
RELEASE_AUTHORIZATION=<signed file>
RELEASE_TRUST_POLICY=<exact file>
EXECUTION_TOKEN=<signed one-use file>
EXECUTION_CLAIM_ROOT=<shared canonical UNC/root>
RUNBOOK_MANIFEST=<exact frozen file>
RUN_INTENT_ACKNOWLEDGEMENT=<exact file>
CONTRACT=<exact file>
ARMS=<exact file>
ASSET_REGISTRY=<exact file>
RUNTIME_CONFIG=<exact file>
SEED_REGISTRY=<exact file>
PARENT_CHECKPOINT=<branch only>
PARENT_ARTIFACT_INDEX=<branch only>
LINEAGE=<branch only>
SCHEDULE=<branch only>
IDENTITY_POOL=<zero or more exact manifests>
PARENT_CHECKPOINT_SHA256=<full 64 hex>
SCHEDULE_DIGEST=<full 64 hex; COMMON_PARENT uses 64 zeroes>
IDENTITY_POOL_BINDING_DIGEST=<full 64 hex; COMMON_PARENT uses 64 zeroes>
RESUME_CHECKPOINT_SHA256=<full 64 hex; START uses 64 zeroes>
RESUME_RECEIPT_DIGEST=<full 64 hex; START uses 64 zeroes>
```

本机 AI 不填缺失值，也不去 Downloads/Desktop/邻近 run 中自动寻找。

`CANONICAL_DATASET_ROOT` 用于完整 content-ledger 重验和 endpoint；
`CLASSIFICATION_DATA_ROOT` 仅用于 Ultralytics `train/`、`val/` DataLoader。二者可以是
不同目录，但必须在支持 hardlink 的同一卷上。训练文件必须与 canonical source 是
同一物理文件；同名同标签同 SHA 的独立 copy 也不接受。classification view 中出现
额外 class、额外图片、symlink、junction、缺失 link 或训练期间 inode 漂移时立即停止。

正式 loader 的目录合同是 `materialized_dataset_binding.v4`：classification view 顶层
只允许冻结的 `train/` 与 `val/` role roots；每个 role root 内只允许 loader 选中物理
文件及这些文件到 role root 的祖先目录闭包。setup 和终态重验都先以不跟随链接的方式
检查每一级祖先的 symlink/junction/reparse attribute，再核对 exact tree。任何 sibling
class、sibling role、嵌套额外目录或 unsupported filesystem entry 都失败，不能自动忽略。

## 3. 只读环境预检

```powershell
Set-Location <CHECKOUT_ROOT>
git status --porcelain=v1
git rev-parse HEAD
git diff --check
uv lock --check
nvidia-smi
```

要求：tracked/untracked source 均符合 source manifest，Python 为 3.11/3.12，单张
3090 可见，driver/CUDA/Torch 与 release runtime identity 匹配。不要用
`CUDA_VISIBLE_DEVICES` 隐式重映射后仍在 receipt 写原始 device；resolved device
必须可审计。

资产预检命令见 `TRAINING_OPERATIONS_MANUAL.md`。dataset content full verification
应得到 384,000 files 和 content digest `EDA939...DD6E`。若本地 data path 不同但
字节完全一致，可以通过；若代码自动 fallback 到别处，必须失败。

随后单独核对 `TRAINER_OVERRIDES.data == CLASSIFICATION_DATA_ROOT`；不要把 canonical
root 直接填进 `data`，除非它本身已经按冻结 manifests 原子物化了准确的 hardlink
classification view。runner 在 optimizer step 0 前、RESUME 后、endpoint 前和
completion 前都会重读 row-level Parquet binding 并扫描额外文件。

## 4. 生成本 job 的 run-intent acknowledgement

Runbook manifest 由 release authority 在 clean frozen checkout 中用
`build_runbook_manifest.py` 生成，一旦任何被登记文档变化就必须废弃重建。训练机在
读完该 manifest 所列全部文档、确认第 1 节十个问题，并取得本 job 的已签名 release
与 one-use token 后，执行：

```powershell
uv run --python <PYTHON_VERSION> python scripts/stage1_sctsr_v4/build_run_intent_acknowledgement.py `
  --repository-root <CHECKOUT_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --action START `
  --run-role <RUN_ROLE> `
  --logical-run-id <LOGICAL_RUN_ID> `
  --arm-id <ARM_ID> `
  --training-seed <TRAINING_SEED> `
  --formal-identity <FORMAL_IDENTITY> `
  --trainer-overrides <TRAINER_OVERRIDES> `
  --identity-manifest <IDENTITY_MANIFEST> `
  --asset-registry <ASSET_REGISTRY> `
  --parent-checkpoint-sha256 <PARENT_CHECKPOINT_SHA256> `
  --schedule-digest <SCHEDULE_DIGEST> `
  --identity-pool-binding-digest <IDENTITY_POOL_BINDING_DIGEST> `
  --release-authorization <RELEASE_AUTHORIZATION> `
  --execution-token <EXECUTION_TOKEN> `
  --execution-claim-root <EXECUTION_CLAIM_ROOT> `
  --resume-checkpoint-sha256 <RESUME_CHECKPOINT_SHA256> `
  --resume-receipt-digest <RESUME_RECEIPT_DIGEST> `
  --runbook-manifest <RUNBOOK_MANIFEST> `
  --acknowledgement-id <unique ACK ID> `
  --operator-agent-id <exact operator/AI ID> `
  --machine-id <exact machine ID> `
  --acknowledge-all-required-statements `
  --acknowledgement-output <RUN_INTENT_ACKNOWLEDGEMENT> `
  --output <RECEIPT_ROOT>/build_run_intent_acknowledgement.json
```

不得凭手工抄写计算三个 digest。Branch 的 schedule digest 来自已验证 schedule；pool
binding digest 来自 `validate_identity_pool_artifacts` 的完整结果；parent SHA 来自已
closeout parent。Common parent 的 schedule/pool digest按 schema 固定为 64 个 `0`。
START 的两个 resume digest 固定为 64 个 `0`。生成器不会 claim token、创建 trainer
或开始训练；它只产生一份最长有效期七天、与 exact token bytes绑定的确认书。

随后独立验证：

```powershell
uv run --python <PYTHON_VERSION> python scripts/stage1_sctsr_v4/validate_run_intent_acknowledgement.py `
  --acknowledgement <RUN_INTENT_ACKNOWLEDGEMENT> `
  --runbook-manifest <RUNBOOK_MANIFEST> `
  --repository-root <CHECKOUT_ROOT> `
  --output <RECEIPT_ROOT>/validate_run_intent_acknowledgement.json
```

正式 runner 仍会独立从实际输入重新派生 context；上述自检 PASS 不能替代 runner
比较。任何 job 参数或 token bytes变化都必须生成新的 acknowledgement。

## 5. 正式授权前真实图工程 canary

在任何 release、seed 或正式 runner 之前，只能用
`scripts/stage1_sctsr_v4/run_engineering_canary.py` 执行一个训练角色真实图 microbatch。
完整命令和判定字段见 `IMPLEMENTATION_GUIDE.md` 第 11 节。训练机 AI 必须确认输出
语义为 `ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT`，并确认 `formal_training_started`、
`assignments_generated`、`engineering_gate_generated`、`pilot_release_generated`、
`blind_holdout_opened`、`test_accessed`、`method_effectiveness_claimed` 全部为 JSON
`false`。该 PASS 只解除“本机能否读图、跑一步和写恢复产物”的工程疑问，不解除
R2 policy/pool identity mismatch、release、seed 或科学合同阻断。

## 6. Common parent START 模板

以下模板中的每个 `<...>` 都必须由签名 job 参数单替换；出现任何占位符即停止：

```powershell
uv run --python <PYTHON_VERSION> python scripts/stage1_sctsr_v4/run_common_parent.py `
  --repository-root <CHECKOUT_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --training-seed <TRAINING_SEED> `
  --identity-manifest <IDENTITY_MANIFEST> `
  --trainer-overrides <TRAINER_OVERRIDES> `
  --formal-identity <FORMAL_IDENTITY> `
  --execution-mode formal `
  --release-authorization <RELEASE_AUTHORIZATION> `
  --release-trust-policy <RELEASE_TRUST_POLICY> `
  --execution-token <EXECUTION_TOKEN> `
  --execution-claim-root <EXECUTION_CLAIM_ROOT> `
  --run-intent-acknowledgement <RUN_INTENT_ACKNOWLEDGEMENT> `
  --runbook-manifest <RUNBOOK_MANIFEST> `
  --source-tree-manifest <SOURCE_TREE_MANIFEST> `
  --contract <CONTRACT> `
  --arms <ARMS> `
  --asset-registry <ASSET_REGISTRY> `
  --runtime-config <RUNTIME_CONFIG> `
  --seed-registry <SEED_REGISTRY> `
  --output <RECEIPT_ROOT>/run_common_parent.json
```

`--run-intent-acknowledgement` 与 `--runbook-manifest` 都是 mandatory 参数。runner
在 claim token 和构造 trainer 前独立验证二者；任一缺失、过期或 context 不同即失败。

## 7. Branch START 模板

```powershell
uv run --python <PYTHON_VERSION> python scripts/stage1_sctsr_v4/run_branch.py `
  --repository-root <CHECKOUT_ROOT> `
  --output-root <OUTPUT_ROOT> `
  --parent-checkpoint <PARENT_CHECKPOINT> `
  --parent-artifact-index <PARENT_ARTIFACT_INDEX> `
  --arm-id <ARM_ID> `
  --training-seed <TRAINING_SEED> `
  --epoch 121 `
  --identity-manifest <IDENTITY_MANIFEST> `
  --trainer-overrides <TRAINER_OVERRIDES> `
  --formal-identity <FORMAL_IDENTITY> `
  --lineage <LINEAGE> `
  --schedule <SCHEDULE> `
  --identity-pool <PRIMARY_POOL_MANIFEST> `
  --identity-pool <FALLBACK_POOL_MANIFEST_IF_REQUIRED> `
  --execution-mode formal `
  --release-authorization <RELEASE_AUTHORIZATION> `
  --release-trust-policy <RELEASE_TRUST_POLICY> `
  --execution-token <EXECUTION_TOKEN> `
  --execution-claim-root <EXECUTION_CLAIM_ROOT> `
  --run-intent-acknowledgement <RUN_INTENT_ACKNOWLEDGEMENT> `
  --runbook-manifest <RUNBOOK_MANIFEST> `
  --source-tree-manifest <SOURCE_TREE_MANIFEST> `
  --contract <CONTRACT> `
  --arms <ARMS> `
  --asset-registry <ASSET_REGISTRY> `
  --runtime-config <RUNTIME_CONFIG> `
  --seed-registry <SEED_REGISTRY> `
  --output <RECEIPT_ROOT>/run_branch.json
```

NR 没有 identity pool；单 pool arm 传一次；T→R2 传 T 和 R2 各一次。不得用参数
顺序猜 primary/fallback，manifest role 和 schedule 会共同验证。

## 8. 运行中监控

训练机只监控，不修改训练：

- GPU utilization、memory、temperature、power；
- process CPU/RAM；
- disk free/IO；
- 每 epoch base rows=120,000、steps=938；
- replay planned/actual slots；
- checkpoint/Parquet 原子 publish；
- receipt chain 和 rolling pointer；
- data wait、epoch time、evaluation time。

不得因 GPU utilization 偶尔下降而增加 workers、prefetch、batch 或并发进程。不得
因为磁盘压力删除已完成 generation。异常按 stop rule 处理。

## 9. START 与 RESUME 不可混用

- START：output root 不存在，token action=`START`；
- RESUME：已有未终结 run、完整 contiguous prefix、独立 token action=`RESUME`、
  `--resume` 和 canonical `--resume-setup-root`；
- 已 terminal complete 的 run 不可 resume；
- 旧 START token 不可用于 resume；
- 同 token/nonce 不可再 claim；
- 手工把 `.inprogress` 改名成 `.complete` 永久作废。

claim 成功后，resume preparation、trainer setup、training、branch finalization context
提取和 fenced finalization 全部处于同一个 terminalization boundary。任一阶段抛出异常，
当前 fence 必须立即写 `FAILED` heartbeat 和 hash-bound terminal receipt；只有随后签发的
合法 RESUME token 才能以 `fence_generation + 1` 接管，不能等待 lease 自然过期或手工改
claim registry。

RESUME 命令参数与 START 相同，另加：

```text
--resume
--resume-setup-root <OUTPUT_ROOT>/10_resume_setup/epoch_<next>.generation_1
```

执行前必须先运行 recovery preflight 并取得 required-free-bytes PASS。
RESUME 必须先以 `--action RESUME`、新 execution token、last valid checkpoint SHA 和
receipt-chain digest重新运行第 4 节 acknowledgement 命令；不得复用 START 确认书。

## 10. Job 结束后的固定命令

```powershell
uv run --python <PYTHON_VERSION> python scripts/stage1_sctsr_v4/validate_run.py `
  --run-root <OUTPUT_ROOT> `
  --output <RECEIPT_ROOT>/validate_run.json
```

不要在正式模式加 `--allow-synthetic-columnar-fallback`。然后由独立审查者运行
closeout；训练机自身的“完成”消息不能替代 closeout。

## 11. 本机退出时必须汇报

- job parameters 和所有输入 SHA；
- start/end UTC；
- process exit code；
- canonical run status；
- completed epoch range；
- last valid receipt/pointer/checkpoint SHA；
- validation/closeout status；
- quarantined paths；
- peak GPU/RAM/disk；
- output bytes 和 manifest SHA；
- formal side-effect booleans；
- 明确写 `SCIENTIFIC_RESULT_NOT_YET_INTERPRETED`。

禁止只汇报“训练完成”“loss 正常”或“GPU 跑满”。
