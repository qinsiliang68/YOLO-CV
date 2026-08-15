# SCTSR v4 实施与运行手册

## 1. 定位与权威规范

SCTSR v4 是在 `qinsiliang68/YOLO-CV` 的冻结历史基线
`main@a70ba60485dd32c2f8b4268b8f28ea2d3549f42f` 之上增加的隔离实现线。唯一完整规范是：

`artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md`

冻结 taskbook blob 为 `b201d021712e9c6614e119d35f0e14bdf405c6be`。本实现只证明“代码合同可施工、可失败封闭审计”，不证明 SCTSR、T、R1、R2 或任何候选信号有效。

当前明确禁止：正式训练、正式 seed、assignment、engineering gate、pilot release、selector training、blind/test 访问、用 `val_op` 选方法或 checkpoint，以及任何方法有效性声明。

## 2. 安装与运行环境

正式合同只接受 Python 3.11 或 3.12。仓库通过 `uv.lock` 冻结依赖，PyArrow 合法范围是 `>=19,<22`，正式列式产物必须是真实 Zstd Parquet。

```powershell
Set-Location C:\GitHub\YOLO-CV
uv sync --extra dev
uv lock --check
uv run python -c "import sys, pyarrow, torch; print(sys.version); print(pyarrow.__version__); print(torch.__version__)"
```

禁止用另一个 Python 环境的 `pytest` 冒充 `uv run pytest`。正式 release 还必须绑定 Python、Torch、CUDA build、GPU/driver 和实际导入的 Ultralytics 文件身份。

## 3. 代码边界

新增行为只允许位于：

- `stage1_sctsr_v4/`：合同、运行时、证据、评价、恢复和审计；
- `scripts/stage1_sctsr_v4/`：显式 CLI；
- `configs/stage1_sctsr_v4/`：机器合同和模板；
- `integrations/ultralytics/`：窄 ClassificationTrainer overlay；
- `tests/stage1_sctsr_v4/`：失败优先测试；
- `docs/stage1_sctsr_v4/`：实施和审查说明。

不得改写 `stage1_gapvalue240`、`stage1_dynamic_replay_v3`、已执行训练产物、旧 queue/release/assignment 或 `YOLOv11/ultralytics` 上游 learner。source-tree manifest 会覆盖所有 v4 可导入文件、overlay、依赖锁、任务书以及实际引用的上游文件。

## 4. 先验证，再生成

所有 CLI 都要求显式 `--output` receipt。非法输入返回非零并写机器错误；receipt 不得放进将被其自身索引的 artifact root。

基础验证：

```powershell
uv run python scripts/stage1_sctsr_v4/validate_contract.py `
  --contract configs/stage1_sctsr_v4/contract_v1.json `
  --arms configs/stage1_sctsr_v4/arms_phase1_v1.json `
  --schemas configs/stage1_sctsr_v4/schema_registry_v1.json `
  --output <receipt.json>

uv run python scripts/stage1_sctsr_v4/validate_assets.py `
  --registry configs/stage1_sctsr_v4/asset_registry_v1.json `
  --repository-root C:\GitHub\YOLO-CV `
  --output <receipt.json>

uv run python scripts/stage1_sctsr_v4/validate_schedule.py `
  --synthetic --training-seed 20260606 `
  --output <receipt.json>
```

资产验证会实际重算 path、bytes、SHA、row count、identity digest、split role 和互斥性；没有“跳过大文件 SHA”的正式入口。

图像内容身份另有强制门禁。正式资产包含 384,000 行 Zstd Parquet
`DATASET_CONTENT_LEDGER_v1.parquet`，覆盖 canonical train、val_model、
val_cal 和 val_op，不包含 test/blind。每行绑定 canonical relative path、
manifest SHA、标签、图像 bytes/SHA、尺寸、mode 和 format。训练机必须运行：

```powershell
uv run python scripts/stage1_sctsr_v4/validate_dataset_content.py `
  --registry configs/stage1_sctsr_v4/asset_registry_v1.json `
  --repository-root C:\GitHub\YOLO-CV `
  --dataset-root C:\GitHub\YOLO-CV\data\final_sewerml_dataset `
  --output <receipt.json>
```

这不是可选的运维检查：正式 trainer 在构造前会再次逐图哈希，closeout
再复验一次。文件名、标签、目录结构和 CSV 都正确，但任一图像字节错误时，
必须返回 `DATASET_CONTENT_MISMATCH`。不得用 `--ledger-only` 的结果替代正式
物理复验；该开关只用于快速检查 ledger/manifest 自身。

## 5. Identity pool 和 schedule

`build_identity_pools.py` 只允许四个登记角色：

- `T_STRESS`：冻结的 2.5% 压力集合，不是已验证 selector；
- `R1_GLOBAL_RANDOM`：完整 eligible base 上的全局随机，自然 T overlap 必须报告；
- `R2_MATCHED_RANDOM`：先排除 T，精确匹配 label、历史 dynamic bucket 和 OOF fold；
  仅对 filename-bucket surrogate `oof_group_id` 使用 owner-approved 最小偏移；
- `CURRENT_LOSS_HELD`：接口存在但第一阶段必须拒绝启用。

R2 matcher 在匹配前只接收预终端白名单投影。loss、confidence、RHO、gradient、forgetting、AUM、未来结果和 endpoint 字段不可见。算法同时排除 T sample ID、T image SHA 和候选内部 image-SHA alias；先耗尽每个四字段 exact cell，再只在同 label/dynamic/fold cell 内填充 379 个记录的 group displacement（原 quota 缺口 378，加 1 个内容别名排除），group TV 为 `0.12633333333333333`。三字段 quota 仍不可满足、出现第二字段 relaxation、replacement、内容重复或回用 T 时必须抛出 `R2_QUOTA_INFEASIBLE`。

`R2_U` 与 `R2_F` 不是两套各 3,000 IDs 的抽样。两者及
`T_TO_R2_AT_160` fallback 必须复用 selection seed `20260812`、3,000 unique 和
identity digest `957346D5...A0D194B` 的同一 pool；U/F 只允许 schedule 不同。

正式 pool 从登记资产派生分母；CLI 不接受人为 `--base-denominator`。生成后保存 manifest、五组 membership、候选全集 selection ledger 和 quota audit。

`build_schedule.py` 生成完整 E1–E200 计划。第一阶段只有八臂：

`NR, R1_U, R2_U, T_U, R2_F, T_F, T_TO_R2_AT_160, T_TO_NR_AT_160`

U 是 E121–E200 每 epoch 5/1000；F 是 E121–E160 每 epoch 10/1000、之后 0。U/F 的 pool、累计 occurrence 和逐 ID multiplicity 相同，只有 epoch distribution 不同。stop 臂不冒充 dose-matched；fallback 臂 E161–E200 明确进入 R2-U。

## 6. Common parent 和 branch

未来正式运行必须先由 release authority 提供：

- 有效签名 release manifest 和登记 trust key；
- 绑定同一 experiment ID、精确 release ID 与 canonical shared root 的 v2 claim registry；
- clean source-tree manifest；
- contract、arms、asset registry、runtime policy、formal seed registry；
- formal identity；
- 120,000 行 training identity manifest；
- trainer overrides；
- branch 所需 pool、schedule、lineage、E120 parent checkpoint 和 parent artifact index。

`run_common_parent.py` 只执行 E1–E120 no-replay。每个 training seed 只允许一个 parent。checkpoint 包含 model、EMA、optimizer、scheduler、AMP scaler、Python/NumPy/Torch CPU/CUDA RNG、epoch、global step 和全部输入身份。

`run_branch.py` 只执行 E121–E200。它先全量复核 parent run tree，再校验 lineage、pool、schedule 和训练 manifest，随后从相同 E120 全状态 checkpoint 分叉。

正式输入在训练前被复制到每个 run 的 `00_contract/` 与 `01_assets/`；closeout 会同时重算快照和原外部文件。训练后替换 release、source manifest、合同、registry、seed、identity manifest、pool 或 parent，都会使收尾失败。

## 7. Fixed base-step 训练语义

基础 DataLoader 始终为 120,000 个 canonical base occurrence、batch 128、每 epoch 938 base step。replay 不进入 dataset 长度，也不产生额外 optimizer step。

每个 base step 的顺序固定为：

1. base forward，沿用冻结 upstream base loss；
2. base backward；
3. 在独立 counter-domain RNG 中执行 replay forward；
4. replay CE 为 `sum(per_sample_ce) / 128`；
5. replay backward，保留对 parameter gradient 的贡献；
6. 恢复 replay 前的 Python/NumPy/Torch CPU/CUDA RNG；
7. 恢复所有 BatchNorm running buffers；
8. 一次 unscale、clip、scaler/optimizer step 和 scaler update；
9. 一次 EMA update；
10. scheduler、warmup 和 global step 只按 base-step clock 前进。

replay microbatch 不得超过实际 base batch 的 25%，包括尾 batch。OOM、world size>1、隐式梯度累积、自动减 batch、拆 step 或继续训练全部失败封闭。

## 8. 证据、事务和恢复

正式模式每个 epoch 写：occurrence、optimizer-step、exposure、telemetry 四类 Zstd Parquet，以及 checkpoint、identity、summary 和 receipt。每个 base/replay occurrence 和每个 optimizer step 都有独立行；candidate signal 始终标为非 utility。

事务流程是 `.inprogress -> 完整校验 -> generation manifest -> 原子 rename .complete -> append-only receipt -> recovery pointer`。kill、OOM、disk-full、半写 Parquet/JSON 或 receipt 损坏会进入 `09_quarantine`，不推进 canonical pointer，也不覆盖旧 generation。

resume 只读取最后完整 generation，复核 source/contract/assets/parent/RNG/history，并在 `<run>/10_resume_setup/epoch_XXXX.generation_1` 建立新的 upstream setup。E120/140/150/160/180/200 和滚动恢复 checkpoint 按合同保留。

## 9. 评价与收尾

正式 endpoint 固定为 E200、`val_op`、EMA、batch 128。真实图片只可通过登记 split component manifest 解析。模型必须同时返回 raw logits 和 softmax probability，二者逐行一致；prediction 绑定 checkpoint SHA、split bundle、sample-label digest 和 source identity。

`evaluate_checkpoint.py` 生成 FN=0..95 的 96 个 tie-safe 原始 frontier 点、normalized AUC、`TN_at_FN95`、`FN_at_TN68253`、两个独立 threshold、tie size 和 reachability。E120/140/150/160/180 只能标记 `NOT_FOR_METHOD_SELECTION`。

`validate_run.py` 是只读全树审计。`closeout_run.py` 还要求：

- summary completion audit；
- 从任务书解析出的精确 206 项 detailed self-audit；
- 每项原始 stdout/stderr、exit、bytes、SHA、源码/测试和风险；
- 206 项全部 PASS；
- 所有正式副作用字段为 JSON false。

closeout 唯一可发布状态是 `IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION`，不会创建训练 release 或 assignment。

## 10. Synthetic canary

```powershell
uv run python scripts/stage1_sctsr_v4/run_synthetic_canary.py `
  --artifact-root <new-empty-root> `
  --repository-root C:\GitHub\YOLO-CV `
  --training-seed 20260606 `
  --output <outside-root-receipt.json>
```

默认环境已有 PyArrow 时不应使用 portable fallback。Canary 真实执行 tiny model forward/backward/optimizer/EMA/checkpoint、八臂 schedule、96 点 frontier 和四种故障注入，但所有产物必须标记 `SYNTHETIC_NOT_SCIENTIFIC_RESULT`。

## 11. 本地真实图工程 canary

该入口只读取 `train_manifest.csv` 与 `normal_train_manifest.csv` 的前两条，逐图重算
bytes/SHA，加载已冻结 `yolo11l-cls.pt`，执行一个 4 张 base + 1 张 replay 的工程
microbatch。它不会构造正式 120,000-row DataLoader、不会生成 seed/release/assignment，
也不能估计模型效果：

```powershell
uv run --isolated --python 3.11 --extra dev `
  python scripts/stage1_sctsr_v4/run_engineering_canary.py `
  --repository-root C:\GitHub\YOLO-CV `
  --dataset-root C:\GitHub\YOLO-CV\data\final_sewerml_dataset `
  --artifact-root <NEW_EMPTY_ENGINEERING_CANARY_ROOT> `
  --training-seed 20260814 `
  --device cuda:0 `
  --output <OUTSIDE_ARTIFACT_ROOT_RECEIPT.json>
```

成功必须同时证明：真实训练图/权重 SHA、base/replay forward/backward、恰好一次
optimizer/EMA、replay RNG/BN 恢复、真实 Zstd Parquet、完整 checkpoint round-trip、
损坏 checkpoint 拒绝、半写 generation quarantine 和 FN=0..95 共 96 点。权威结果
必须写成 `ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT`，且七个正式副作用字段全为
JSON `false`。此 microbatch 的秒数不能外推正式 epoch 或 3090 campaign 工期。

## 12. 开发验收命令

```powershell
uv run pytest tests/stage1_sctsr_v4 -q
uv run pytest tests/stage1_dynamic_replay_v3 -q
uv run python -m compileall -q stage1_sctsr_v4 scripts/stage1_sctsr_v4 integrations/ultralytics tests/stage1_sctsr_v4
git diff --check
```

不能把 partial test、历史 receipt 或测试数量声明当作当前全量结果。当前硬阻断见 `KNOWN_BLOCKERS.md`，审查流程见 `INDEPENDENT_REVIEW_CHECKLIST.md`。
