# SCTSR v4 正式训练操作总则

## 1. 权限和当前状态

本手册描述未来 release 后怎样运行，不是当前训练授权。当前所有 arm 的
`formal_state=HELD`。R2 addendum 已由 owner 批准并完成本地物化验证，但正式
key、8+14 training seeds、release、逐 job token 和 shared claim registry 尚未生成。

只有 release authority 可以：

- 验证已批准科学 addendum 的 SHA 和 source commit；
- 冻结 discovery/confirmation seeds；
- 签名 matrix release；
- 为每个 logical process 签发一次性 execution token；
- 指定机器、GPU、输出根和允许的 START/RESUME。

训练机 AI/operator 只能执行已签名的精确 job。它不能改 arm、seed、schedule、
pool、parent、data root、output root 或 retry 语义，也不能自行生成替代 token。

## 2. 必须按顺序完成的阶段

### Phase 0：代码和科学规格冻结

1. 独立复审 R2 addendum 实现与真实 pool digest；
2. code review 无 P0/P1；
3. clean commit 生成 source-tree manifest；
4. contract/arms/runtime/registry/runbook manifest 全部冻结；runbook manifest 必须由
   `build_runbook_manifest.py` 对部署清单中的每份 Markdown逐字节生成；
5. test/blind 继续密封。

任何源文件、配置、文档或账本变化都会改变 source/asset/runbook digest，必须重新
走本阶段。不得在机器上 `git pull` 后继续旧 job。

### Phase 1：资产和环境预检

在每台机器的 exact checkout 执行：

```powershell
uv lock --check
uv sync --extra dev
uv run --python 3.11 python scripts/stage1_sctsr_v4/validate_contract.py `
  --contract configs/stage1_sctsr_v4/contract_v1.json `
  --arms configs/stage1_sctsr_v4/arms_phase1_v1.json `
  --schemas configs/stage1_sctsr_v4/schema_registry_v1.json `
  --output <machine_receipts>/contract.json
uv run --python 3.11 python scripts/stage1_sctsr_v4/validate_assets.py `
  --registry configs/stage1_sctsr_v4/asset_registry_v1.json `
  --repository-root <exact_checkout> `
  --output <machine_receipts>/assets.json
uv run --python 3.11 python scripts/stage1_sctsr_v4/validate_dataset_content.py `
  --registry configs/stage1_sctsr_v4/asset_registry_v1.json `
  --repository-root <exact_checkout> `
  --dataset-root <canonical_dataset_root> `
  --output <machine_receipts>/dataset_content.json
```

每条命令必须保存 stdout、stderr、exit code、receipt bytes/SHA。只要有一条非零，
该机器不进入 job pool。

`--dataset-root` 指 canonical Sewer-ML root，不是 Ultralytics classification view。
后者由 `trainer_overrides.data` 指定，必须在同一卷预先原子物化为 hardlink-only
`train/no_target`、`train/target_defect`、`val/no_target`、`val/target_defect`。正式
runner 分别绑定 canonical root 与 materialized root；每个 loader row 都保存两个
resolved path、bytes/SHA、physical file identity 和 `samefile_as_canonical=true`。
独立 copy、symlink/junction、额外文件或训练期间替换均在 step 0/RESUME/endpoint/
completion 边界失败封闭。classification view 的绝对路径必须进入 overrides、
run-intent 和 execution token，不能由机器临时搜索或复用相似缓存。

### Phase 2：pool 和 schedule 物化

Pool 由 release authority 在独立冻结目录一次性生成，不由各训练机器各自随机生成。
每个 pool 保存 manifest、membership Parquet、selection ledger、quota audit 和 CLI
receipt。R2 只能按已批准 policy ID、selection seed `20260812` 和 expected digest
成功；`R2_U`、`R2_F` 与 fallback 必须读取同一个 pool artifact，不得在训练机重抽。

每个 arm/seed 的 E1-E200 schedule 由已冻结 pool 物化。训练机只读 schedule，
不得重新抽样或根据本机性能改变 slots。

### Phase 3：training identity manifest

Common parent 使用 120,000 行无 pool annotation 的 manifest。每个 branch 使用
exact schedule 和所需 pool 生成 manifest。manifest 必须覆盖 canonical base
恰好一次，不能新增/漏掉身份，也不能改变 label、dynamic、fold/group 或 pool
membership。

### Phase 4：common parent

每个 training seed 只执行一个 `COMMON_PARENT_NR`：E1-E120、无 replay。运行前
确认 output root 不存在；已存在的目录不是“可继续”的理由，只有显式 RESUME token
和完整 receipt chain 才能恢复。

成功条件不是 Python 退出 0，而是：120 个连续 complete generation、120 个完整
checkpoint、receipt chain、recovery pointer、parent receipt、artifact index 和
read-only `validate_run` 全部一致。

### Phase 5：八个 branch

同 seed 的八个 branch 必须绑定同一个 parent checkpoint SHA 和 parent artifact
index。每个 branch E121-E200 独立输出，不能共享可写目录。NR 也必须作为 branch
执行，不能把 parent 延长后冒充 NR child。

### Phase 6：fixed endpoint 和评价

Branch E200 完成后，由 runner 用 E200 EMA 和 val_op 发布 prediction/frontier。
不得手工运行 `best.pt`，不得从轨迹图选择更好 epoch。然后运行 `validate_run`，
最后由独立 closeout 流程核对全部 source/assets/receipts。

### Phase 7：跨 seed 汇总

只有同一对 contrast 的两臂在同 seed、同 parent、同 endpoint 均 canonical complete
时才形成 paired observation。缺失 pair 不得用其他 seed 或 arm 均值补齐。探索和
确认 seeds 分离；只有探索规则通过，才签发确认 release。

## 3. Job 的不可变输入

每个 job 必须有且只有一组：

- exact clean checkout commit/source digest；
- Python 3.11 或 3.12 锁定环境；
- one numeric CUDA device；
- formal identity；
- signed matrix release；
- signed one-use execution token；
- run-intent acknowledgement；
- runbook manifest；
- canonical asset registry/data ledger；
- trainer overrides；
- 120k identity manifest；
- parent job：initial checkpoint；
- branch job：E120 parent + parent index + lineage + schedule + pool；
- new empty output root；
- receipt path在 output root 外部。

训练机不得自动选择最新 checkpoint、最新 schedule、同名 pool、空闲 GPU 或相似输出
目录。具体值由 token/acknowledgement给出。

## 4. 13 台 3090 的分配原则

最多 12 台并行，1 台作为故障替补；job 可简单随机分配，不需要复杂 GPU 锁或调度器。
正确性来自 one-use token、共享 v2 claim registry、numeric CUDA device 和 logical-job
fence，而不是机器序号。common-parent 初期只有已发布 seed 数量那么多可并行任务。

- 每个 process 单 GPU，world size 固定 1；
- 一块 3090 同一时刻最多一个 formal process；
- common parent 优先分配并完成/验证，再释放同 seed branches；
- 同 seed branches 可在不同机器执行，但 parent bytes 必须 SHA 相同；
- output 建议写本机 NVMe 后原子复制到统一只读 evidence store，复制完成需双端 SHA；
- 共享 execution-claim registry 必须是所有机器看到的同一个 canonical UNC root，
  不能各自复制一份 descriptor；
- 每个 token 只允许一个进程 claim；并发抢同 token 时必须恰好一台成功；
- 不为追求 GPU 利用率而在 R2/资产/release 未通过时先跑部分正式 arm。

总工作量：每 seed 120 parent epochs + 8×80 branch epochs = 760 epoch-equivalents。
Discovery 8 seeds 为 6,080；若门槛通过，confirmation 14 seeds 另需 10,640。
运行时间用工程 canary 实测 `seconds_per_epoch` 计算：

`wall_hours ≈ epoch_equivalents × measured_seconds_per_epoch / 3600 / effective_parallel_GPUs`

必须另加 parent-before-branch barrier、dataset preflight、E200 val_op、复制/closeout 和
故障余量。未获得 3090 全量 epoch benchmark 前，不得把估算写成承诺。

## 5. 磁盘和保留

合同保留每个 epoch 完整 checkpoint 和所有 occurrence/step/exposure/telemetry，不能
训练后只留 `last.pt`。现有真实工程 canary 的单 checkpoint 约 154.74 MB，但正式
optimizer state 和全量 ledger 可能更大。每 seed 有 760 个 epoch generation，
仅按 154.74 MB 下界已经约 117.6 GB，尚未计全量 occurrence、prediction、复制和
安全余量。

部署前应按至少 200 GB/seed 的规划值分配，并以首个完整正式工程 canary 的最大
generation bytes 更新。13 台机器建议每台至少保留 500 GB 独占可用 NVMe；统一
evidence store 至少预留 3 TB discovery 空间，confirmation 前根据 discovery
实测重算。空间不足必须在 job 前阻断，不能等训练中途自动删旧 epoch。

## 6. 允许的停止

训练进程只可因以下原因停止：

- operator 明确中止；
- OOM/NaN/nonfinite gradient；
- disk preflight/full/write failure；
- source/asset/data/parent/schedule/seed/token mismatch；
- telemetry/evidence/checkpoint/receipt 无法原子发布；
- GPU/driver/process kill；
- release/token expiry 或重复 claim；
- blind/test/forbidden role 请求。

停止后不得改配置继续。只有 `FAILURE_AND_RECOVERY.md` 允许的 quarantine + RESUME
流程可以恢复。

## 7. Canonical success

一项 job 只有同时满足以下条件才是 complete：

- 预期 epoch 全部连续；
- 每 epoch base rows=120,000、base steps=938；
- planned/actual replay 守恒；
- checkpoint/Parquet/summary/generation manifest 全部 SHA 验证；
- receipt chain、pointer、artifact index 连续；
- parent/lineage/source/contract/assets/RNG 身份一致；
- branch 有 E200/EMA/val_op prediction 和 96 点 frontier；
- validate_run PASS；
- closeout PASS；
- test/blind false；
- best.pt false；
- method-effectiveness claim false。

exit 0、GPU 跑满、出现 `last.pt`、目录很大或 README 写“完成”均不是上述证明。
