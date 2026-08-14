# SCTSR v4：训练机和审查者先读

## 当前判定

本目录描述的是隔离的 SCTSR v4 二分类门控实验，不是仓库顶层说明中的六类
multi-label 主训练任务，也不覆盖任何历史 120/240-run 结果。SCTSR 的训练身份只能
来自冻结 `contract_v1.json`、`asset_registry_v1.json`、taskbook 和签名 release；
不得根据顶层 README 猜测 dataset、labels、trainer 或 endpoint。

截至本 runbook 冻结前：

- 代码实现和训练前 review 正在完成；
- 正式训练尚未授权；
- R2 四字段 exact quota 在真实资产上缺 172 strata / 378 occurrences；
- owner 已于 2026-08-15 批准只放宽 filename-bucket surrogate `oof_group_id`，
  其余三字段 exact、3,000 unique、T overlap=0 保持不变；
- `R2_U`、`R2_F` 和 fallback 必须复用 digest 为
  `075FC31...19B6BECC` 的同一 R2 pool；
- 正式 seed、matrix release 和 execution token 尚未发布；
- `val_target` 不存在，故 A/gradient-alignment保持 HELD，但不阻断第一阶段
  timing/stop/fallback 代码审查；
- test/blind 未打开；
- 没有任何 SCTSR 方法有效性结论。

因此，训练机现在只允许执行 unit/integration/synthetic/engineering canary 和只读
验证。R2 规格阻断已经由 owner addendum 解除，但这不等于训练授权；release
authority 仍必须把 addendum、可物化 R2 pool 和新的 contract/source identities
一并冻结，并另行签发 seed、release、token 和 shared claim registry。

## 阅读顺序

训练机 AI、operator、release authority 和 reviewer 都按以下顺序阅读：

1. `EXPERIMENT_INTENT.md`：研究问题、T、八臂、可推出和不可推出的结论；
2. `FAIRNESS_CONTRACT.md`：唯一允许变化的 treatment 和直接作废条件；
3. `ASSET_IDENTITY_LEDGER.md`：数据、权重、OOF、T 和 image-byte identity；
4. `TRAINING_OPERATIONS_MANUAL.md`：端到端阶段、10 台 3090 和成功定义；
5. `MACHINE_RUNBOOK.md`：单 job 参数、确认书、START/RESUME 命令；
6. `FAILURE_AND_RECOVERY.md`：OOM/kill/disk/partial/receipt恢复规则；
7. `ARTIFACT_AND_SCHEMA_GUIDE.md`：每个 ledger/checkpoint/prediction/frontier字段；
8. `DEPLOYMENT_CHECKLIST.md`：发布前逐项机器验收；
9. `SPECIFICATION_CHANGE_REQUEST_R2_INFEASIBLE.md` 与 canonical
   `SCTSR_R2_MATCHING_ADDENDUM_20260815.md`：原不可行证据和已批准决定；
10. `INDEPENDENT_REVIEW_CHECKLIST.md`、`KNOWN_BLOCKERS.md` 和最终 review report。

任务书仍是实施范围的上游权威规范；本 runbook 是对已经实现代码和本轮 review
修复后的操作解释。若二者冲突，不能由训练机自行选择：必须登记冲突，停止，并由
owner/release authority确认是否已被 canonical addendum 替代。Markdown声明不能覆盖机器 validator。

## 一个 job 何时算开始、完成或作废

- “开始”：合法 one-use token已在共享 registry原子 claim，并真正进入 formal epoch；
- “工程可跑”：canary完成，但仍是 `ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT`；
- “完成”：全部连续 generations、receipts、checkpoint/Parquet SHA、固定 endpoint、
  `validate_run` 和独立 closeout均 PASS；
- “科学可分析”：同 seed 的预注册 pair均 canonical complete；
- “方法有效”：只能由未见 seed 的正式配对干预和预注册统计规则给出；
- “作废”：任何身份、split、公平预算、base step、RNG/BN、恢复或 endpoint合同失败。

进程 exit 0、GPU utilization高、loss下降、存在 `last.pt`、文件很多或日志写
“success”均不能改变上述定义。

## 绝对禁止的自动化

训练机不得自动：

- 找 Downloads/Desktop或相邻目录中的“最新/相似”文件；
- 选择空闲 GPU 后改写 signed device；
- 超出 addendum 再放宽 R2、允许 replacement或复用 T；
- 减 batch、改 workers/imgsz/accumulation或在 OOM 后继续；
- 从 best.pt/val_op/轨迹图挑 checkpoint、停止点或阈值；
- 复制 execution-claim registry以绕过一次性 claim；
- 修改 receipt/JSON、重命名 partial generation或删除失败证据；
- 打开 test/blind；
- 把 candidate signal、synthetic或engineering结果称作 utility evidence。

## 运行前最后一句话

执行者必须能够用自己的话说清楚：本 job 的 treatment、comparator、唯一允许差异、
exact parent/seed/data/schedule/pool identities、失败停止规则和完成证据。随后才可生成
与 exact token绑定的 `RUN_INTENT_ACKNOWLEDGEMENT.json`；确认书不是权限，签名 token
也不是理解证明，二者必须同时通过。
