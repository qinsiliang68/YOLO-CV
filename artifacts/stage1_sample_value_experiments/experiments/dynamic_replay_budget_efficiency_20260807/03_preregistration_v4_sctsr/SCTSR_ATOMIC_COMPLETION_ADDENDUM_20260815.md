# SCTSR v4 原子完成与终点恢复附录

状态：`CODE_IMPLEMENTED_NOT_TRAINING_AUTHORIZATION`

本附录修复旧实现的 false-COMPLETE 窗口：旧 branch 在 E200 训练结束后先写
`FORMAL_BRANCH_COMPLETE`，随后才运行 `val_op` endpoint、补 terminal receipt 和重建 artifact index。进程在这些步骤之间退出时，会留下“已完成”字样，同时旧 resume 又因 E200 已存在而拒绝继续。

## 1. 新完成状态机

正式 common parent 和 branch 的根级 run-state 文件不再单独宣告 canonical completion：

- parent E120 完成后：`FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION`
- branch E200 完成后：`FORMAL_BRANCH_EPOCHS_COMPLETE_PENDING_ENDPOINT`
- branch endpoint 完整且重新验证后：`FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT`

唯一 canonical completion fact 是最后原子写入的：

`FORMAL_COMPLETION_RECEIPT.json`

其状态仅可为：

- `FORMAL_PARENT_COMPLETE`
- `FORMAL_BRANCH_COMPLETE`

因此，看到 E120/E200 checkpoint、根级 `PARENT_RECEIPT.json` / `BRANCH_RECEIPT.json`、`RUN_MANIFEST.json` 或 endpoint 的部分文件，都不能单独判断 run complete。

## 2. completion marker 的前置条件

`publish_formal_completion` 在写 marker 之前必须逐项验证：

1. run role、run ID、arm、training seed 在调用参数、run-state 和 `RUN_MANIFEST.json` 三方一致；
2. parent 固定 checkpoint 为 E120，branch 固定 checkpoint 为 E200；
3. `ARTIFACT_INDEX_GENERATIONS.json` 为完整 epoch-generation index；
4. `ARTIFACT_INDEX.json` 与当前实际文件逐路径、bytes、SHA 完全一致；
5. branch 必须已有 `08_receipts/FORMAL_ENDPOINT_RECEIPT.json`；
6. run-state 的 epoch receipt-chain digest 已冻结；
7. completion marker 此前不存在。

marker 绑定 run-state、run manifest、generation index、exhaustive artifact index 和 endpoint receipt 的 SHA，以及最终 checkpoint、epoch receipt digest 和身份字段。任一被绑定文件之后改变，`validate_formal_completion` 必须失败。

为避免自引用，`ARTIFACT_INDEX.json` 明确排除自身和最后写入的
`FORMAL_COMPLETION_RECEIPT.json`；completion marker 反向绑定 artifact index 的 SHA 和 digest。

## 3. terminal-epoch finalization-only RESUME

`inspect_formal_resume_context` / `prepare_formal_resume_context` 新增显式
`allow_terminal_epoch_for_finalization`。只有正式 runner 将该参数设为 true；普通 resume 保持“terminal epoch 不可继续训练”。

当 E120/E200 已完整提交但 completion marker 不存在时：

1. 只读重验全部 epoch generation、checkpoint、RNG、receipt chain 和 replay history；
2. 签发并 claim 精确绑定 terminal checkpoint/receipt 的新 RESUME token 与 fence；
3. 允许使用 `resume_epoch=121`（parent）或 `201`（branch）恢复控制面；
4. 不再执行任何训练 epoch；
5. 重新建立/核验 logical index、run manifest、endpoint 和 final index；
6. 最后原子发布 completion marker。

若 marker 已存在，任何 START/RESUME 均必须拒绝。

## 4. endpoint 恢复

branch finalization 在新 attempt 获得最新 logical-job fence 后处理旧 endpoint：

- 六项 endpoint 文件（split bundle、prediction parquet/summary、frontier parquet/summary、endpoint receipt）全部存在且重新验证通过：直接复用，不重复执行 120k 图像推理；
- 任一文件缺失或完整集合验证失败：把所有已存在文件逐项移动到唯一的
  `09_quarantine/formal_endpoint.incomplete.<timestamp>.<uuid>/`，记录原路径、目标路径、bytes 和 SHA，然后从空 canonical 路径重建；
- quarantine 只保存失败 attempt 的证据，不计入科学 endpoint；
- 已存在 canonical completion marker 时禁止移动或重建 endpoint。

## 5. schema

- `formal_parent_receipt`: `stage1.sctsr.formal_parent_receipt.v3`
- `formal_branch_receipt`: `stage1.sctsr.formal_branch_receipt.v3`
- `formal_completion_receipt`: `stage1.sctsr.formal_completion_receipt.v1`
- `formal_endpoint_quarantine`: `stage1.sctsr.formal_endpoint_quarantine.v1`

## 6. 失败优先与回归证据

本回滚单元的失败优先测试先证明：

- 缺少新 completion 模块时测试 collection 失败；
- branch 没有 endpoint receipt 时不能生成 completion marker；
- terminal epoch 默认不可 resume，仅 finalization recovery 可得到 121/201；
- `formal_training.py` 不得出现 branch complete 状态；
- runner 调用顺序必须是 endpoint 在前、completion marker 在后；
- 半写 endpoint 必须 quarantine；
- 完整 endpoint 必须验证后复用；
- marker 后修改 run-state 必须被检测。

全量测试只能证明实现与合成/单元合同一致，不是 SCTSR 方法有效性的科学证据，也不签发正式 release、seed、assignment 或训练授权。
