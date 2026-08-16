# SCTSR v4 失败、隔离与恢复合同

## 1. 核心原则

失败不是“少一点证据的成功”。只要 epoch 的 checkpoint、Parquet、summary、SHA、
generation manifest、receipt 或 pointer 任一项未完成，它就不能成为 canonical
epoch。恢复只能从最后一个连续、完整、可重算的 generation 开始。

禁止：

- 手工重命名 `.inprogress` 为 `.complete`；
- 复制另一个 run 的 checkpoint/receipt；
- 修改 JSON 让 validator 通过；
- 删除失败证据后从同一 token 重跑；
- `--resume` 时改变 seed、arm、parent、source、contract、asset、schedule、pool、
  optimizer 或 output root；
- 用 `last.pt`、`best.pt` 或目录中“最新文件”推断恢复点。

## 2. Epoch commit point

单 epoch 的顺序固定为：

1. 创建唯一 `epoch_EEEE.generation_G.inprogress`；
2. 写 transaction identity；
3. 写 occurrence/step/telemetry/exposure 临时 Parquet；
4. 原子写完整 checkpoint 和 summary；
5. 校验 required paths、Parquet footer/Zstd/schema/rows、JSON、bytes/SHA；
6. 写 generation manifest；
7. 原子 rename 为 `.complete`；
8. 原子追加/替换完整 receipt chain；
9. 更新 artifact index；
10. 更新 rolling recovery pointer。

**receipt row 是 canonical commit point**。只有 `.complete` 但没有合法 receipt 的
generation 是 orphan，reconciliation 必须 quarantine；有合法 receipt 但 secondary
index/pointer 丢失时，可以从 receipt+immutable generation 重建 secondary metadata，
不能反过来相信 pointer 而忽略 receipt。

## 3. 失败分类与动作

| failure | 当前进程动作 | 是否允许 resume | 必需证据 |
| --- | --- | --- | --- |
| source/asset/data SHA mismatch | trainer 构造前拒绝 | 否；修复后新 token | FAIL receipt、expected/observed |
| R2 approved three-field cell仍不足或 digest不符 | pool generation 拒绝 | 否；新 owner addendum | shortage/displacement audit |
| duplicate/expired/invalid token | claim 前拒绝 | 否；authority 新 token | claim registry evidence |
| claim 后 resume/trainer/finalization setup 异常 | 当前 fence 立即写 FAILED terminal receipt | 是；新 RESUME token、generation+1 | heartbeat、terminal receipt SHA、原始异常 |
| output root 已存在 | START 拒绝 | 仅合法 RESUME | root inventory |
| OOM | 当前 epoch abort | 是，固定 batch/steps 不变 | OOM code、quarantine |
| NaN/nonfinite/AMP overflow | 当前 epoch abort | 依原因审查后 | step receipt、scaler/grad state |
| process kill/power loss | 留 partial | 是 | process/telemetry end、partial inventory |
| disk full/write error | abort/quarantine | 扩容后 | disk telemetry、write error |
| telemetry stop failure | 继续关闭两 writers 和 transaction abort | 视主异常 | exception notes |
| half Parquet/JSON/checkpoint | quarantine | 是 | file validation error |
| receipt chain corrupt | 停止，不选“最近可读行” | 仅可由 immutable prefix重建且经审查 | chain audit |
| parent/source/RNG mismatch on resume | resume 拒绝 | 否 | mismatch receipt |
| endpoint publication failure | training prefix保留，run 未 complete | 修复发布链后独立审查 | endpoint partial inventory |
| blind/test access request | 立即拒绝 | 否 | forbidden-role receipt |

## 4. 失败清理必须保留根因

Epoch 主异常是第一原因。即使 telemetry stop、occurrence writer abort、step writer
abort 或 quarantine rename 再失败，也必须全部尝试。清理错误作为 exception note
附在主异常后，不得替换主异常。

Recorder 构造本身也在 transaction `try` 内：第一 writer 成功、第二 writer 或
telemetry 启动失败时，已打开组件仍需关闭，再 quarantine transaction。

## 5. Reconciliation

每次恢复前先只读枚举：

- `.inprogress`；
- `.complete`；
- quarantine；
- receipt chain；
- artifact index；
- rolling pointer。

规则：

- unreceipted complete → quarantine；
- receipted complete → 保留；
- receipted complete 缺 index/pointer → 从 receipt/generation 重建；
- receipt 指向缺失/变更 generation → fail closed；
- 多 generation 同 epoch 只有 receipt 链明确的 canonical generation 可保留；
- quarantine append-only，不覆盖同名旧失败；
- 每次 reconciliation 产出自身 receipt 和 before/after SHA。

## 6. Resume preflight

RESUME 只能在以下全部 PASS 后执行：

1. run 尚无 terminal manifest/receipt；
2. formal identity、authorization snapshot、execution evidence可重算；
3. contiguous epoch prefix 从 role 起点开始无缺口；
4. 每个 generation 文件、SHA、schema、rows 验证；
5. checkpoint payload 的 model/EMA/optimizer/scheduler/scaler/RNG/epoch/global step
   完整；
6. occurrence ledger 重放得到的 replay history 与 summary/checkpoint 一致；
7. receipt chain、index、pointer 同一末端；
8. source/contract/asset/parent/seed/arm 不变；
9. 新 prepared trainer 的科学 binding 与原 binding 相同；
10. dataset-content 384,000 图像复验相同；
11. free disk ≥ `max(minimum_resume_free_bytes, 1.25 × largest_generation ×
    remaining_epochs + 2 × last_checkpoint_bytes)`；
12. 有 action=`RESUME` 的新一次性 token 和 acknowledgement。

Resume epoch 固定为 `last_complete_epoch + 1`。不能由 CLI 自由选择。setup root
固定在：

`<run>/10_resume_setup/epoch_EEEE.generation_1`

该 setup 只是恢复准备证据，不能冒充 epoch generation。

## 7. OOM 与公平性

OOM 后禁止：

- 减 batch；
- 改 imgsz；
- 增 gradient accumulation；
- 减 replay microbatch；
- 拆成两次 optimizer step；
- 禁用 AMP；
- 跳过本 epoch 的 replay；
- 换另一型号 GPU 后不改 runtime identity。

这些变化都会改变 treatment dose 或 base process。合法动作只有：结束当前 attempt，
保留失败证据；若同规格在满足 runtime contract 的 3090 上可运行，由 authority 签发
绑定新机器/runtime 的 RESUME token。

## 8. Disk full

正式证据全量保留，不能用自动清理“修复”磁盘。Disk full 前兆来自每秒 telemetry
和 epoch `disk_bytes_written`。达到预注册安全线时应在下一个 epoch 前停止，而不是
写到半 checkpoint。

扩容后：

1. 验证旧 volume 内容 SHA 未变；
2. quarantine partial generation；
3. 重算 remaining capacity；
4. output root 不参与科研 logical key，但首个 START fence 会把它固定为授权存储根；
   RESUME 必须使用原 root。迁移只能使用未来 owner-signed migration receipt，不能
   通过换目录隐式重开同一任务；
5. 不删除 completed generations。

## 9. Endpoint partial failure

E200 training generation完成不代表 branch complete。若 prediction/frontier 发布失败：

- 保留 E121-E200 transaction prefix；
- 不写/不认可 terminal branch receipt；
- partial endpoint 进入独立 quarantine；
- 重试必须仍绑定相同 E200 checkpoint SHA、EMA、val_op manifest和 source；
- 不能改成 MODEL、best.pt、较早 epoch 或另一 split；
- 重试后重新生成 exhaustive artifact index并做 closeout。

## 10. 恢复后必须证明

- 原 complete generation bytes 未变；
- quarantine append-only；
- 新 epoch 从正确 RNG/global step/replay history开始；
- base order/augmentation 与无故障反事实相同；
- optimizer/scheduler/EMA update counts 连续；
- no duplicated/skipped replay occurrence；
- receipt chain 和 pointer只增加合法新行；
- final endpoint和无故障路径的身份合同相同。

“loss 接得上”或“最终精度接近”不能替代这些证明。
