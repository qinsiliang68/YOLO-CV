# P64 / S008 / T_U 部署备注

记录时间：2026-08-25（Asia/Shanghai）

状态：`DEPLOYED_AND_FORMAL_TRAINING_ACTIVE`

- 节点：`P64`（RTX 3090）
- run：`SCTSR_DISCOVERY_S008_T_U`
- arm / seed：`T_U` / `906427910`
- 训练输出：`D:\ssh\AI\artifacts\sctsr_v4_formal_discovery_s008_t_u_20260824_20a9558`
- 冻结源码：`20a9558e36b8782857f54670ae8cf79d3fb2554d`
- source tree digest：`97F3E579511066B4028226165D3000F0C217F2F8FE80E483908848B1AF37823B`
- 父模型 SHA-256：`4016DEDC5339B2076B6020163D163E334DAFC5848DA5044AA54B96FEBDFE4570`
- schedule digest：`2BEAB0016B677EB548FC5CA9288C29FDFDCD746777C058D4D98A7D5E85110CA4`
- T pool digest：`D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7`

部署没有复制 28 GiB 数据。P64 已有的 canonical 图像通过同卷硬链接形成训练视图，共 384,000 个链接；补齐的冻结 Ultralytics 源码仅 345 个文件、约 1.03 MiB。训练数据内容没有经网络重传。

## 首个完整 epoch 的独立核对

最终机器收据：`P64_E121_VALIDATION.json`，SHA-256 为 `6F512435C1A0CB2EEF8F5D65840EAA55E3ABA545998A5EBF2D4DD3C2188B439C`，状态 `PASS`。

- E121 generation digest：`14897598D5BF748C4E012D44F44D160255B233A5B9012EE0AF70DA8F7B14B0F1`
- E121 checkpoint SHA-256：`A57889B91356CCA3C77E4B3A8ADBCEA9F0E8C197FBB9F32D979C70251A5FEF6F`
- generation manifest 约束的全部文件 bytes/SHA 均匹配。
- occurrence 共 120,600 行：120,000 个 base occurrence，600 个 replay occurrence，938 个 base step。
- 600 个 replay ID 全部唯一，ID 集合与 E121 冻结 schedule 完全一致；按 seed/epoch 重建的 replay step-slot 映射也逐槽一致。
- `selection_policy=T_STRESS`、`identity_pool_id=T_STRESS_POOL`，样本级 `replay_role` 与标签语义一致。
- 收据生成时 E122 已完成，E123 正在运行。

两份先行验证收据的 `FAIL` 被原样保留：第一份误把 schedule 原始 ID 列表顺序当成运行时 step-slot 顺序；第二份误把样本级 `replay_role` 当成池策略。它们是验证器假设错误，不是训练或数据错误；最终收据已按冻结运行时代码重建 step plan，并用 `selection_policy` 和 `identity_pool_id` 核实 T_STRESS。

## 本轮代码修复

- `db7e25906bb0321cb7a975f2d7b2ab4c79a8eabe`：避免重复 SCTSR 数据集验证。
- `408d4b9965930639a43a06d3245ec3b5a09050f4`：允许保留且受约束的 Ultralytics role cache。
- `d58967c9a94807139febb16947a1d8b020f18e1f`：在物理数据扫描前确认完整冻结 Ultralytics 依赖树。
- `5f75febc22cee2ff6c76488b5ba3125ca764ae0b`：formal trainer setup 强制离线，禁止辅助权重下载重试拖延启动。

相关定向测试为 `14 passed`，compileall 通过。失败 claim generation、部署脚本、最终收据和每个 epoch 的 ledger/checkpoint 均继续保留在 P64；没有生成实验有效性结论，也没有删除中间产物。
