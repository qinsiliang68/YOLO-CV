# 当前阻断、矛盾与非声明

状态不是“训练就绪”，也不是“方法有效”。当前实现保持 `IMPLEMENTED_NOT_FORMALLY_RUN`；严格 taskbook self-audit 在所有实施项真正通过前必须输出 `SELF_AUDIT_FAIL`。

## 1. 正式 release 和 seed 阻断

`formal_release_trust_v1.json` 没有登记 authorized key，仓库也没有未来签名 release。`seed_registry_schema_v1.json` 的 discovery/confirmation seed 为空，状态为 `FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE`。因此 formal runner 必须在构建 trainer、生成 assignment 或读取正式 split 前拒绝。

## 2. R2 旧规格矛盾已解决，不再是当前阻断

内容唯一派生 T 为 3,000 IDs/3,000 image SHA，digest 是 `D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7`。在排除 T 后，按 `(label, dynamic bucket, OOF fold, oof_group_id)` 精确匹配仍有 172 个 joint strata 缺口，累计短缺 378 个 occurrence。

2026-08-15 owner 已批准唯一推荐修订：保持 3,000 unique、零 overlap 和
`(label, dynamic bucket, OOF fold)` exact，四字段先按可用容量填满，再在相同三字段
cell 内随机填充不可避免的 378 个 deficit；group TV 为容量下界 0.126。正式 matcher
已显式激活 content-disjoint v2 政策，真实资产物化得到 R2 identity digest
`A6DAA20A70F02B30D15B7C3E4079EA86903051AEED264F53E0A104A4C1AA80B6`，
selected-content digest
`A48B721CA37AD66D65B8C5972C5AE66C328C09194BA3C8C22C19B8FECE40F819`。
`R2_U`、`R2_F` 和 fallback 必须复用该同一 pool。任何第二字段放宽、replacement、
T sample/content overlap 或标签替换仍失败封闭。当前 3,000 个 R2 identity 对应
3,000 个唯一图像 SHA，T/R2 image-SHA overlap=0。详见 `SPECIFICATION_CHANGE_REQUEST_R2_INFEASIBLE.md`
、canonical `SCTSR_R2_MATCHING_ADDENDUM_20260815.md` 及
`SCTSR_DATA_CONTENT_AND_T_REPAIR_ADDENDUM_20260815.md`。

## 3. v3 回归口径矛盾

任务书 SA-266 和附录 C 要求 `tests/stage1_dynamic_replay_v3` 至少 231 passed。2026-08-15 在当前冻结 clean checkout 上以 Python 3.11 可复现的是 `181 passed, 3 skipped`；1 项需要登记 Desktop mirror，2 项需要本地 21 份 literature anchor source files。这些都是本地 evidence integration skips，不是 SCTSR 训练逻辑失败。现有 tracked v3 测试树仍不能支持 231 passed 的历史口径。

这不是 SCTSR v4 测试失败，但在任务书未修订或缺失测试未恢复前，SA-266 必须 FAIL，不能把历史“231 passed”声明当作当前执行证据。详见 `SPECIFICATION_CHANGE_REQUEST_V3_231_BASELINE_MISMATCH.md`。

## 4. val_target 不存在

当前 registry 只有 train/OOF/reference、val_model、val_cal、val_op；没有独立、群组隔离且 SHA 冻结的 `val_target`。A/gradient alignment 只能返回 `BLOCKED_BY_VAL_TARGET`，不得生成 arm、assignment 或正式 gradient artifact。

## 5. Blind/test 保持密封

SCTSR v4 runner、split-bundle CLI 和 endpoint publisher 都没有 test/blind role。它们要等方法、代码、阈值、停止规则和多重比较完全冻结后才能由独立授权打开。

## 6. 科学结果尚不存在

没有 SCTSR v4 正式 training seed、parent、branch、prediction 或 paired intervention。Synthetic canary 只验证机制，语义固定为 `SYNTHETIC_NOT_SCIENTIFIC_RESULT`。因此不能声称：

- SCTSR 优于 R1/R2/current-loss/no-replay；
- timing、stop 或 fallback 有收益；
- T 是有效 selector；
- Q/R/A/D 任一信号有 utility；
- FN95 安全前沿已改善。

## 7. 外部科学审计仍有独立阻断

BudgetedReplay 报告所称的三个源码载体在既有现场审计中为 `REPORT_ONLY_SOURCE_MISSING`。这不阻止隔离 v4 代码施工，但阻止把专家仓库逐行审计标成完成，也不能用报告摘录代替源码。

## 8. 严格解除顺序

1. 对 R2 addendum 实现提交做独立复审并冻结新 source manifest；
2. owner 对 v3 231 基线矛盾作修订或恢复缺失测试；
3. 3090 正式规格单 epoch engineering benchmark 通过；
4. release authority 登记 key、8+14 training seeds、签名 release、逐 job token 和
   全部训练机共享的 v2 claim registry；registry 必须绑定 exact experiment/release；
5. 只在所有机器 preflight PASS 后另行授权正式训练；
6. blind/test 继续密封。
