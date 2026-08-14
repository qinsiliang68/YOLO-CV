# 当前阻断、矛盾与非声明

状态不是“训练就绪”，也不是“方法有效”。当前实现保持 `IMPLEMENTED_NOT_FORMALLY_RUN`；严格 taskbook self-audit 在所有实施项真正通过前必须输出 `SELF_AUDIT_FAIL`。

## 1. 正式 release 和 seed 阻断

`formal_release_trust_v1.json` 没有登记 authorized key，仓库也没有未来签名 release。`seed_registry_schema_v1.json` 的 discovery/confirmation seed 为空，状态为 `FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE`。因此 formal runner 必须在构建 trainer、生成 assignment 或读取正式 split 前拒绝。

## 2. R2 真实资产不可行

冻结 T 为 3,000 IDs，digest 是 `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`。在排除 T 后，按 `(label, dynamic bucket, OOF fold, oof_group_id)` 精确匹配有 172 个 joint strata 缺口，累计短缺 378 个 occurrence。

实现正确行为是抛出 `R2_QUOTA_INFEASIBLE`。2026-08-14 的冻结资产审计已提出唯一推荐修订：保持 3,000 unique、零 overlap 和 `(label, dynamic bucket, OOF fold)` exact，四字段先按可用容量填满，再在相同三字段 cell 内随机填充不可避免的 378 个 deficit；group TV 为容量下界 0.126。该提案尚未 owner 接受，也尚未激活为 formal matcher。此前不得用放宽 quota、nearest bucket、replacement、T overlap 或标签替换绕过。详见 `SPECIFICATION_CHANGE_REQUEST_R2_INFEASIBLE.md` 和 canonical `08_reports/sctsr_v4_r2_specification_audit_20260814/`。

## 3. v3 回归口径矛盾

任务书 SA-266 和附录 C 要求 `tests/stage1_dynamic_replay_v3` 至少 231 passed。当前冻结 checkout 可复现的是 `183 passed, 1 skipped`；skip 是需要登记 Desktop mirror 的本地 evidence integration test。现有 tracked v3 测试树没有另外 48 个可执行测试可供复跑。

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

1. owner 对 R2 规格矛盾作书面、预注册决定；
2. owner 对 v3 231 基线矛盾作修订或恢复缺失测试；
3. 独立代码审查通过；
4. 从 exact clean source commit 生成 source manifest；
5. release authority 登记 key、seed 和签名 release；
6. 仍先做 canary/资源 gate，不自动启动正式训练；
7. blind/test 继续密封。
