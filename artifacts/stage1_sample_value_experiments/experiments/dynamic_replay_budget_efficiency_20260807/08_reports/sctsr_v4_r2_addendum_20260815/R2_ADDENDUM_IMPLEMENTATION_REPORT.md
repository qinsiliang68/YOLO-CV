# SCTSR v4 R2 addendum 实施与验证报告

报告日期：2026-08-15

## 结论

Owner 批准的单字段放宽已实现。仅 `oof_group_id` 从完全精确匹配改为容量下界上的最小偏移；`y_true`、`historical_dynamic_bucket` 和 `oof_fold` 仍逐层精确，R2 仍为 3,000 unique IDs，与 T 零身份重叠。

`R2_U`、`R2_F` 和 `T_TO_R2_AT_160` 的 fallback 复用同一个 R2 pool，而不是分别抽取两个 3,000-ID pool。U/F 总曝光都是 48,000，每个 ID 都是 16 次，仅时间分布不同。因此，两个时间表对照没有额外的样本身份差异。

当前结论为 `R2_SPECIFICATION_BLOCKER_RESOLVED_NOT_TRAINING_AUTHORIZATION`。这不是 SCTSR 方法有效性证据，也没有启动正式训练。

## 为什么放宽 `oof_group_id`

当前 `oof_group_id` 是从数字文件名派生的 bucket surrogate，不是真实 video/source/pipe identity。它在四个原匹配字段中因果含义最弱。冻结 T 并排除 T 后，四字段完全精确匹配有 172 个 shortage strata、378 个 occurrence 缺口；放宽该字段后，只在同一 `(label, dynamic bucket, fold)` cell 内填充这 378 个不可避免的缺口。

R1 仍是 global random，作为 co-primary identity-neutral control。正式结果必须报告 378/3000 displacement 和 group TV=0.126，不得把 T-R2 差异伪称为已完全控制真实视频或来源。

## 真实资产物化结果

- canonical base rows：120,000；
- T unique IDs：3,000；
- R2 unique IDs：3,000；
- T/R2 identity overlap：0；
- 五个 R2 identity groups：G0..G4 各 600；
- exact fields：`y_true`, `historical_dynamic_bucket`, `oof_fold`；
- sole relaxed field：`oof_group_id`；
- displacement rows：378；
- group total variation：0.126；
- R2 identity digest：`075FC31FE487D3646E89BA1043E5124D9FE49CE9FCC61C1A8041A9CB8196BECC`；
- displacement ledger digest：`4DC4918D07858E112498613D7313A4F0960BB100C0E57FA2755E9118B5762015`；
- policy SHA-256：`E58E9D7093C6148687472769A1781AA0F3985A76C8C4269DF143C5D1B565EA3C`。

同一正式构造命令独立执行两次，两次的 R2 identity digest 和 displacement ledger digest 一致。物化目录位于本地临时测试空间，不是正式训练资产，不纳入科学结果目录。

## 调度公平性

- `R2_U` schedule digest：`D976728392D9177AC8DA1EF107DE098EE6369AA38DC5391E8F0781DC85384079`；
- `R2_F` schedule digest：`0E0A592349D4C19D978C0570386EC8882E7A5E155F8E14BCF86AE9E8DA5F01EB`；
- `T_TO_R2_AT_160` schedule digest：`7A607BEC1A6737F6BB9A72AD301D3E907F942C4B84BD5F6525C51CBB81EDE348`；
- U/F 身份 digest 相同；
- U/F 总 occurrence 同为 48,000；
- U/F 每 ID multiplicity 同为 16；
- fallback 使用完全相同的 3,000 IDs。

## 代码与测试

- base commit：`30e50ea3130694af73ae486827b7109930e61d68`；
- implementation commit：`e940199`；
- documentation commit：`07f428d`；
- Python 3.11 v4：404 passed；
- Python 3.12 v4：404 passed；
- Python 3.11/3.12 compileall：PASS；
- v3 regression：181 passed, 3 skipped；
- runbook v4：18 documents, manifest SHA-256 `7D11D7DF3151CE1D6E0AD7B54B691C7A19F572F3EC88E88A56C165B1C3E2885F`。

## 仍未解除的边界

1. 尚未登记正式 8+14 training seeds。
2. 尚未签发 matrix release、one-use job tokens 和 10 台机共享 claim registry。
3. 尚未在正式 3090 训练机上做 single-epoch engineering benchmark。
4. `val_target` 缺失仍使 Phase 2 A/gradient alignment 保持 HELD，但不是 Phase 1 timing/stop/fallback 的 R2 构造阻断。
5. 任何 SCTSR utility 判断都必须等待真实 replay 相对 R1/R2/no-replay 的配对干预。

`formal_training_started=false`，`assignment_generated=false`，`engineering_gate_generated=false`，`blind_holdout_opened=false`。
