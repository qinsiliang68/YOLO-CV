# SCTSR v4 实验意图与科学问题

## 1. 这次实验到底在研究什么

本次 Stage1 研究的名称是 **State-Conditional Tail-Safe Replay
(SCTSR)**。第一阶段不训练 selector，也不证明 Q/R/A/D 有效。它只回答一个
更基础、可证伪的问题：

> 在 E120 的同一模型、优化器、调度器、AMP 和 RNG 状态之后，给一组固定
> 身份的训练样本额外曝光时，样本身份、额外曝光发生的时间，以及 E160 后
> 停止定向回流或退回匹配随机，是否能在未见 training seed 上稳定改变
> E200 的 FN=0..95 尾部安全前沿？

这里的“价值”不是图像的永久属性。研究对象是下一次额外曝光的条件边际效用：

`V(S | theta_t, epoch=t, previous_replay_count=k, schedule, training_seed)`

- `S` 是一个集合，不是孤立图片；
- `theta_t` 是当前模型/优化器状态；
- `k` 是集合中身份此前已获得的额外曝光次数；
- utility 只能由真实 replay 干预相对严格随机/no-replay 的配对结果定义；
- loss、confidence、RHO、gradient、forgetting、AUM 和 coverage 只是候选信号，
  不能标成 utility。

## 2. T 是什么，不是什么

`T` 是从冻结历史压力集合派生的、内容唯一的 v4 压力集合：

- historical source 原样保留：
  `artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/generated/selections/RUN_010/selection_manifest.csv`；
- v4 canonical source：
  `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/assets/T_STRESS_CONTENT_UNIQUE_v1.csv`；
- 3,000 个 unique canonical train IDs；
- 全部为 normal/label 0；
- identity digest：
  `D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7`；
- 正式角色：
  `HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR`。

派生过程只替换一条重复图像字节的身份，并保持 label/dynamic/fold/oof_group 四字段
quota 不变；详见 `SCTSR_DATA_CONTENT_AND_T_REPAIR_ADDENDUM_20260815.md`。T 被选来做
压力测试，是因为历史实验中它跨 seed 出现过正、负和混合方向。T
不是已证明有用的 selector，不代表 Q/R/A/D，不代表“最难样本”，也不允许在
结果出来之前被称为高价值样本。

## 3. 第一阶段八臂

所有 arm 从同一 training seed 的同一 E120 common-parent checkpoint 分叉：

| arm | E121-E160 | E161-E200 | 目的 |
| --- | --- | --- | --- |
| `NR` | no replay | no replay | 基础过程；额外曝光总效应的零点 |
| `R1_U` | global random，600/epoch | 同左 | 身份中立随机回流 |
| `R2_U` | matched random，600/epoch | 同左 | T-U 的严格身份对照 |
| `T_U` | T，600/epoch | 同左 | 固定 T 身份、均匀时序 |
| `R2_F` | matched random，1,200/epoch | no replay | R2 的前置集中时序 |
| `T_F` | T，1,200/epoch | no replay | 固定 T 身份、前置集中 |
| `T_TO_R2_AT_160` | T，600/epoch | R2，600/epoch | E160 后停止定向并退回随机 |
| `T_TO_NR_AT_160` | T，600/epoch | no replay | E160 后停止全部 replay |

对 U/F 同一身份池：

- U：80 × 600 = 48,000 optimizer-visible replay occurrences；
- F：40 × 1,200 = 48,000 optimizer-visible replay occurrences；
- pool：3,000 unique IDs；
- 每个 ID 累计额外 multiplicity：16；
- 二者只允许时间分布不同。

`T_TO_R2_AT_160` 总额外曝光仍为 48,000，但前 24,000 来自 T、后 24,000
来自 R2，因此它与 `T_U` 比较的是“后期定向身份是否应替换为随机”，不是纯
stop effect。`T_TO_NR_AT_160` 只有 24,000 次额外曝光，与 fallback 比较的是
共同前缀后“继续普通 replay 还是不 replay”。

## 4. 预注册比较和可推出结论

| contrast | treatment - comparator | 直接回答 | 不能推出 |
| --- | --- | --- | --- |
| C01 | `T_U - R2_U` | U 时序下 T 身份是否优于匹配随机 | T 有永久/内在价值 |
| C02 | `T_F - R2_F` | F 时序下 T 身份是否优于匹配随机 | 任意 selector 都有效 |
| C03 | `T_F - T_U` | 固定 T、dose、multiplicity 后的时序差 | 可推广到其他集合 |
| C04 | `R2_F - R2_U` | 匹配随机集合的时序差 | T 的身份价值 |
| C05 | `T_TO_R2_AT_160 - T_U` | 后期把 T 换成 R2 是否更安全 | 纯停止或纯 dose 效应 |
| C06 | `T_TO_NR_AT_160 - T_TO_R2_AT_160` | 共同 T 前缀后，后期普通 replay 是否有价值 | 两臂累计 dose 相同 |
| C07 | `R2_U - NR` | 匹配随机额外曝光的总体效应 | 选择方法有效 |
| C08 | `R1_U - NR` | 全局随机额外曝光的总体效应 | 匹配策略有效 |

如果 C01/C02 均不稳定，而 C03/C04 稳定，研究方向应转向 budget scheduling；
如果 C05 稳定为正，才有证据继续研究状态条件的 abstention/fallback；如果严格
控制后所有身份/停止信号仍跨 seed 翻转，则应停止扩展 selector，转向 pAUC /
Neyman-Pearson / rate-constrained objective 的独立预注册路线。

## 5. 数据角色与泄漏边界

- `train`：canonical base 和 replay 身份的唯一来源；
- OOF：只提供预终端 fold/group/reference；不能使用本模型未来结果；
- `val_model`：训练轨迹研究/上游固定 validation loader；不得选正式方法；
- `val_cal`：未来冻结阈值/校准的独立角色；不得冒充 val_target；
- `val_op`：只评价冻结的 E200/EMA endpoint；不得选 arm、E160、checkpoint、
  stop 或 threshold；
- `val_target`：当前不存在，因此 A/gradient-alignment 正式模块必须阻断；
- blind/test：在全部方法、代码、阈值、停止规则和统计规则冻结前保持密封。

第一阶段 timing/stop/fallback 不使用 A，因此缺少 val_target 不阻止代码审查，
但任何 A arm 或 selector 训练仍为 `BLOCKED_BY_VAL_TARGET`。

## 6. 正式 endpoint 和指标顺序

唯一正式模型 endpoint 是：

- epoch：E200；
- variant：EMA；
- split：`val_op`；
- `best.pt`：禁止；
- E120/E140/E150/E160/E180：轨迹锚点，`NOT_FOR_METHOD_SELECTION`。

评价首先发布 FN=0..95 的 96 个不拆 tie 原始 frontier 点，再计算 normalized
AUC。两个锚点使用独立阈值：

- `TN_at_FN95`；
- `FN_at_TN68253`。

禁止把二者拼成一个混淆矩阵。必须同时报告 seed 胜率、worst seed 和双端同时
恶化 seed 数。探索阶段门槛为至少 7/8 同方向；确认阶段门槛为至少 12/14、
worst-seed primary delta 非负、双端同时恶化为 0，并用 exact paired
sign-flip 与 Holm 控制多重比较。

## 7. 当前状态

截至本文件版本：

- `formal_training_started=false`；
- `assignments_generated=false`；
- `engineering_gate_generated=false`；
- `pilot_release_generated=false`；
- `blind_holdout_opened=false`；
- `test_accessed=false`；
- `method_effectiveness_claimed=false`。

代码和 synthetic/engineering canary 的成功只说明训练链路可施工，不是方法
有效。R2 四字段零重叠精确配额在真实资产上不可行这一事实仍成立；owner 已批准
只放宽 filename-bucket surrogate 的预注册 addendum，代码现可物化公平的共享 R2
pool。正式八臂训练仍必须等待新 commit 独立复审及签名 release/control plane。
