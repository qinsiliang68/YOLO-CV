# SCTSR v4 R2 最小组偏移匹配 Addendum

文档状态：`OWNER_APPROVED_IMPLEMENTED_NOT_TRAINING_AUTHORIZATION`

- 批准日期：2026-08-15
- 适用阶段：SCTSR v4 Phase 1
- 上游任务书：`SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md`
机器政策：`configs/stage1_sctsr_v4/r2_matching_policy_v1.json`

## 1. Owner 决定

研究负责人批准只放宽一个条件：`oof_group_id`。该字段在现有资产中的语义是
`FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID`，即按数字文件名派生的 bucket，
不是真实视频、管段或来源身份。相比 label、历史 dynamic bucket 和 OOF fold，
它是四个匹配条件中因果含义最弱的一项。

“两个对照都放宽”在机器合同中的唯一解释是：

- `R2_U` 使用本 addendum 的 R2 pool；
- `R2_F` 使用同一个、字节身份相同的 R2 pool；
- `T_TO_R2_AT_160` 的 E161-E200 fallback 也只能使用该同一 R2 pool；
- 不为 U/F 各自另抽一套 3,000 IDs，不需要 6,000 个不同 R2 身份；
- `R1_U` 仍是完整 eligible base 上的 global random，不把它伪装成 matched R2。

## 2. 保持不变的条件

R2 仍必须同时满足：

1. 恰好 3,000 unique canonical-base IDs；
2. 与 T 的 3,000 IDs 身份交集为 0；
3. `y_true` 精确匹配；
4. `historical_dynamic_bucket` 精确匹配；
5. `oof_fold` 精确匹配；
6. selection seed 固定为 `20260812`；
7. matcher 只能读取预终端白名单投影；
8. loss、confidence、RHO、gradient、forgetting、AUM、future outcome、
   val_model/val_cal/val_op/test 指标全部不可见；
9. U/F total occurrence 均为 48,000，每 ID multiplicity 均为 16；
10. U/F 只允许时间分布不同；
11. 不允许 replacement、T overlap、少于 3,000 unique、改 label、改 fold、
    改 dynamic bucket 或再放宽第二个字段。

## 3. 被本 addendum 精确替代的旧条款

任务书 SA-051、SA-053、SA-054 中“`oof_group_id` 必须完全精确且不得出现任何
relaxation”的部分被本文件替代。其他任务书条款继续有效。

旧四字段 exact 定义在冻结资产上有 172 个 shortage strata、378 个 occurrence
缺口，其中 30 个 strata 完全没有零重叠候选。该不可行性仍保留为事实证据；
本 addendum 不把旧定义改写成“其实可行”。

## 4. 唯一允许的构造算法

政策 ID：

`MINIMUM_OOF_GROUP_DISPLACEMENT_ZERO_OVERLAP_V1`

算法必须严格按以下顺序执行：

1. 先用 `TerminalFieldGuard` 做白名单投影；
2. 从 canonical base 排除全部 T identities；
3. 按 `(y_true, historical_dynamic_bucket, oof_fold, oof_group_id)` 统计 T
   需求和零重叠候选容量；
4. 在每个四字段 cell 内先取满 `min(required, available)`；
5. 对剩余的精确 cell 缺口建立 378 个逐 occurrence requested-group slots；
6. 只在同一个 `(y_true, historical_dynamic_bucket, oof_fold)` cell 内，用
   counter hash `R2_minimum_displacement_fill` 随机排序未选候选；
7. 将候选逐一绑定到 requested-group slot，并保存 selected sample、requested
   group、actual selected group 和 counter hash；
8. 验证四字段联合距离恰等于容量下界 378；
9. 验证 `oof_group_id` total variation 恰为 `378/3000 = 0.126`；
10. 若三字段 cell 仍不足，立即 `R2_QUOTA_INFEASIBLE`；不得继续放宽。

## 5. 冻结身份

- T identity digest：
  `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`；
- R2 selection seed：`20260812`；
- R2 expected identity digest：
  `075FC31FE487D3646E89BA1043E5124D9FE49CE9FCC61C1A8041A9CB8196BECC`；
- displacement row count：378；
- displacement ledger digest：
  `4DC4918D07858E112498613D7313A4F0960BB100C0E57FA2755E9118B5762015`；
- group total variation：0.126；
- machine policy semantic digest：
  `33B4681FFF92360FD253AE7B0CC92CC0A2D7E27AEAB65A238730600ECEFE7D4F`；
- machine policy file SHA-256：
  `E58E9D7093C6148687472769A1781AA0F3985A76C8C4269DF143C5D1B565EA3C`；
- amended contract file SHA-256：
  `DF30244A9B86417E5C63DF6587E1677844CADF83D7F737AC5B38A3328BEE0271`。

上述 R2 digest 只对当前冻结 base/OOF/dynamics/T bytes、selection seed 和算法版本
成立。任一输入变化都必须得到新 digest；不得用“统计分布差不多”继续沿用。

## 6. 公平性与 estimand

本修订后的 C01/C02 不再估计“对四字段完全相同且零重叠的随机身份”的差，而是估计：

> 在 label、历史 dynamic bucket、OOF fold 完全相同，且 filename-bucket surrogate
> 偏差达到数据容量可实现最小值时，T 相对 matched random 的差。

因此正式结果必须同时报告：

- C01：`T_U - R2_U`；
- C02：`T_F - R2_F`；
- C07：`R2_U - NR`；
- C08：`R1_U - NR`；
- 378/3000 displacement；
- group TV=0.126；
- R1 作为 co-primary identity-neutral random control。

不能把任何 T-R2 差异完全归因于真实视频/来源控制，因为 `oof_group_id` 本身不是真实
视频 ID，而且本 addendum 在该 surrogate 上存在最小不可避免 imbalance。

## 7. 训练授权边界

本文件只批准科学规格并允许代码/测试/正式 pool 预物化验证。它不生成或授权：

- discovery/confirmation training seeds；
- 签名 matrix release；
- one-use job tokens；
- shared claim registry；
- assignment、engineering gate 或 pilot release；
- formal training；
- blind/test 访问；
- 方法有效性结论。

当前布尔事实仍为 `formal_training_started=false`。只有新 source commit、合同、policy、
asset、pool、schedule、seed registry、runtime 和 release 全部 SHA 绑定并通过独立
训练前审查后，才可另行决定是否部署正式训练。
