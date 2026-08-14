# SCTSR v4 R2 规格审计与唯一推荐方案

状态：`PROPOSED_OWNER_PREREGISTRATION_DECISION_REQUIRED`

本报告只回答 R2 是否可构造、不同修订会改变什么公平性与 estimand。它不生成 identity pool，不授权训练，也不声称 T、R2 或 SCTSR 有效。

## 1. 冻结输入

- canonical base：120,000 个唯一训练身份；
- T：3,000 个唯一身份，digest `85D462C1D95F30FB8B519162BBAD762CC4E9506A185C07D719145F07FE003B4B`；
- T 组成：3,000/3,000 为 `y_true=0`、`normal_replay`、`learnable_hard`；
- OOF fold：10 个；
- T 覆盖 959 个 `oof_group_id`；
- `oof_group_id` 语义：`FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID`，不能在报告中称作真实 video/source ID；
- selection seed：`20260812`；
- matching 前可见字段只有 sample identity、label、历史 dynamic bucket、OOF fold、OOF group surrogate 和 base membership；loss、confidence、GapCritical、RHO、gradient、AUM、future outcome 与 endpoint 均未进入审计器。

机器审计：

- `R2_SPECIFICATION_AUDIT.json`：41,581 bytes，SHA-256 `0467E607D91C0A56DBF7CB12E4BD2B7DE66EE7FCD6ECC58222A593E743FA91A7`；
- audit digest：`3C42E9B362AE788B6FA6B45C36D12BBDB7AA4EEB6FCFA503CC3AEDB6ED04A180`；
- receipt：1,037 bytes，SHA-256 `0F58B232F0408C19D4B2E3D1D641F30F92E0DCEB106569545AF55FFC09E5E629`。

## 2. 不可行性不是实现错误

原规格要求 R2 同时满足：

1. 3,000 unique IDs；
2. 与 T 零身份重叠；
3. `(label, dynamic bucket, OOF fold, oof_group_id)` 四字段联合 quota 与 T 完全相同；
4. 不 replacement、不放宽、不回用 T。

冻结数据中 T 有 959 个四字段 cell。排除 T 后：

- 172 个 cell 候选不足；
- 累计短缺 378 个 occurrence；
- 其中 30 个 cell 的非 T 候选为零，涉及 61 个 occurrence；
- 最多只能保留 2,622 个 exact-match T 身份，即 87.4%。

因此四条要求联合无解。当前 `build_registered_r2` 抛出 `R2_QUOTA_INFEASIBLE` 是正确的 fail-closed 行为。

## 3. 候选规格比较

| 方案 | 可构造 | unique/overlap | 保留的公平条件 | 代价 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 原四字段 exact | 否 | 目标 3,000/0 | 全部 | 缺 378 | 保持失败封闭 |
| exact + replacement | 否 | unique 会下降 | 四字段 occurrence | 30 个空 cell 无法 replacement；repeat 不公平 | 拒绝 |
| exact + 允许 T overlap | 是 | 最少重叠 378；seeded random 实际 1,285，42.83% | 四字段 exact | 对照被 treatment 污染、有效差异下降 | 拒绝 |
| matchable T subset | 是 | 2,622/0 | 四字段 exact | 改 T 身份、2.5% pool rate、schedule 与研究对象 | 拒绝第一阶段采用 |
| 直接删 group 后 hash random | 是 | 3,000/0 | label/dynamic/fold exact | group TV 39.233%，不必要地放大残余失衡 | 被支配 |
| 增加新 canonical data | 当前不可测 | 未知 | 可恢复 exact | 改 120,000 分母、E1–E120 common parent 和固定基础过程 | 本轮拒绝 |
| minimum-displacement zero-overlap | 是 | 3,000/0 | label/dynamic/fold exact；87.4% 四字段 exact | 378 个 group occurrence 位移，group TV 12.6% | **唯一推荐** |

## 4. 推荐算法的无歧义定义

推荐 policy 暂命名：

`R2_ZERO_OVERLAP_MINIMUM_GROUP_DISPLACEMENT_RANDOM`

输入仍为冻结 T、canonical base 和预终端投影。算法必须按以下顺序执行：

1. 从 canonical base 删除全部 T IDs；禁止在后续 fallback 中重新加入；
2. 按 `(y_true, historical_dynamic_bucket, oof_fold, oof_group_id)` 建立 T required quota 与非 T available quota；
3. 对每个四字段 cell，选择 `min(required, available)` 个唯一候选；cell 内只使用 counter-based random，domain 固定为 `R2_exact_capacity`，key 包含 selection seed、完整 cell token 和 sample ID；
4. 将未满足数按 `(y_true, historical_dynamic_bucket, oof_fold)` 汇总。本资产总 deficit 必须恰为 378；
5. 只从相同三字段 cell 的剩余唯一候选中填充 deficit；使用独立 counter domain `R2_minimum_displacement_fill`；不得跨 label、dynamic bucket 或 fold；
6. 不允许 replacement，不允许 duplicate ID，不允许 T overlap，不允许最近 group、group 数值距离、terminal score 或现场随机状态；
7. 输出必须恰为 3,000 unique IDs，并验证三字段联合 quota 与 T 完全相同；
8. 计算四字段 selected quota 与 T quota 的半 L1 displacement。它必须等于容量下界 378，不能只声称“接近”；
9. 计算 `oof_group_id` quota total variation。本冻结输入和 seed 下必须为 `378/3000 = 0.126`；
10. 五个 600-ID group 只能在最终 3,000-ID pool 冻结后按既有稳定散列划分，不能反向影响 R2 选择；
11. R2-U、R2-F 和 `T_TO_R2_AT_160` 必须引用同一 pool digest `075FC31FE487D3646E89BA1043E5124D9FE49CE9FCC61C1A8041A9CB8196BECC`；不同 schedule 不得重新抽 R2；
12. 任意输入 SHA、T digest、selection seed、shortage count、displacement lower bound、最终 digest 或 quota 改变时 fail closed，必须重开 addendum，禁止“尽量匹配”后继续。

## 5. 这个修订改变和不改变什么

改变：R2 不再声称 exact `oof_group_id` quota；12.6% occurrence 使用同 label/dynamic/fold、不同 filename-bucket surrogate 的候选。R2 estimand 因而是“在主要预终端难度/折别相同、group 失衡最小但非零时的随机回流”，不是“逐 group 完全条件化随机回流”。

不改变：T、T digest、3,000 unique pool size、零身份重叠、label、dynamic bucket、OOF fold、selection seed、五组、U/F/stop/fallback schedule、每 ID multiplicity、累计 exposure、base steps、optimizer steps、E120 parent、E200 endpoint 和评价规则。

残余风险必须贯穿结果解释：`oof_group_id` 虽只是 filename-bucket surrogate，12.6% group 位移仍可能携带来源/近重复结构。因而 R2 不能单独证明 selector utility。R1 全局随机必须保留为共同主要对照；若 T 只胜 R2、不胜 R1，结论不能写成稳定样本价值成立。

## 6. 激活门

本报告推荐但不激活该规格。只有 owner 明确接受本 addendum 后，才能在新的最小回滚提交中：

- 替换 formal `build_registered_r2` policy；
- 扩展 pool/quota audit schema，显式记录 coarse exact、strict displacement lower bound、observed displacement 和 group TV；
- 更新 formal pool validator、selection ledger、文档和失败优先测试；
- 重新生成正式候选 pool 只读预检并核对上述固定 digest；
- 继续保持 release、seed、assignment、gate 和 formal training 关闭。

在此之前，正式状态仍为 `R2_QUOTA_INFEASIBLE`，而不是“代码差一点就能跑”。
