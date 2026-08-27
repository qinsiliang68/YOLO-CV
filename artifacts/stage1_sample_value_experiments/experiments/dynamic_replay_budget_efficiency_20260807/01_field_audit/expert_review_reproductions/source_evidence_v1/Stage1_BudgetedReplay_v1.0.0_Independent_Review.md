# Stage1 Budgeted Replay v1.0.0 独立多角度审查报告

**审查日期：2026-08-09**  
**审查对象：`Stage1_BudgetedReplay_Learnability_20260809_v1.0.0`**  
**结论级别：正式研究启动前审查**

## 1. 总结性结论

### 当前判定：**NO-GO（禁止直接启动正式大矩阵）**

研究方向是成立的，仓库也不是空壳；但当前版本存在会改变论文结论符号或使结论失去独立性的阻断问题。最严重的不是代码能否运行，而是：

1. **真实 K-fold OOF 数据结构无法按当前实现正确表示；**
2. **默认 checkpoint 选择目标与高召回业务目标不一致；**
3. **`val_op` 被同时用于方法构造与工作阈值选择；**
4. **默认 canonical 直接使用 test oracle；**
5. **两个业务指标被错误压在同一个冻结阈值上；**
6. **Critical-CLD 的 trajectory 对齐与原方法/自身新实现均不一致；**
7. **正式 GPU/DDP 与实际曝光预算尚未获得集成证据。**

因此，先前“56 passed、1 skipped”只能证明现有单元测试与现有实现自洽，**不能证明科学协议正确，也不能证明真实 YOLO11m-cls 训练路径正确**。本次重新运行测试仍为 56 passed、1 skipped；fresh coverage 约 69%，而训练 runner、Ultralytics adapter/backend、predict、legacy pipeline、scorers/gates、experiment validation 等关键模块覆盖明显偏低。

本报告不否定以下主线：

\[
\text{有限预算 replay}
\rightarrow
\text{可学习性门控}
\rightarrow
\text{目标方向}
\rightarrow
\text{覆盖/冗余控制}
\rightarrow
\text{相对随机对照的真实干预验证}
\]

它否定的是“当前 v1.0.0 已经可直接作为正式论文实验基础”的判断。

## 2. 审查范围与方法

### 2.1 工程与运行

- 重新解压发布包；
- fresh `pytest`；
- fresh coverage；
- 关键 CLI/配置/模块静态审查；
- OOF、阈值、quota、矩阵数量的独立最小复现；
- 重点审查数据角色、checkpoint、DDP、预算、统计、provenance 和恢复语义。

### 2.2 科学与因果

审查了：

- T/R1/R2 是否同预算；
- proxy 是否与 intervention 结果混淆；
- selection/checkpoint/threshold/test 的数据角色；
- K-fold OOF 是否真正 cross-fitted；
- 两个业务 operating point 是否独立冻结；
- 多方法、多预算、多 seed 的选择偏倚；
- CCTV 连续帧/视频/管段泄漏。

### 2.3 文献方法一致性

重点核对原论文的方法章节、公式、算法流程和消融，而不是只读摘要：

- Nguyen et al., **Changing the Training Data Distribution to Reduce Simplicity Bias Improves In-distribution Generalization**（USEFUL，NeurIPS 2024）；
- Mindermann et al., **Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt**（RHO-LOSS，ICML 2022）；
- CLD/Confidence-Loss-Difference 数据剪枝工作；
- Balles et al., **A Negative Result on Gradient Matching for Selective Backprop**；
- AFSS, **Does YOLO Really Need to See Every Training Image in Every Epoch?**；
- PyTorch `DistributedSampler` 与 Ultralytics 当前 trainer/classification metrics 官方实现。

## 3. 阻断级问题（P0）

### P0-01：K-fold OOF 轴定义错误

`data/oof.py:171-179` 把 `trajectory_id/run_id/seed/fold` 全部拼为全局 trajectory ID；随后 `load_long_dynamics` 要求完整的 `sample×epoch×trajectory` 笛卡尔积。

标准 K-fold OOF 中，样本只由**没有训练过它的那个 fold 模型**产生预测。fold 是样本 assignment，不是每个样本都应覆盖的重复轨迹轴。最小复现中，2 fold×2 seed 的正确 OOF 输入被 strict converter 判定缺失 24 cells；允许缺失后 50% 的 cube 为 NaN，事件逻辑又将结构性缺失转成 `never_learned_fraction=0.5`，默认可靠性门控最终排除全部样本。

这会同时破坏：

- OOF reference loss；
- forgetting/relearning；
- slow-learning cluster；
- reliability gate；
- Critical-CLD；
- 跨 fold/seed consistency。

**必须重构后才能接真实 OOF。**

### P0-02：`best.pt` 按错误目标选 epoch

仓库训练后优先返回 `best.pt`。Ultralytics 分类 validator 的 fitness 使用 top-1/top-5；而你们的模型选择目标是高召回约束下的 TN/FN 工作点。二分类中 top-5 还基本失去区分力，因此默认 `best.pt` 与业务目标没有对齐。

正确选择只能是二选一：

- 所有 arm 统一固定 final epoch，使用 `last.pt`；或
- 保存所有候选 epoch，只在独立 `val_model` 上按预注册高召回规则选 epoch。

不能让 Ultralytics 默认 fitness 在不同 arm 中隐式决定 checkpoint。

### P0-03：`val_op` 被方法构造提前消费

Critical-CLD 与 critical gradient 的文档和代码明确使用 `val_op`；同一 `val_op` 又用于选择部署阈值。这样 proposed method 已经针对 operating-set 的标签/分数适配，再与对照比较，收益中混入对 val_op 的开发适配。

需要独立：

- `val_target`：构造 critical loss/gradient；
- `val_model` 或 `val_study`：checkpoint、方法、预算筛选；
- `val_cal`：概率校准；
- `val_op`：冻结两个工作阈值；
- `test`：只在最终冻结后一次性评估。

### P0-04：默认 test oracle 作为 canonical

示例配置、CLI 和 run path 默认 `canonical_source=test_oracle_curve`。即使字段旁边写了 analysis-only，只要汇总表、排名或人类决策使用它，就已经造成 test reuse。

63/99 个 triad 甚至未来 80 个 triad 反复查看 test oracle 后再选“最佳方法”，最终 test 不再是最终独立证据。

### P0-05：两个业务指标需要两套阈值

当前新 pipeline 只按 `FN≤95` 选一个阈值，再在该阈值的 confusion matrix 同时填：

- `TN_at_FN95`；
- `FN_at_TN68253`。

后者的定义应当是：寻找满足 `TN≥68253` 的阈值，再读取该阈值下的 FN。两者通常不是同一个阈值。

独立差分测试证明：oracle curve/tie 枚举本身没有发现错误，但 frozen-threshold orchestration 错了。旧 `evaluation.py` 反而正确保存两套阈值。

### P0-06：Critical-CLD 没有按 matched trajectory 计算

旧实现对每个 sample 只按 epoch 排序后 `shift`，会在多 seed/fold/model 的行之间跨轨迹做差；target 也只保留 epoch，无法 matching。新实现先跨 trajectory 取 median，再相关，也不是原 CLD 的 matched trajectory 过程。

应在每个 seed/model trajectory 内：

1. 计算样本 loss difference；
2. 计算独立 target 上对应 trajectory 的 critical/class-specific loss difference；
3. 做相关；
4. 最后跨 trajectory 聚合。

### P0-07：无真实 GPU/DDP 证据

当前环境没有真实 CCTV 数据、真实 OOF、YOLO checkpoint、Ultralytics runtime/CUDA，因此无法验证：

- dataset adapter 是否与现有训练器一致；
- physical replay 是否实际重复取样并独立增强；
- DDP 各 rank 的真实曝光；
- resume 后 schedule 是否严格连续；
- best/last checkpoint 的真实产物；
- 200 epoch 指标链。

这不是普通限制说明，而是正式实验前必须完成的验收阶段。

## 4. 高风险问题

| ID | 问题 | 结论影响 |
|---|---|---|
| H-01 | 动态特征、指标、校准、训练后端、统计均有重复实现 | 修一处可能仍从另一条路径跑出旧语义 |
| H-02 | schedule ledger 只记录计划曝光 | DDP 补索引/resume 可导致实际预算不相等 |
| H-03 | 依赖版本宽；`optimizer:auto` | 环境/优化器随版本漂移，科学参数未真正冻结 |
| H-04 | 无 current prediction 时 reducibility 静默 fallback | 将 RHO-inspired 当前可约性偷换为历史慢学习 |
| H-05 | embedding 缺 checkpoint/layer/preprocess provenance | alignment 可计算但语义可能错误 |
| H-06 | USEFUL 标题错误；多方法被写得过于接近原方法 | 论文方法陈述和复现声明风险 |
| H-07 | last-layer alignment 被误读为完整 influence | 只能作为局部一阶 diagnostic |
| H-08 | calibration 与 ranking 未完全分离 | isotonic ties 可改变 oracle ranking |
| H-09 | 3 seed + 大量比较 | 不能支撑确认性“稳压”结论 |
| H-10 | CCTV 泄漏只查 ID/path/duplicate_group | 同视频/管段/近重复帧可跨角色 |
| H-11 | B 默认是每 epoch slots | 200 epoch 累计预算可能放大 200 倍 |
| H-12 | 示例矩阵不是 240 run | 实际执行与冻结实验协议不一致 |
| H-13 | dynamic 实为 offline snapshots | 不能宣称在线 state-dependent replay |
| H-14 | controls 默认可与 T 重叠 | treatment contrast 被稀释，matching 不完整 |
| H-15 | 聚类/门控/组合权重均为研究超参数 | 容易在开发过程中选择性过拟合 |
| H-16 | physical repeat 未做真实 adapter/DDP 验证 | 逻辑重复不等于 optimizer 实际重复 |

## 5. 文献一致性审查

### 5.1 USEFUL

原 USEFUL 的关键结构是：

1. 从固定初始化训练到 early separating epoch；
2. 类别内对该时点的 network output 做 `k=2` 聚类；
3. 识别先被分开的简单特征群；
4. 将其余样本上采样一次；
5. 从**同一初始化**重新训练；
6. 与随机同量上采样、high-loss/misclassified、倍率与时机做对照。

仓库使用完整 trajectory、PCA、embedding、cluster quota 和预算 replay，这可以是合理的新方法，但只能叫：

> **Budgeted-USEFUL adaptation**

不能叫原方法复现。文档中的 USEFUL 正式标题也必须改正。

### 5.2 RHO-LOSS

原 RHO-LOSS 是在线 batch selection：从随机大候选 batch 中，用 current loss 减独立 holdout/irreducible-loss model loss，再选小 batch 训练。它的价值在于排除“高 loss 但所有模型都学不会/不值得学”的样本。

仓库的全局 OOF late loss + offline replay 是可研究的 adaptation，但：

- OOF reference 不是理论上的真实 irreducible loss；
- 没有 current checkpoint 时不能继续沿用同一个 reducibility 名称；
- 必须把 historical proxy 作为独立 arm。

### 5.3 CLD

原 CLD 使用单样本 loss-difference trajectory 与 held-out validation 的 class-specific loss-difference trajectory 做相关，并做类别平衡选择。仓库构造 FN95 局部 pairwise target 是有价值的任务适配，但必须满足 trajectory matching 和独立 target set。

### 5.4 Gradient matching/alignment

已有严格负结果显示：在补上随机基线、对齐学习率/反向传播预算后，高 loss 和最后一层 gradient matching 都未能稳定优于随机。因此：

\[
\|g_i\| \text{ 大}
\quad\text{或}\quad
\text{gradient matching 好}
\]

都不能直接当作 sample utility。仓库保留 anti-alignment control 是正确的；最终仍要靠 replay intervention 验证。

### 5.5 AFSS 与覆盖机制

AFSS 的重要启示不是“hard 全部多训”本身，而是：动态状态选择必须配套 easy review、moderate coverage、最近未见样本优先和周期更新。仓库目前的 snapshot schedule 没有完整实现这个 live feedback loop，因此不能把它称为 AFSS 等价实现。

## 6. 指标与统计审查

### 6.1 指标实现的正面结论

对随机、极端类别分布、大量相同分数和边界条件共 4,860 个用例，当前 oracle curve 的：

- `TN_at_FN95`；
- `FN_at_TN68253`；
- score tie 原子分组；

与独立暴力枚举一致，未发现曲线枚举 bug。

### 6.2 统计结论边界

3 seed 可以用于探索性筛选，但无法可靠估计跨 seed 胜率的尾部，更不能在数十个方法/预算/终点中挑最大均值后仍使用普通 bootstrap CI 当确认性证据。

主结论至少需要预注册：

\[
\Delta TN_{T-R1}>0,
\quad
\Delta TN_{T-R2}>0,
\quad
\Delta FN_{T-R1}\le0,
\quad
\Delta FN_{T-R2}\le0
\]

并同时报告：

- 双对照安全胜率；
- worst seed；
- leave-one-seed-out；
- paired randomization/permutation；
- 多重比较校正；
- 对短名单增加独立 seed 的确认实验。

## 7. 预算与 sampler 审查

### 7.1 预算单位

当前 `append_slots_per_epoch` 下，selection 的 repeat_count 每个 epoch 都重放；因此：

\[
B_{cumulative}=B_{per\ epoch}\times epochs
\]

必须与旧 240-run 的预算语义核对。若旧实验 B3000 指整个 run 只增加 3000 次曝光，当前实现会严重超预算。

### 7.2 cluster quota bug

`cluster_quota_select` 先按簇形成 provisional quota，随后把 provisional 与全局候选拼起来，再按 score/caps 全局填充；最终可能违反 quota。最小复现中 equal quota 目标为 A2/B2，实际返回 A1/B3。

### 7.3 random control 的真正优势

当前证据支持把 random 的优势拆成两个可检验机制：

1. score 本身可能错误；
2. score 可能尚可，但 deterministic Top-K 丢失覆盖、探索与长期复习。

下一版应固定 score，只替换 sampler，避免把两个因素混在一起。

## 8. 代码质量与工程正面发现

- T/R1/R2 同预算因果框架方向正确，优于只看 proxy 与最终 accuracy。
- TN_at_FN95 与 FN_at_TN68253 的 oracle 曲线枚举及 tie 处理，经 4,860 组差分用例未发现错误。
- manifest 作为标签权威可避免 ImageFolder 类别字母排序导致的 0/1 反转。
- selection manifest、repeat histogram、caps、hash、原子写入、run state 和多机 claim 具备良好工程基础。
- 代码明确区分 hardness、learnability、direction 和 set-level redundancy，研究思路本身成立。
- anti-critical/reference-risk negative controls 值得保留，可检验 proxy 是否真正预测方向。

## 9. 修复优先级

### 第一批：不修不能接数据

1. 重构 K-fold OOF schema；
2. 修复双阈值 evaluation；
3. 禁用 test oracle canonical；
4. 建立 val_target/val_study；
5. 固定 checkpoint selection；
6. 统一 Critical-CLD 实现。

### 第二批：不修不能证明同预算

1. 实际 DDP exposure ledger；
2. 明确 B 是 per-epoch 还是 total-run；
3. 冻结 optimizer 与精确环境；
4. physical replay integration test；
5. 唯一 canonical pipeline。

### 第三批：不修不能写“稳压”

1. 预注册 primary endpoint；
2. 多重比较控制；
3. 短名单增加 seed；
4. video/pipe/source grouped split；
5. test 一次性最终验收。

## 10. 建议的最小正式实验路径

在修复后，不要立即跑全部复合方法。先做四层递进：

1. **score validity**：Random、Hardness、Slow、Current-high/OOF-ref-low、Current-high/OOF-ref-high；
2. **sampler validity**：同一 score 下 Top-K、soft sampling、stratified、coverage、random super-pool；
3. **direction validity**：positive critical alignment、anti-critical、gradient norm；
4. **combined method**：Learnability + reliability + direction + coverage。

每一步只有在跨 seed 对 R1/R2 都出现稳定正方向后，下一层才允许进入组合。这样才能知道最终收益来自哪个机制，而不是一次堆满所有文献组件后无法归因。

## 11. 最终验收标准

仓库只有同时满足以下条件才可从 NO-GO 转为 GO：

- 真 K-fold OOF fixture 通过；
- OOF cube 无结构性 NaN；
- 两套阈值分别冻结与应用；
- test oracle 从所有开发排名中删除；
- val_target 与 val_op 分离；
- checkpoint objective 冻结且审计；
- 单卡与 DDP actual exposure 完全对账；
- frozen_240run 配置恰为 80 triads/240 arms；
- Ultralytics 真实 1-epoch parity smoke 通过；
- resume 前后 schedule/exposure 一致；
- video/pipe/source 泄漏为 0；
- 统计决策规则预注册；
- 文献方法全部改为准确的 reproduction/adaptation 声明。

## 12. 审查边界

本审查尽力通过源代码、独立复现、单元测试、覆盖率和原始文献交叉验证降低错误风险，但不能诚实地承诺“绝对零错误”。由于当前环境缺少真实数据、GPU、真实 Ultralytics runtime 和原始训练仓库，真实训练 adapter/DDP 路径仍属于未验证区域；本报告因此把它列为明确的 GO 阻断项，而不是假定正确。
