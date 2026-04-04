# 第一阶段 Max-Filter 套件阶段性报告

## 1. 报告目的

本报告用于整理当前 `stage1_gate_maxfilter_suite` 的完整结果，并回答三个问题：

1. 第一阶段在 `hn02 + calibration + PTSG` 之后是否仍有训练侧改进空间。
2. 如果有，最有效的增益来自哪里：
   - 更贴合工作点的损失函数
   - 更细的 hard mining
   - 通用型 reweighting loss
   - defect oversampling
3. 在新训练策略下，`P0/P2` 的关系是否发生变化。

本报告不是最终论文正文，而是论文收口前的事实依据与解释底稿。

## 2. 统一比较口径

本轮 suite 固定以下条件不变：

- 主模型：`yolo11l-cls`
- HN：固定 `hn02`
- 校准协议：固定 `val-cal / val-op`
- 后处理评估：统一执行 `P0~P4`
- 总排序规则：
  1. `Spec@R99.5`
  2. `Spec@R99.0`
  3. `Prec@R99.0`
  4. `PTR@R99.0` 升序

这意味着本轮比较的不是“谁训练日志更好看”，而是：

> 在同一套 stage-1 gate 协议下，哪种训练策略最终能让 gate 在高召回工作点拦下更多 normal。

## 3. 参赛实验

本轮共比较 6 组：

- `H0`：当前最优基线 `yolo11l-cls + calibration + hn02 + P2`
- `Selective / recall-constrained loss`
- `Hard positive + hard normal mining`
- `Weighted BCE`
- `Focal BCE`
- `Defect oversampling`

其中：

- `Selective` 代表训练目标层面的直接工作点导向尝试
- `HardMix` 代表 failure-distribution 层面的训练数据组织增强
- `WBCE / Focal` 代表通用型 reweighting / hard-example loss
- `DefectOS` 代表提高 abnormal 出现频次的简单 oversampling

## 4. 核心总结果

总表见：

- `stage1_maxfilter_suite_summary.csv`
- `stage1_maxfilter_suite_summary.json`
- `stage1_maxfilter_suite_best_metrics.png`

最重要的结论先说：

> 本轮 suite 的综合最优组是 **Hard positive + hard normal mining**。

它的最佳后处理变体不是 `P2`，而是：

> **P0 = calibrated p_abnormal**

这说明当前最有效的增益并不是继续在后处理端叠加更复杂的 trust 修正，而是：

> 通过更精确地暴露训练侧 hardest normal 与 hardest positive，先把训练边界本身修得更好。

## 5. 与当前最佳基线 H0 的直接对比

### 5.1 H0 当前最佳基线

`H0 = yolo11l-cls + calibration + hn02 + P2`

- `Spec@R99.5 = 0.52381`
- `Spec@R99.0 = 0.559524`
- `Prec@R99.0 = 0.918322`
- `PTR@R99.0 = 0.89881`
- `TN@R99.5 = 44`
- `FN@R99.5 = 2`
- `TN@R99.0 = 47`
- `FN@R99.0 = 4`

### 5.2 HardMix 最优组

`Hard positive + hard normal mining`

- 最佳变体：`P0`
- `Spec@R99.5 = 0.52381`
- `Spec@R99.0 = 0.583333`
- `Prec@R99.0 = 0.922395`
- `PTR@R99.0 = 0.894841`
- `TN@R99.5 = 44`
- `FN@R99.5 = 2`
- `TN@R99.0 = 49`
- `FN@R99.0 = 4`

### 5.3 直接解释

与 `H0` 相比，`HardMix`：

- 在 `R99.5` 下没有进一步增加拦截数量：
  - `44 -> 44`
  - `FN 2 -> 2`
- 但在 `R99.0` 下进一步多拦截了 `2` 张 normal：
  - `47 -> 49`
  - `FN 4 -> 4`

同时：

- `Prec@R99.0` 从 `0.918322 -> 0.922395`
- `PTR@R99.0` 从 `0.89881 -> 0.894841`

因此它虽然没有打破 `Spec@R99.5`，但在与 `H0` 主排序第一项持平的情况下，凭借更好的：

- `Spec@R99.0`
- `Prec@R99.0`
- `PTR@R99.0`

拿到了本轮综合第一。

换句话说，这轮结果的最准确表述不是：

> 第一阶段被全面重写了

而是：

> 第一阶段在更严格的 `R99.5` 主锚点上已经很难继续抬高，但在 `R99.0` 及次级指标上，hard-mix 训练仍然挖出了真实增益。

## 6. 各方法线分别说明了什么

### 6.1 Selective / recall-constrained loss

最佳变体：`P1`

- `Spec@R99.5 = 0.488095`
- `Spec@R99.0 = 0.52381`
- `TN@R99.5 = 41`
- `FN@R99.5 = 1`
- `TN@R99.0 = 44`
- `FN@R99.0 = 3`

它的特点很鲜明：

- `FN` 比 `H0` 更少
  - `R99.5`: `2 -> 1`
  - `R99.0`: `4 -> 3`
- 但 `Specificity` 明显更低

因此这条线的意义不是“综合最优”，而是：

> 它确实把决策边界推得更保守了

也就是说，recall-oriented surrogate 这条线并非没有效果，只是当前参数下它更像是在用更多放行 normal 换更少的 defect 阻塞。

如果未来任务偏向“宁可多放，也不能多漏”，这条线仍然值得保留为备选。

### 6.2 Hard positive + hard normal mining

最佳变体：`P0`

这是本轮真正的赢家。

它说明：

> 当前 stage-1 剩余 headroom 更像是样本暴露问题，而不是简单的通用损失问题。

也就是说，模型已经不是不知道 normal / abnormal 的大致边界，而是：

- 还需要更充分地看到最危险的 normal
- 还需要更充分地看到最弱的 abnormal

### 6.3 Weighted BCE

最佳变体：`P0`

- `Spec@R99.5 = 0.488095`
- `Spec@R99.0 = 0.559524`
- 只在 `R99.0` 上与 `H0` 持平
- 综合并未形成新优势

这说明单纯提高 abnormal 正类权重，并不足以在当前协议下稳定带来新的 gate 收益。

### 6.4 Focal BCE

最佳变体：`P2`

- `Spec@R99.5 = 0.440476`
- `Spec@R99.0 = 0.535714`

整体弱于 `H0`。

说明在当前第一阶段上，`Focal` 这类通用 hard-example emphasis 并没有自动转化成更好的 fixed-recall filtering 能力。

### 6.5 Defect oversampling

最佳变体：`P3`

- `Spec@R99.5 = 0.404762`
- `Spec@R99.0 = 0.535714`

这是本轮最弱的一条线。

它说明当前第一阶段的主要瓶颈并不是“abnormal 看得不够多”，而是：

> 顽固 normal 的伪缺陷结构仍然在干扰边界

## 7. 一个非常重要的新现象：P0/P2 关系变了

这轮 suite 最重要的附加信息，不只是“谁赢了”，而是：

> 不同训练策略会改变 `P0` 和 `P2` 谁更强。

在旧主线 `H0` 中：

- `P2` 明确优于 `P0`

但在新 suite 里：

- `HardMix` 最优变体是 `P0`
- `WBCE` 最优变体也是 `P0`
- `Selective` 最优变体甚至是 `P1`
- 只有 `Focal` 还维持 `P2` 最优
- `DefectOS` 的最优变体则变成了 `P3`

这说明：

> `trust` 并不是一个在所有训练分布下都固定增益的万能后处理层。

更准确地说：

- 在旧的 `hn02` 主线下，trust 是有效补丁
- 但当训练侧样本组织已经显著改变后，plain calibrated score 可能已经足够好
- 此时再叠加 trust，未必继续增加收益，甚至可能过修正

这也是为什么第二张图 `stage1_maxfilter_suite_p0_p2_behavior.png` 很关键。
它不是在证明 trust 无效，而是在证明：

> trust 的收益依赖于训练后形成的 score geometry，不是绝对固定的。

## 8. 对当前第一阶段的最终判断

到目前为止，第一阶段已经先后经历了：

- calibration
- HN 比率扫描
- cross-capacity HN 验证
- PTSG 第一轮
- 多原型 next-wave
- SupCon 强 embedding
- max-filter suite

从这条证据链看，当前最稳的结论是：

1. 第一阶段确实还能继续优化，但剩余空间已经不大。
2. 当前最有效的新增益来自：
   - **更细的 hard positive + hard normal mining**
3. 当前最优 stage-1 训练侧扩展不是：
   - selective loss
   - WBCE
   - focal
   - defect oversampling
4. 在这一轮新训练策略下，最优 gate 决策信号退回到了：
   - **plain calibrated p_abnormal**

因此，现阶段最合理的新主线候选是：

> `yolo11l-cls + calibration + hn02 + hard positive + hard normal mining`

并在其最终工作点选择中直接使用：

> `P0 = calibrated p_abnormal`

## 9. 对论文写作的建议口径

这轮结果在论文里不应写成：

> 本文提出了一个革命性新损失

因为数据不支持这种说法。

更稳的写法应是：

> 在已经完成 calibration、HN 回流与 PTSG 验证后，本文进一步从训练目标与样本组织两个层面测试第一阶段是否仍有剩余 headroom。结果表明，在当前统一协议下，直接面向工作点的 recall-constrained loss 并未形成综合最优，而通过同时强化训练侧 hardest normal 与 hardest positive 的 hard-mix 训练策略，第一阶段在保持 `Spec@R99.5` 不退化的前提下，进一步提升了 `Spec@R99.0`、`Prec@R99.0` 并降低了 `PTR@R99.0`。这说明第一阶段残余增益主要来自更精确的 failure-distribution 暴露，而非继续堆叠更复杂的 post-hoc 修正或简单的 abnormal oversampling。 

## 10. 当前一句话结论

> 本轮 max-filter suite 证明：第一阶段仍有少量可挖增益，但最有效的方向不是更重的通用损失，而是更细的 hard positive + hard normal mining；在该训练分布下，plain calibrated score 已经优于额外 trust 修正，因此当前新的 stage-1 最优候选应收敛为 `hardmix + P0`。
