# 第一阶段 Gate 理论基础与 Max-Filter 套件报告

## 0. 文档目的

这份文档是当前项目第一阶段 `stage-1 gate` 方向的内部研究报告，不是最终论文正文。

它的作用不是替代论文，而是把下面这些东西一次性讲清楚：

- 第一阶段到底在优化什么
- 为什么第一阶段不能按普通二分类去理解
- 当前 `calibration -> hn02 -> P2` 这条主线为什么成立
- 为什么后面还可能存在更“硬”的方法
- 这次新增的 `max-filter suite` 到底在测什么
- 各条方法线分别解决哪类问题，风险在哪里
- 后面论文收口时，哪些话可以写，哪些话不能乱写

这份文档会比论文写得更细、更啰嗦一些。
目的不是追求简洁，而是把思路和方法基础留成可复查记录，避免最后只剩几个结论，没人记得它们为什么成立。

## 1. 第一阶段到底是什么

第一阶段不是最终缺陷检测器。
第一阶段是两阶段系统里的**高召回门控器**。

它的职责可以写成两句话：

1. 尽量不要把真实缺陷过早挡掉
2. 在这个前提下，尽量多过滤正常背景

所以第一阶段真正回答的问题不是：

> 这个二分类模型默认阈值下准不准

而是：

> 在缺陷召回固定住的前提下，它能多拦下多少 normal

这就是为什么第一阶段不该主要盯着：

- `top1_acc`
- `accuracy`
- 默认阈值 `F1`

而是要主要盯着：

- `Spec@R99.5`
- `Spec@R99.0`
- `Prec@R99.0`
- `PTR@R99.0`

## 2. 第一阶段的混淆矩阵到底怎么读

当前 binary gate 里：

- 正类 = `abnormal / defect`
- 负类 = `normal`

但第一阶段里的 `TP/TN/FP/FN` 不能按普通分类课本那种抽象语气去读，最好直接按 gate 语义来理解。

### 2.1 TP

`TP` 表示：

- 样本真实是缺陷
- 第一阶段也把它保留下来，当成 abnormal

也就是说，这个缺陷样本被正确送到了第二阶段。

### 2.2 FN

`FN` 表示：

- 样本真实是缺陷
- 第一阶段却把它挡成了 normal

这就是第一阶段最危险的错误。
通俗地说，就是：

> 真缺陷被当成背景提前拦掉了

### 2.3 TN

`TN` 表示：

- 样本真实是 normal
- 第一阶段成功把它挡掉了

这就是第一阶段最有价值的正确结果，因为它直接减少了第二阶段的无效输入。

### 2.4 FP

`FP` 表示：

- 样本真实是 normal
- 第一阶段却把它当成 abnormal 放过去了

这不是最终意义上的“漏检”，但对第一阶段来说仍然是错误，因为它没能把本该拦住的正常背景拦住。

所以在当前项目里，最实用的翻译是：

- `TN` = 成功拦下的 normal
- `FP` = 仍被放行到第二阶段的 normal
- `FN` = 被误挡成背景的 defect

## 3. 为什么主指标一定是 Spec@R99.5 / Spec@R99.0

第一阶段的比较逻辑不是“谁全局最好”，而是“谁在目标工作点附近最好”。

先写公式，再解释含义：

- `Recall = TP / (TP + FN)`
- `Specificity = TN / (TN + FP)`
- `Precision = TP / (TP + FP)`
- `PTR = (TP + FP) / N`

这里的 `PTR` 是 pass-through rate，也就是最终仍送到第二阶段的样本比例。

从工程角度看：

- `Recall` 越高，说明越少缺陷被挡掉
- `Specificity` 越高，说明越多 normal 被过滤掉
- `Precision` 越高，说明被放过去的样本里“真异常”占比更高
- `PTR` 越低，说明第二阶段负担越小

当前项目对第一阶段的排序规则已经固定为：

1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0` 越低越好

这套规则其实已经把第一阶段的系统目标写死了：

- 先保证 defect recall 不掉
- 再比较谁拦 normal 更强
- 再比较放行池子干不干净
- 最后比较第二阶段负担大不大

这也是为什么训练日志里的 `top1_acc` 从来不能直接拿来当最终裁判。

## 4. 为什么第一阶段不能按普通二分类理解

如果把第一阶段当作普通二分类，那么最自然的动作会是：

- 追 `accuracy`
- 追默认阈值 F1
- 追全局 AUROC
- 用最常见的分类 loss 做泛化优化

但这会错位。

因为第一阶段不是最终裁判，它更像：

- selective classification
- reject option
- safe-normal filtering

第一阶段真正要做的是：

> 只有当样本足够安全地像 normal 时，才把它挡掉；否则宁可放给第二阶段

因此，第一阶段的本质不是“把所有样本尽量分对”；
而是：

> 在高召回约束下，尽量构造一个更大的安全 normal 区域

这个任务定义，会自然推出为什么下面这些东西都重要：

- calibration
- HN backflow
- trust score
- operating-point evaluation
- selective / recall-constrained loss

## 5. 当前这条主线已经证明了什么

在这次 max-filter 套件之前，第一阶段其实已经走出了一条很完整的证据链。

### 5.1 calibration 是必要的

当前固定协议是：

- `val-cal = 30%`
- `val-op = 70%`
- `seed = 20260330`

它的作用不是改变 backbone，而是把分数刻度统一起来，让不同模型、不同训练策略的阈值比较更可信。

所以 calibration 改的是：

- score scale
- threshold selection quality
- operating-point stability

不是改模型本体。

### 5.2 HN 2% 已经证明有效

这里的 `HN` 不是把验证集里的错样本拿回去训。

它的准确含义是：

1. 用当前 gate 模型去重新扫描 `train/Normal`
2. 找出最像 abnormal 的 true normal
3. 把这批 hardest normal 少量回流到下一轮训练数据里

换句话说：

> HN 回流 = 训练侧 hard-normal mining + 轻度过采样

它不是：

- 验证集回灌
- 改标签
- 用 val 错样本特调模型

当前比率扫描已经把主线固定为：

- `hn02`

说明最值钱的不是继续堆 normal 数量，而是让模型更充分学习最危险的 normal 尾部。

### 5.3 P2 证明了 trust 有用

第一轮 PTSG 实验已经证明：

- `P2 = calibrated p_abnormal + trust`

优于只用校准后异常概率。

这件事的意义很大，因为它说明：

> 第一阶段是否拦截一个样本，不应该只看一维概率分数

而应该同时考虑：

- 这个样本分数多高
- 它在特征空间里更像 normal 还是 abnormal

也就是说，第一阶段确实受益于 embedding geometry。

### 5.4 轻量 post-hoc 深挖已经出现边际递减

后面我们又试了：

- 多原型 trust
- margin trust
- SupCon 强 embedding

结果都没有明确稳定超过当前最优 `P2`。

这件事不是坏消息，而是重要的边界信息：

- 第一阶段不是没有继续卷
- 而是已经卷到一定程度后，边际收益开始变小

这就为现在的 max-filter suite 提供了合理背景：

> 如果还想继续压第一阶段，就该从“训练目标”和“样本组织”这种更系统的层面下手，而不是继续微调轻量 post-hoc 公式。

## 6. 为什么不能把 val 最难样本拿回去训练

这个问题必须单独说清楚，因为它在工程上很诱人。

直觉上会有人说：

> 既然 val 里有最难的样本，为什么不把它们挪到 train，再从外面补一点验证样本回来

问题不在于“数量够不够”，而在于：

> 你已经使用了验证信息来挑训练样本

这会导致：

- data leakage
- selection bias
- 验证结果不再干净

即使你后来又从外面补了新样本，原来的验证协议也已经变了。

从研究可信度角度看，这样得到的模型更像：

> 对验证集特调过的模型

而不是：

> 在未见数据上保持泛化说服力的模型

所以当前项目坚持只改这些“方法变量”：

- 回流比例
- loss 类型
- mining 策略
- trust 公式
- 阈值规则

而不把验证集信息回灌进训练过程。

## 7. 支撑当前阶段思路的理论线

当前 stage-1 的方法判断，不是拍脑袋来的，而是能分别落到几条成熟的方法线里。

### 7.1 selective classification / reject option

这条线支持的是：

> 第一阶段可以被正式建模成“选择性门控”，而不是普通分类器

代表性工作：

- Geifman and El-Yaniv, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, PMLR 2019

核心启发是：

- 模型不一定要对所有样本同样激进地下最终判决
- accept / reject 本身可以成为建模目标

这和第一阶段高度一致，因为第一阶段本来就是：

- safe-normal => 过滤
- not safe enough => 放行给第二阶段

### 7.2 trust score / 几何一致性

这条线支持的是：

> 原始 confidence 不是唯一可靠信号，几何一致性可以提供额外可信度信息

代表性工作：

- Jiang et al., *To Trust Or Not To Trust A Classifier*, NeurIPS 2018

它支持现在的 `P2` 口径：

- 不只看 `p_abnormal`
- 还看 embedding 里更像哪一类

### 7.3 prototype-based geometry

这条线支持的是：

> 用类原型或类中心描述 embedding 空间结构是合理的

代表性工作：

- Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017

它给当前 prototype-trust 逻辑提供了几何基础。

### 7.4 hard example reweighting

这条线支持的是：

> 训练时不应让 easy 样本主导梯度

代表性工作：

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017

虽然 focal 最早是为 dense detection 提出的，但“easy sample 主导训练、hard sample 需要更大权重”这个逻辑，对 stage-1 一样成立。

### 7.5 contrastive feature structuring

这条线支持的是：

> selective gate 的表现不只是最后一层阈值问题，特征层本身也可能影响 gate 质量

代表性工作：

- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020
- Wu et al., *Confidence-aware Contrastive Learning for Selective Classification*, PMLR 2024

这也是我们为什么做过 SupCon 强 embedding 路线。
虽然当前它没有赢，但它不是乱试，而是有明确理论支撑。

## 8. 为什么第一阶段仍然可能有更硬的方法

当前第一阶段已经很强，不代表它没有更硬的方法。

理论上还可以继续优化的部分包括：

- 训练目标
- 样本分布
- hard example 暴露方式
- embedding 空间结构
- score-to-decision 的映射

但问题不在于“还有没有方法”，而在于：

> 哪种方法最可能在当前协议下继续带来有意义的增益

既然：

- `hn02` 已经固定
- `P2` 已经有效
- `P5/P6` 没继续赢
- `SupCon` 没继续赢

那么下一步继续压第一阶段时，就应该优先从：

- 训练目标是否更贴合 gate
- hard 样本组织是否还能更精准

这两个方向去下手。

这就是当前 `max-filter suite` 的出发点。

## 9. 当前 max-filter 套件的四个层级

这次新增的一键套件，本质上是在测试四类方法族。

### 9.1 第一档：selective / recall-constrained loss

#### 它想解决什么

当前第一阶段最危险的错误不是“普通误分”，而是：

> 把 defect 错挡成 normal

所以一个更贴合 stage-1 任务的 loss，不应只做平均意义上的分类优化，而应对正类 recall 更敏感。

#### 当前实现

现在新增的 loss 类型是：

- `recall_constrained_bce`

实现位置：

- `YOLOv11/ultralytics/utils/loss.py`

它的逻辑是：

1. 用加权 BCE 做基础项
2. 只对 abnormal 正类再加一个 margin penalty

当前近似公式可以写成：

```text
L = BCE(logit, target) + penalty * relu(margin - abnormal_logit_on_positive)
```

它的含义不是“让所有样本都更大分”，而是：

> 如果正类 abnormal 的 logit 还不够安全，就额外惩罚

当前默认参数是：

- `cls_pos_weight = 1.5`
- `cls_recall_margin = 1.0`
- `cls_recall_penalty = 0.5`

#### 为什么它排第一档

因为它是当前套件里唯一真正从**目标函数层面**去贴合第一阶段 gate 任务的路线。

它不是简单调权重，而是在逼近这样一个思想：

> 高召回约束先满足，再去做 normal 过滤

### 9.2 第二档：更细的 hard positive / hard normal mining

#### 它想解决什么

我们已经知道 hard normal 很关键。
但 stage-1 还有另一类样本也值得盯：

- hard positive

也就是：

- 真 abnormal
- 但当前模型打得不够像 abnormal

因此第二档不是只挖 hard negative，而是同时挖：

- hard negative：`normal` 但 `p_abnormal` 很高
- hard positive：`abnormal` 但 `p_abnormal` 很低

#### 当前实现

当前套件先通过：

- `scripts/stage1_score_train_samples.py`

给训练侧全部样本打分，输出：

- `train_sample_scores.csv`
- `hard_negative_candidates.csv`
- `hard_positive_candidates.csv`

然后通过：

- `scripts/stage1_build_augmented_gate_dataset.py`

构建 hard-mix 数据集。

当前默认参数是：

- `hard_negative_top_k = 22`
- `hard_negative_repeat = 1`
- `hard_positive_top_k = 22`
- `hard_positive_repeat = 1`

也就是：

- 额外再复制一小批 hardest normal
- 额外再复制一小批 hardest abnormal

#### 为什么它排第二档

因为它仍然非常对题，依旧直指当前最可能出错的尾部样本；
但它本质上还是数据组织层面的干预，不如第一档那样直接改变优化目标。

### 9.3 第三档：weighted BCE / focal BCE

#### 它们想解决什么

这两条线是更常见、更通用的 hard example / class imbalance 处理方式。

`weighted_bce` 的意思是：

- 正类 abnormal 更重要，训练时权重大一点

`focal_bce` 的意思是：

- easy 样本少说话
- hard 样本多说话

#### 当前实现

当前已有配置：

- `stage1_gate_l_hn_wbce_suite.json`
- `stage1_gate_l_hn_focal_suite.json`

默认参数是：

- `weighted_bce`: `cls_pos_weight = 1.5`
- `focal_bce`: `gamma = 2.0`, `alpha = 0.25`

#### 为什么它们只排第三档

因为它们有用，但比较泛。

它们更像是在说：

> 训练里不要让 easy 样本占太大便宜

而不是在说：

> 第一阶段这个 gate 在固定 recall 工作点下到底该怎么优化

所以它们应该做，但不应该抢走主叙事。

### 9.4 第四档：defect oversampling

#### 它想解决什么

这条线的想法很直接：

- abnormal 样本多看一点
- 也许 recall 会更稳

#### 当前实现

同一个数据集构建脚本可以对全部 abnormal 训练样本做额外复制。

当前默认参数：

- `abnormal_repeat_all = 1`

#### 为什么它排最后

因为按当前实证判断，第一阶段最痛的瓶颈并不是：

> abnormal 看得还不够多

而是：

> 有一小撮 normal 的伪缺陷特征太顽固

如果盲目强化 abnormal，风险是：

- recall 稳一点
- 但模型更容易把 normal 往 abnormal 一侧推
- specificity 反而掉

所以这条线值得测，但理论优先级确实不如前面三档。

## 10. 当前一键流水线到底做了什么

当前默认入口已经改成：

```powershell
uv run main.py
```

默认任务是：

- `stage1_gate_maxfilter_suite`

配置文件是：

- `YOLOv11/configs/runtime/stage1_gate_maxfilter_suite.json`

整条流水线顺序是：

1. 用当前最佳 `yolo11l + hn02` miner 模型给训练侧全量样本打分
2. 构建 hard-mix 数据集
3. 构建 defect-oversample 数据集
4. 训练 selective-loss 版本
5. 训练 hard-mix 版本
6. 训练 weighted-BCE 版本
7. 训练 focal-BCE 版本
8. 训练 defect-oversample 版本
9. 每个版本训练完都自动导出特征
10. 自动重建 prototype bank
11. 自动重跑 PTSG 评估
12. 自动汇总总表

这个设计很重要，因为它保证了：

> 所有候选方法最终都在同一套 PTSG 与 operating-point 协议下比较

这样比较的是：

- 不同训练策略是否让最终 gate 变强

而不是：

- 每个方法随便带一套后处理，最后谁的故事更好看

## 11. 当前套件里有哪些实验

当前 suite 固定包含 5 组：

1. `Selective / recall-constrained loss`
2. `Hard positive + hard normal mining`
3. `Weighted BCE`
4. `Focal BCE`
5. `Defect oversampling`

对应配置分别是：

- `stage1_gate_l_hn_selective.json`
- `stage1_gate_l_hardmix.json`
- `stage1_gate_l_hn_wbce_suite.json`
- `stage1_gate_l_hn_focal_suite.json`
- `stage1_gate_l_defectos.json`

每组训练完成后，都会进入同一条后处理链：

- export features
- build bank
- run PTSG
- compare against `H0 current best hn02 + P2`

所以这个 suite 的科学问题写得很清楚：

> 在固定主模型、固定 hn02、固定 calibration、固定 trust 评估协议的前提下，训练侧再做哪些改动还能让 gate 继续变强？

## 12. 结果出来后应该怎么读

结果出来后，不能先看：

- 谁训练 loss 最低
- 谁 `top1_acc` 最好
- 谁训练曲线更漂亮

正确读法是：

> 谁在同一套 stage-1 排名规则下，进一步提高了 normal 拦截能力

### 12.1 如果第一档赢

说明：

- stage-1 还值得从目标函数层面继续优化
- 简单的 generic reweighting 不够
- recall-aware loss 确实更贴合 gate 本质

### 12.2 如果第二档赢

说明：

- 当前瓶颈更像 failure distribution exposure 问题
- 模型还需要更精准地“多见几次最危险的样本”

### 12.3 如果第三档赢

说明：

- 通用型 reweighting 已经够用
- 不一定需要特别重的定制 loss

### 12.4 如果第四档赢

说明：

- abnormal under-exposure 比此前判断得更严重

### 12.5 如果谁都没明显赢

说明：

- 第一阶段在当前数据、当前协议下基本接近饱和
- 不应再继续大量投入主线精力
- stage-2 detector 才应该继续承担主要创新空间

这不是坏结论，反而是成熟研究该有的收口方式。

## 13. 哪些步骤需要 GPU

这个也要提前说清楚，不然容易把所有步骤都误判成高成本训练。

### 13.1 明显需要 GPU 的部分

- 任何重新训练
- 大规模特征导出

### 13.2 基本不依赖 GPU 的部分

- 数据集重组
- 候选样本排序
- prototype bank 构建
- calibration
- threshold sweep
- summary 汇总

所以当前套件真正吃 GPU 的地方，主要还是训练本身，而不是后处理分析链。

## 14. 对论文写作意味着什么

后面写进论文时，不应该忽然蹦出一堆公式。
更合理的顺序是：

1. 先重申第一阶段的系统职责
2. 再讲 gate 语义下的 `TP/TN/FP/FN`
3. 再讲为什么工作点比默认阈值更重要
4. 再讲 HN 回流的真正含义
5. 再讲 trust 为什么有效
6. 最后再讲为什么 loss / mining / oversampling 是合理延伸
7. 到最后一步才给公式

这样读者先理解：

- 你到底在优化什么

再去看：

- 你拿什么数学手段去优化它

这会比“先上公式，后补任务定义”稳很多。

## 15. 当前最小参考文献清单

如果后面要把这条线写进论文或整理成方法说明，下面这些参考足够支撑当前思路：

- Geifman, Y. and El-Yaniv, R. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. PMLR, 2019.
- Jiang, H., Kim, B., Guan, M., and Gupta, M. *To Trust Or Not To Trust A Classifier*. NeurIPS, 2018.
- Snell, J., Swersky, K., and Zemel, R. *Prototypical Networks for Few-shot Learning*. NeurIPS, 2017.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. *Focal Loss for Dense Object Detection*. ICCV, 2017.
- Khosla, P., Teterwak, P., Wang, C., et al. *Supervised Contrastive Learning*. NeurIPS, 2020.
- Wu, Y.-C., Lyu, S.-H., Shang, H., Wang, X., and Qian, C. *Confidence-aware Contrastive Learning for Selective Classification*. PMLR, 2024.

## 16. 当前一句话总结

当前第一阶段的最准确理解是：

> 第一阶段不是追求普通二分类总体最优，而是在固定高召回约束下尽量扩大 safe-normal 过滤区域；因此，任何值得保留的方法，都必须最终证明自己能在相同 recall 纪律下拦下更多 normal，而不是只拿到更好看的分类日志。

这就是当前 `max-filter suite` 的理论出发点，也是后面判断第一阶段是否该继续深挖的统一标准。
