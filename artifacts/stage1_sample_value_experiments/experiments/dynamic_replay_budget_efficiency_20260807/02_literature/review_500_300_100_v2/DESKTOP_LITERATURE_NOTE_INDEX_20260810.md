# Stage1 有限预算动态回流：文献笔记总入口

## 先读结论

当前文献已由用户宣布足够，停止继续检索和扩充。下一阶段直接进入科学综合、
Q/R/A/D 合同、可证伪实验矩阵、统计规则和仓库施工规格。

科学主线固定为顺序门控或析因，而不是任意加权总分：

1. `Q`：标签与来源是否可靠，是否可能是噪声、不可学或异常样本。
2. `R`：当前模型是否仍未学会、但独立参考表明它具有可约学习空间。
3. `A`：训练方向是否有利于独立的 FN95 局部目标，而非只有大 loss 或大梯度。
4. `D`：加入当前集合后是否增加来源、视频、特征或梯度覆盖，而非近重复。

confidence、loss、RHO、gradient、forgetting、AUM 和 coverage 都只是候选
信号。只有固定基础训练与固定累计 replay 曝光下，对 R1、R2、current-loss
和 no-replay 的真实配对干预，才构成 utility evidence。当前不得声称候选方法
有效。

## 本镜像包含什么

- `01_核心综合`：建议最先阅读，包含文献综合、范围冻结、中文 14 篇继承说明、
  v4 canonical 修复与证据边界。
- `02_结构化全文记录_31`：31 篇当前 v4 结构化记录，包含方法、实验、原文锚点
  和 Stage1 迁移边界。
- `03_历史全文笔记_50`：上一轮 50 篇全文证据笔记，作为继承候选材料。
- `04_BROAD笔记_500`：去重后 BROAD staging 的 500 篇逐篇 Markdown 笔记。
- `05_台账与验证`：阅读台账、证据矩阵、队列 receipt、验证结果和中文 14 篇
  历史文件身份台账。
- `MANIFEST_SHA256.csv`：镜像内每个文件的相对路径、字节数和 SHA-256。
- `PACKAGE_VALIDATION.json`：文件计数、总字节和哈希验证结论。

## 数量口径

- BROAD staging 笔记：500 份。
- SCREENED PRIMARY 全文队列：300 篇，另有 20 篇 reserve-read。
- 当前 v4 结构化全文记录：31 份。
- 历史全文笔记：50 份。
- 用户中文批次：14 篇唯一论文；上一轮按批次报告 6 篇方法级精读、8 篇
  背景/迁移阅读。

这些集合有重叠，文件数不能相加当作唯一论文数。中文批次的原 PDF 当前缺失，
其历史身份已登记，但不用于冒充当前全文门禁。

## 机器门禁边界

“当前文献足够”是停止继续扩充的研究范围决定，不等于旧的形式化
`DEEP_100 subset SCREENED_300 subset BROAD_500` completion audit 已经 PASS。
任何后续报告都必须同时呈现用户范围决定和真实 validator 状态。

本 Desktop 目录只是镜像。事实源仍位于已登记实验目录：

`C:\GitHub\YOLO-CV\artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\02_literature\review_500_300_100_v2`

