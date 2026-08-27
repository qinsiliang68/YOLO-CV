# Stage1 核心方法锚点合同 v2

## 目的

`BROAD-500` 不能只满足数量、随机顺序和 RQ 配额，还必须覆盖能够定义研究问题的奠基方法、严格负结果、强随机基线和业务指标理论。上一版从首批 575 个盲序候选中选出 500 篇，身份与来源可审计，但遗漏了 Dataset Cartography、Example Forgetting、GLISTER、D2 Pruning、TracIn、Neyman-Pearson、OHEM 等概念骨架。因此旧 `broad_freeze_v2` 只保留为可复用的暂存证据，不是最终 BROAD 冻结版。

## 锚点纳入原则

一篇论文只有满足以下至少一项，才可进入锚点表：

1. 定义 Stage1 必须区分的核心概念，例如训练动态、剩余可学习性、目标方向或集合覆盖；
2. 给出必须保留的严格负结果或强随机基线，能够反驳 `hardness = utility`；
3. 定义固定 FN 约束、局部 ROC/pAUC 或初始化依赖等不可缺少的推断条件；
4. 是后续方法反复比较的奠基方法，缺失会使文献综述无法解释方法谱系。

锚点不是“支持本项目观点的论文清单”。支持、反驳和混合结果采用同一来源门禁和阅读协议。锚点身份只保证必须被审查；它不绕过相关性判断，也不自动获得 BROAD、SCREENED 或 DEEP credit。

## 选择语义

- `CORE_METHOD_ANCHORS_v2.csv` 中 40 个身份必须全部完成主源核对和 BROAD 层人工判断。
- `BROAD_V2_ELIGIBLE` 表示已通过旧版 BROAD 人工核对，但仍须在锚点完整版本中重新绑定输入哈希。
- `DISCOVERED_NOT_BROAD_SCREENED` 表示候选库已有身份，但此前未进入人工审查批次。
- `EXTERNAL_PRIMARY_IDENTITY_VERIFIED_PENDING_SCREEN` 表示已从官方会议、PMLR、CVF、ACL、OpenReview 或 arXiv 核对身份，但尚未完成本地来源获取和 BROAD 阅读记录。
- 只有人工决定为 `ELIGIBLE_BROAD` 且来源验证通过的锚点，才进入 `mandatory_canonical_work_ids`。
- 合格锚点必须先于 RQ 配额和普通填充入选；缺失即失败。
- 锚点仍受 `maximum_transfer`、身份去重、主源、全文格式和必填证据约束，不能借“重要”绕过门禁。
- 被排除的锚点必须保留标题、主源、排除原因和阅读范围，不得从审计链中消失。

## 覆盖结构

| 家族 | 作用 | 数量 |
|---|---|---:|
| 训练动态与难度分型 | 区分 easy、ambiguous、slow、forgotten、hard-to-learn | 7 |
| 剩余可学习性与困难区间 | 检验 current loss 相对参考损失及中等难度窗口 | 3 |
| 方向、影响与目标效用 | 区分影响大小、梯度方向和集合效用 | 8 |
| 标签可靠性 | 防止将噪声或不可学尾部放大 | 4 |
| 多样性与覆盖 | 检验随机基线是否因覆盖优势获胜 | 6 |
| 时机、预算与随机基线 | 约束刷新、累计曝光和严格随机比较 | 8 |
| FN95 局部目标与状态依赖 | 定义约束指标及 seed/初始化依赖 | 4 |
| **合计** |  | **40** |

## 重建顺序

1. 获取并校验 21 个缺失/外部锚点主源；
2. 为这 21 篇建立独立补充人工审查批次；
3. 将通过的锚点与现有 519 个合格候选重新去重；
4. 以锚点优先、反证最低数、RQ 配额、transfer 上限、冻结随机键的顺序选出精确 500 篇；
5. 输出到 `staging/broad_freeze_anchor_complete_v3/`，不得覆盖 `broad_freeze_v2/`；
6. 再从新 BROAD 中重建精确 300 篇 SCREENED 队列，并复用身份一致、哈希一致的既有 PDF 提取结果。

## 禁止解释

- 锚点进入清单不表示论文方法适合 Stage1。
- 主源身份核对不表示读完方法或实验。
- PDF 下载成功不表示 SCREENED 或 DEEP 完成。
- 论文声称优于随机不表示在固定累计 replay exposure、配对 seed 和 FN95 目标下成立。
- 当前正式计数仍为 `BROAD=0, SCREENED=0, DEEP=0`，直到反伪造验证器逐项通过。
