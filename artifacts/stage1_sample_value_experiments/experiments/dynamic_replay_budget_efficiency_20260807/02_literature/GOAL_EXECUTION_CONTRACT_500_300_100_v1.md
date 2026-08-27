# Goal Execution Contract: 500 / 300 / 100 Literature Audit

## 1. Status And Purpose

```text
contract_version: 1.0
status: ACTIVE_RESEARCH_NO_TRAINING
owner_experiment: dynamic_replay_budget_efficiency_20260807
formal_training: FORBIDDEN
engineering_gate: FORBIDDEN
blind_holdout: SEALED
```

本合同是当前 active goal 的不可删减验收条件。目标不是积累论文数量，而是回答：

> 在固定且严格匹配的有限 replay budget 下，训练过程中哪些样本或样本集合值得额外学习，哪些选择与调度机制有可能跨 seed 稳定优于动态随机回流，并且不伤害弱 defect 尾部？

任何一项硬门禁未通过，goal 不得标记完成。

## 2. Exact Reading Sets

三个集合必须严格嵌套：

```text
DEEP_100 subset of SCREENED_300 subset of BROAD_500
|BROAD_500| = 500
|SCREENED_300| = 300
|DEEP_100| = 100
```

检索命中、重复版本、无关论文、只看标题、只读取二手摘要、无法确认原始来源的记录均不计入 500。

### 2.1 BROAD_500

每篇必须实际检查原始标题、摘要、引言问题定义和结论，形成独立 `Pxxxx.md`。必填内容：

- 唯一论文身份：规范化标题、作者、年份、venue、DOI/OpenReview/arXiv/官方 proceedings URL；
- 来源类型及检索日期；
- 用自己的中文表述写出的摘要，不得复制原摘要充数；
- 论文研究的问题、核心思路和主要结论；
- 与 Stage1 问题的直接相关链条；
- 至少一条可迁移机制或明确反证；
- 纳入理由和不能推出的结论；
- 阅读范围，明确列出实际检查的章节。

仅“可能相关”但无法说清训练样本价值、replay、数据选择、训练动态、目标方向、覆盖、噪声、预算或随机对照关系的论文，写入排除表，不计入 500。

### 2.2 SCREENED_300

必须来自 BROAD_500，并进一步阅读方法和实验章节。除上面字段外，每篇必须补充：

- 核心公式、伪代码或可执行算法步骤；
- 样本价值或选择变量的精确定义；
- 选择发生的训练阶段和刷新频率；
- budget 单位、unique count、repeat/exposure 和训练计算量；
- 随机基线是否存在，是否匹配预算、训练步数和刷新时机；
- seed/repetition 数量与不确定性报告方式；
- 主要数据集、模型和任务；
- 至少一个精确实验结果或表格定位；
- 消融、负结果、失败条件和作者承认的局限；
- 与 Stage1 的相同点、不同点和迁移风险；
- `REPLICATION`、`INSPIRED_ADAPTATION`、`MECHANISM_ONLY` 或 `NOT_TRANSFERABLE` 分类。

缺少方法或实验正文的论文不能进入 300。没有随机基线的论文可以进入，但必须显式标为不能证明优于随机。

### 2.3 DEEP_100

必须来自 SCREENED_300，并取得可核验全文。每篇必须逐节精读，不得只依赖摘要页、博客、综述或他人笔记。除上面字段外，每篇必须补充：

- PDF 或官方全文页面 SHA-256；
- 实际阅读的章节和页码范围；
- 至少三个可回到原文定位的 evidence anchors；
- 关键公式的变量含义、假设和推导关系；
- 算法流程、复杂度、状态依赖和随机性来源；
- 数据划分、选择集、验证集和测试集角色；
- 是否存在 leakage、oracle selection 或 winner's curse 风险；
- 完整随机对照、公平预算和 checkpoint 选择细节；
- seed 级结果、均值之外的最差情形或方差信息；
- 关键消融及其实际支持和不支持的机制；
- 实现所需字段、接口、计算成本和可复现参数；
- 对当前专家仓库、当前 v3 和下一版 canonical pipeline 的逐项代码映射；
- 一段独立批判性综述，说明为什么该论文可能不适用于 Stage1。

拿不到全文、全文解析失败、无法给出原文定位或只复述作者结论的论文不能进入 100。

## 3. Relevance Contract

行业不限，但每篇计数论文必须直接属于至少一个机制域：

```text
training dynamics and learning speed
reducible or irreducible loss
curriculum, self-paced learning, and Goldilocks difficulty
data valuation, influence, gradients, or target alignment
active learning and uncertainty with diversity
coreset, pruning, coverage, and redundancy
replay, continual learning, rehearsal, or data echoing
noisy labels, shortcuts, harmful or unlearnable samples
class imbalance, Neyman-Pearson, pAUC, or high-recall local objectives
video/stream redundancy and source-aware sampling
budget-aware stochastic sampling and repeated-random baselines
distributional effects across seeds and randomized interventions
```

跨行业论文必须明确写出可迁移的底层机制。仅仅出现 `sample`、`hard`、`gradient`、`active` 或 `replay` 等关键词不构成相关性。

## 4. Deduplication And Identity

- 同一工作从 arXiv 到会议/期刊版本只计一篇，优先最终正式版本；
- 标题变更通过 DOI、作者、摘要和方法身份合并；
- workshop extended abstract 与正式全文重复时不重复计数；
- survey 与其引用的原始研究分开登记，但 survey 不能替代原始论文证据；
- 每篇使用稳定 `P0001`--`P0500` ID，ID 一经发布不复用；
- 所有重名、合并和版本选择写入 `DEDUPLICATION_LEDGER.csv`。

## 5. Required Per-Paper Artifacts

每篇计数论文必须有独立 Markdown：

```text
review_500_300_100_v1/papers/Pxxxx.md
```

不得用一张总表代替逐篇笔记。每个 Markdown 同时包含：

```text
Bibliographic identity
Reading level and evidence status
Chinese abstract
Critical mini-review
Direct relevance
What is supported
What is not supported
Stage1 transfer boundary
Source and hash evidence
```

进入 300 或 100 时在同一文件上增加对应章节，保留升级历史和阅读日期，不复制成多个相互漂移的笔记。

## 6. Anti-Fake-Reading Gates

自动验证器至少检查：

1. 精确计数和 `100 subset 300 subset 500`；
2. DOI/标题/版本去重；
3. 每篇 Markdown 存在且 ID 与账本一致；
4. 每层必填字段非空且不是 `TODO/TBD/unknown/待补/同上` 等占位内容；
5. 100 篇全部有可读取全文、SHA-256、章节/页码和 evidence anchors；
6. 300 篇全部有方法、实验、budget、random baseline、seed、结果和局限记录；
7. 500 篇全部有独立中文摘要、批判性小综述和直接相关链；
8. URL 可追溯到论文原始来源，二手页面不能作为唯一来源；
9. 检测大段重复模板、跨论文复制和与标题明显不匹配的笔记；
10. 检测虚构数字：精确结果必须带表号、图号、页码或章节定位；
11. 检测把 abstract-only 误标为 full-text；
12. 检测无关论文和没有迁移链的记录；
13. 随机抽检至少 10% 的 500、15% 的 300、20% 的 100，重新对照原文；
14. 对最关键 30 篇做第二遍独立事实核对；
15. 验收报告列出所有失败项，不能在存在失败时输出 PASS。

字数不能单独证明阅读。验收以可回到原文的具体证据、方法细节、实验边界和跨论文一致性为主。

## 7. Expert Material Contract

专家原话、交付报告和审查结论必须阅读并登记，但不自动视为正确。每项重要主张分成三列：

```text
expert_claim
code_or_reproduction_evidence
current_v3_evidence
```

专家 v1.0.0 的第一次“完整可运行”声明与第二次 `NO-GO` 审查必须共同保留。源码缺失时只能标记 `SOURCE_NOT_AVAILABLE`，不得根据报告猜测源码行为。专家引用的论文只有在独立核对原文后才能进入 500/300/100。

## 8. Final Literature Deliverables

必须产出：

```text
BROAD_500_INDEX.md
SCREENED_300_INDEX.md
DEEP_100_INDEX.md
READING_LEDGER.csv
DEDUPLICATION_LEDGER.csv
SCREENING_EXCLUSIONS.csv
PAPER_NOTE_VALIDATION.json
RANDOM_AUDIT_REPORT.md
KEY_30_SECOND_PASS.md
EVIDENCE_MATRIX.csv
NEGATIVE_RESULTS_AND_FAILURES.md
RANDOM_BASELINE_EVIDENCE.md
SYSTEMATIC_LITERATURE_REVIEW.md
EXECUTIVE_SYNTHESIS.md
```

`SYSTEMATIC_LITERATURE_REVIEW.md` 必须综合而不是简单罗列，至少回答：

- 难度、可学习性、可靠性、方向和集合覆盖分别能解释什么；
- 哪些代理在严格随机对照下失败；
- 哪些方法直接在 replay 中优于随机，证据有多少 seed；
- 为什么视频重复和预算语义可能决定结果；
- 哪些机制最值得先做最小干预实验；
- 哪些结论在 Stage1 上仍然未知。

## 9. Code And Experiment Deliverables

文献门禁通过后，才能冻结：

- 专家 v1.0.0、专家 NO-GO 审查和当前 v3 的行级对比报告；
- 唯一 OOF 数据语义；
- train/val_target/val_model/val_cal/val_op/test 角色合同；
- 固定 checkpoint 和双阈值语义；
- replay budget、unique samples、per-epoch slots 和累计实际曝光；
- Q/R/A/D 门控、分层和集合策略，不使用任意加权综合分；
- 动态 repeated-random、method-matched random 和 no-replay 对照；
- 最小机制筛选、独立 seed 确认和停止规则；
- 下一版仓库改造规格和 TDD 验收清单。

## 10. Completion Rule

只有同时满足以下条件才允许把 active goal 标记为 complete：

```text
expert_delivery_audit = PASS
expert_claim_code_v3_comparison = PASS
broad_500_exact = PASS
screened_300_exact = PASS
deep_100_exact = PASS
nesting_and_deduplication = PASS
per_paper_markdown = PASS
source_and_hash_evidence = PASS
random_audit = PASS
key_30_second_pass = PASS
systematic_review = PASS
scientific_contract = PASS
implementation_spec = PASS
desktop_mirror_and_independent_validation = PASS
formal_training_started = false
blind_holdout_opened = false
```

接近完成、文献数量不足、精读证据不完整、源码缺失、时间紧张或已有总结看起来合理，都不能替代上述 PASS。
