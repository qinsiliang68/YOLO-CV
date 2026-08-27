# Stage1 Evidence-Driven Research Goal v2

## 1. Goal Identity

```text
goal_version: 2.0
owner_experiment: dynamic_replay_budget_efficiency_20260807
execution_state: ACTIVE_EVIDENCE_AUDIT
scientific_state: EVIDENCE_AUDIT_REQUIRED
formal_training: FORBIDDEN
engineering_gate: FORBIDDEN
pilot_release: FORBIDDEN
blind_holdout: SEALED
historical_contract: GOAL_EXECUTION_CONTRACT_500_300_100_v1.md
```

本文件是 v1 的澄清版，不删除、不篡改 v1。项目负责人已明确重新启动证据审查；该授权不包含正式训练、engineering gate、pilot release 或 blind holdout。

## 2. Single Objective

以可复现的代码证据、原始论文证据和严格匹配的随机对照原则，完成 Stage1 有限预算动态回流的研究定案，回答：

> 在固定基础训练过程和固定累计 replay 曝光预算下，怎样识别当前模型状态下仍可学习、对 FN95 业务目标方向有利、标签可靠且不冗余的样本集合，并用什么最小干预实验判断该策略能否跨未见 training seed 稳定优于严格匹配的随机回流和 no-replay？

本目标的完成产物是一个经过证据审计、另一名工程师可直接实施的科学合同和仓库改造规格，不是“候选方法已经有效”的结论，也不包含正式训练结果。

## 3. Fixed Research Semantics

以下概念必须分开，不得合并成任意加权总分：

```text
Q = reliability: 样本标签和观测是否可靠，是否疑似噪声、离群或不可学
R = residual learnability: 当前 run 尚未学会、但 cross-fitted OOF 参考表明可学的程度
A = target direction: 该样本更新方向是否有利于独立目标集上的 FN95 局部目标
D = set diversity: 在已经选择的集合条件下是否提供新的来源、视频、特征或梯度覆盖
```

Q/R/A/D 是顺序门控、分层变量或析因实验因素，不是：

```text
0.4 * Q + 0.3 * R + 0.2 * A + 0.1 * D
```

任何样本属性都只能称为 candidate signal。只有真实 replay 干预相对 R1、R2 和 no-replay 的配对结果，才能称为 sample/set utility evidence。

## 4. Work Package A: Expert Delivery Audit

逐项盘点 `C:\Users\28898\Downloads` 中的两轮专家材料：

1. 第一轮动态回流工程返回包、校验文件和交付声明；
2. BudgetedReplay v1.0.0 的交付报告、release/post-package validation 和 SHA ledger；
3. BudgetedReplay v1.0.0 独立 NO-GO 报告、Findings JSON、GO/NO-GO checklist、review evidence 和校验表；
4. 当前实际存在的源码 ZIP、TAR、Wheel 或解压目录。

每个文件必须记录：

```text
artifact_id
expected_filename
observed_path
present_or_missing
byte_size
sha256
expected_sha256
hash_match
archive_integrity
manifest_member_count
extraction_result
evidence_role
```

必须区分：

```text
PRESENT_AND_VERIFIED
PRESENT_BUT_HASH_MISMATCH
REPORT_ONLY_SOURCE_MISSING
SOURCE_AVAILABLE_NOT_YET_AUDITED
```

专家报告声称存在但本机没有的源码，必须标记 `REPORT_ONLY_SOURCE_MISSING`。不得根据交付报告、代码摘录或审查结论反推并声称完整源码已经验证。源码缺失是“专家独立仓库逐行审查”完成门禁的硬阻断项，但不妨碍先审计现有报告与复现证据。

## 5. Work Package B: Three-Way Code Evidence Comparison

对以下三类证据逐项比较：

```text
E1 = 专家 v1.0.0 实际源码；若缺失则明确不可验证
E2 = 专家独立 NO-GO 主张及其最小复现证据
E3 = C:\GitHub\YOLO-CV 当前 v3 的实际代码和测试
```

必须覆盖：

1. K-fold OOF 数据语义、fold assignment 和 held-out 身份证明；
2. train、OOF、val_target、val_model/study、val_cal、val_op、test 的唯一角色与泄漏边界；
3. 固定 epoch/checkpoint 选择和 `best.pt` 禁用规则；
4. `TN_at_FN95` 与 `FN_at_TN68253` 的两套独立阈值；
5. replay budget 的 denominator、per-epoch slots、unique IDs、repeat count、累计实际曝光和 optimizer steps；
6. Q/R/A/D 的定义、数据依赖、时间滞后和缺失输入时的 fail-closed 行为；
7. Treatment、global random、method-matched random、current-loss 和 no-replay 的公平性；
8. 实际 optimizer-visible exposure，而非仅计划 exposure；
9. checkpoint、RNG、OOM、kill、磁盘不足、原子写和 resume 一致性；
10. 单 job 入口、assignment generation、supersession、canonical completion 和 closeout；
11. source tree、数据 manifests、预测 sample IDs 和关键产物的身份闭环；
12. resource telemetry 是否采集真实非零测量值。

每个问题必须产出一行证据记录：

```text
issue_id
scientific_or_engineering_claim
expert_source_file_and_line
expert_reproduction_command
expert_reproduction_result
v3_source_file_and_line
v3_test_or_reproduction_command
v3_observed_result
status
remaining_risk
required_fix_or_no_change
```

`status` 只能使用：

```text
CONFIRMED_PRESENT
CONFIRMED_ABSENT
PARTIALLY_MITIGATED
CONTRADICTED_BY_EVIDENCE
NOT_TESTABLE_SOURCE_MISSING
NOT_APPLICABLE
```

所有“已修复”结论必须同时有当前行级源码、失败优先测试和实际通过结果。只引用专家 Markdown 不算复现。

## 6. Work Package C: Exact Nested Literature Sets

三个计数集合必须严格嵌套：

```text
DEEP_100 subset of SCREENED_300 subset of BROAD_500
|BROAD_500| = 500 unique works
|SCREENED_300| = 300 unique works
|DEEP_100| = 100 unique works
```

同一论文的 arXiv、workshop、conference 和 journal 扩展版本按研究身份去重，只计一个 canonical work；版本关系单独登记。旧 155/55/33 账本只能作为候选来源，逐篇按 v2 门禁重新核实后才可计数。

### 6.1 Search And Relevance Gate

必须保存数据库、查询式、查询日期、结果页范围、候选数和排除原因。技术论文优先使用官方 proceedings、期刊页面、OpenReview、PMLR、CVF、NeurIPS、JMLR 或作者正式预印本。

每篇计数论文必须直接回答至少一个问题：

```text
RQ1 哪些训练动态区分可学、已学、慢学、不可学和噪声样本？
RQ2 哪些 residual/reducible 信号比当前 loss 或 confidence 多提供信息？
RQ3 样本梯度的方向、影响或目标对齐怎样关联验证目标？
RQ4 集合覆盖、冗余和来源约束为何改变有限预算选择效果？
RQ5 replay 的时机、频率、重复强度和累计曝光怎样影响收益或伤害？
RQ6 哪些方法在预算、步骤和 seed 匹配的随机对照下成立或失败？
RQ7 哪些局部 ROC、Neyman-Pearson、pAUC 或高召回目标可定义 FN95 方向？
RQ8 seed、初始化、数据顺序和训练阶段怎样使样本价值发生条件性反转？
```

只有关键词相似、任务名相似或泛泛讨论“hard samples”的论文不得计入。

### 6.2 BROAD_500 Definition

每篇必须实际读取原始标题、摘要、问题定义、方法概览和结论，并有独立 `Pxxxx.md`。至少记录：

- canonical identity、作者、年份、venue、DOI/OpenReview/arXiv 和原始 URL；
- 实际阅读范围和日期；
- 自己写的中文摘要与批判性小综述；
- 对应 RQ、直接相关链、支持/反驳/混合结论；
- 可迁移机制和明确不能推出的结论；
- 全字段 schema。尚未进入筛读层的细节字段只能写枚举值 `NOT_ASSESSED_AT_BROAD_LEVEL`，不得编造。

只看标题、搜索摘要片段、二手综述或他人笔记不计入 500。

### 6.3 SCREENED_300 Definition

必须来自 BROAD_500，并阅读方法、实验设置、主要结果、消融和局限。每篇必须补充：

- 核心公式或算法步骤及变量定义；
- 选择发生的时点、更新频率和状态依赖；
- budget、unique samples、repeat/exposure、计算量和 optimizer steps；
- random baseline 的候选池、预算、步骤、时机和 seed 是否匹配；
- 数据集、模型、seed/repetition 数、checkpoint 规则；
- 至少一项带表/图/章节定位的精确结果；
- 消融、负结果、失败条件、局限和 Stage1 迁移风险；
- `REPLICATION`、`INSPIRED_ADAPTATION`、`MECHANISM_ONLY` 或 `NOT_TRANSFERABLE`。

论文未报告某字段时写 `NOT_REPORTED_BY_PAPER` 并给出已检查章节；这不是缺陷掩盖，也不得改写成肯定结论。拿不到方法和实验正文的论文不能进入 300。

### 6.4 DEEP_100 Definition

必须来自 SCREENED_300，取得并逐节阅读可核验全文。每篇必须补充：

- 本地 PDF/官方全文快照路径、字节数和 SHA-256；
- 阅读章节、页码和至少三个可回到原文的 evidence anchors；
- 关键公式的假设、变量、推导关系和适用边界；
- 算法、复杂度、随机性、训练阶段和数据依赖；
- 数据角色、leakage/oracle/winner's-curse 风险；
- 公平预算、随机对照、checkpoint、seed 级差异和最差情形；
- 关键表格、消融、负结果与作者局限的准确定位；
- Stage1 所需输入字段、接口、计算成本和代码映射；
- 一段独立反驳性复核：该论文为什么可能不适用于 Stage1。

没有全文、没有哈希、没有页级定位、只复述作者摘要或无法解释关键方法的论文不能进入 100。

## 7. One Paper, One Auditable Record

每篇计数论文必须有一个且只有一个主笔记：

```text
02_literature/review_500_300_100_v2/papers/P0001.md
...
02_literature/review_500_300_100_v2/papers/P0500.md
```

同一笔记随阅读层级升级，不复制成三套漂移文件。500 篇都必须有完整 schema；允许的缺失状态只有：

```text
NOT_ASSESSED_AT_BROAD_LEVEL
NOT_REPORTED_BY_PAPER
NOT_APPLICABLE_WITH_REASON
SOURCE_UNAVAILABLE_EXCLUDED
```

`TODO`、`TBD`、`unknown`、`待补`、`同上`、空字段和模板复述均为验收失败。

精确数字必须附表号、图号、页码或章节锚点。公式、算法、结果和限制必须与该论文身份一致。不得把综述中的二手描述冒充原论文证据，也不得把自动生成的模板文本冒充阅读记录。

## 8. Literature Validation And Anti-Fabrication

自动验证器必须 fail-closed，并至少检查：

1. 500/300/100 精确计数、严格嵌套和稳定 ID；
2. DOI、标题、作者、年份、版本关系和 URL 去重；
3. 500 个主 Markdown 与账本一一对应；
4. 各阅读层必填字段、合法枚举和来源证据；
5. 300 篇的方法/实验/budget/random/seed/result/limitation 字段；
6. 100 篇全文存在、可读、哈希匹配、页级 anchors 和代码迁移字段；
7. 大段跨论文重复、模板占位、标题与摘要错配和可疑统一措辞；
8. 精确数字无原文定位、abstract-only 冒充 full-text、二手来源冒充原始来源；
9. 每篇直接对应至少一个 RQ，且 Stage1 迁移链不是关键词拼接；
10. 以固定随机种子抽检不少于 500 的 10%、300 的 15%、100 的 20%；
11. 对机制最关键的 30 篇进行时间分离的第二遍全文复核，不伪称独立第二审稿人；
12. 任何失败均输出具体 paper ID 和字段，不得生成总体验收 PASS。

字数和文件数量不能单独证明阅读。最终验收依据是可追溯原文证据、论文特异的方法与实验细节、负结果和迁移边界。

## 9. Work Package D: Evidence Synthesis

完成代码和文献门禁后，形成：

1. 不使用任意权重的 Q/R/A/D 科学合同；
2. 机制优先级，分别标注 `SUPPORTED`、`CONTRADICTED`、`MIXED`、`UNKNOWN_IN_STAGE1`；
3. 最小可证伪实验矩阵，逐层比较 Q、R、A、D 的边际作用；
4. 严格配对的 R1 global random、R2 method-matched random、current-loss 和 no-replay；
5. 明确的 replay denominator、unique count、per-epoch slots、repeat histogram、累计实际曝光和 optimizer steps；
6. 独立 seed 的筛选、确认、停止、失败和多重比较规则；
7. 主指标 `FN=0..95` raw safety-frontier normalized AUC；
8. 次指标 `TN_at_FN95`、`FN_at_TN68253`、双端恶化率、seed 方向胜率和最差 seed；
9. 不使用 test oracle 选方法、checkpoint、阈值或超参数；
10. 下一版仓库的模块、schema、CLI、测试、产物和迁移顺序。

文献只能提供机制先验，不能替代 Stage1 的随机化干预。最终报告必须明确区分：

```text
paper evidence
expert claim
code evidence
synthetic reproduction
real Stage1 observational evidence
future causal intervention required
```

## 10. Required Deliverables

### 10.1 Expert And Code Audit

```text
01_field_audit/expert_v1_inventory.csv
01_field_audit/expert_v1_hash_validation.json
01_field_audit/expert_archive_member_manifest.csv
01_field_audit/expert_claim_ledger.csv
01_field_audit/expert_review_reproductions/
01_field_audit/expert_vs_v3_evidence_matrix.csv
01_field_audit/EXPERT_V1_NO_GO_V3_COMPARISON.md
```

### 10.2 Literature

```text
02_literature/review_500_300_100_v2/papers/P0001.md ... P0500.md
02_literature/review_500_300_100_v2/BROAD_500_INDEX.md
02_literature/review_500_300_100_v2/SCREENED_300_INDEX.md
02_literature/review_500_300_100_v2/DEEP_100_INDEX.md
02_literature/review_500_300_100_v2/SEARCH_QUERY_LEDGER.csv
02_literature/review_500_300_100_v2/READING_LEDGER.csv
02_literature/review_500_300_100_v2/DEDUPLICATION_LEDGER.csv
02_literature/review_500_300_100_v2/SCREENING_EXCLUSIONS.csv
02_literature/review_500_300_100_v2/EVIDENCE_MATRIX.csv
02_literature/review_500_300_100_v2/NEGATIVE_RESULTS_AND_FAILURES.md
02_literature/review_500_300_100_v2/RANDOM_BASELINE_EVIDENCE.md
02_literature/review_500_300_100_v2/KEY_30_SECOND_PASS.md
02_literature/review_500_300_100_v2/RANDOM_AUDIT_REPORT.md
02_literature/review_500_300_100_v2/SYSTEMATIC_LITERATURE_REVIEW.md
02_literature/review_500_300_100_v2/EXECUTIVE_SYNTHESIS.md
02_literature/review_500_300_100_v2/validation/PAPER_NOTE_VALIDATION.json
```

### 10.3 Final Decision Package

```text
03_preregistration_v3/SCIENTIFIC_CONTRACT.md
03_preregistration_v3/MINIMAL_FALSIFIABLE_MATRIX.csv
03_preregistration_v3/DATA_COLLECTION_SCHEMA.md
03_preregistration_v3/STATISTICAL_DECISION_RULES.md
03_preregistration_v3/REPOSITORY_CHANGE_SPEC.md
08_reports/STAGE1_BUDGETED_REPLAY_FINAL_RESEARCH_DECISION.md
08_reports/STAGE1_BUDGETED_REPLAY_FINAL_RESEARCH_DECISION.html
08_reports/COMPLETION_AUDIT.json
```

Desktop 只放可读镜像和交付 manifest，不作为事实源。仓库内已登记实验目录是唯一 authoritative source。

## 11. Execution Order And Stop Gates

```text
Gate 0: owner explicitly restarts execution
Gate 1: expert artifact inventory and hash audit
Gate 2: expert P0/High reproductions and v3 comparison
Gate 3: literature schema and validator tests
Gate 4: BROAD_500 exact validation
Gate 5: SCREENED_300 exact validation
Gate 6: DEEP_100 exact validation
Gate 7: random audit and key-30 second pass
Gate 8: scientific synthesis and implementation specification
Gate 9: independent completion audit and Desktop mirror
```

后续 gate 不得掩盖前一 gate 的失败。可以并行检索和整理候选，但不得在来源和阅读证据不足时提前记入目标数量。

本目标期间不得：

- 启动正式训练或把 canary 计入科学结果；
- 生成正式 engineering gate、pilot release 或正式 assignments；
- 打开、比较或基于 blind holdout/test 选择方法；
- 把 mock、synthetic 或最小复现结果写成 Stage1 方法有效性；
- 因 2026-09-10 算力期限而降低阅读、代码证据或验收标准；
- 在缺失专家源码时宣称专家独立仓库已经通过逐行审查。

## 12. Exact Completion Rule

只有 `08_reports/COMPLETION_AUDIT.json` 对下列每项给出可定位证据并全部 `PASS`，才允许报告目标完成：

```text
expert_expected_artifacts_accounted_for
expert_present_artifacts_hash_verified
expert_source_level_audit_completed
expert_p0_high_claims_reproduced_or_explicitly_not_testable
expert_vs_v3_line_evidence_complete
broad_500_exact_and_relevant
screened_300_exact_and_nested
deep_100_exact_nested_and_full_text_verified
canonical_work_deduplication_passed
all_500_per_paper_notes_passed
all_300_method_experiment_evidence_passed
all_100_hash_page_anchor_evidence_passed
random_audit_passed
key_30_second_pass_passed
systematic_review_passed
scientific_contract_has_no_arbitrary_weighted_score
falsifiable_matrix_and_statistics_passed
repository_change_spec_implementation_ready
desktop_mirror_matches_repository_manifest
formal_training_started_is_false
engineering_gate_generated_is_false
blind_holdout_opened_is_false
```

接近目标数量、引用列表很长、自动生成了 500 个文件、专家说已经测试、源码看起来合理、时间紧张或总结符合预期，都不能替代上述证据。

## 13. Copyable Goal Prompt

> 完成 Stage1 有限预算动态回流的证据驱动研究定案，但不得启动正式训练、engineering gate、pilot release 或 blind holdout。先逐项盘点并按 SHA-256 校验 `C:\Users\28898\Downloads` 中专家两轮交付；报告声称存在但实际缺失的源码必须标记 `REPORT_ONLY_SOURCE_MISSING`，不得根据报告猜测源码。随后将专家实际源码、NO-GO 复现证据和 `C:\GitHub\YOLO-CV` 当前 v3 在 OOF 语义、数据角色与泄漏、checkpoint、双阈值、预算、实际曝光、Q/R/A/D、随机对照、恢复、多机和身份闭环方面逐项对比，每项必须有文件/行号、复现命令和观测结果。建立去重且严格嵌套的 `100 subset 300 subset 500` 文献集：500 篇均需原始来源核对、独立中文摘要、批判性小综述和 Stage1 直接相关链；300 篇进一步核对方法、实验、公式、预算、随机基线、seed、精确结果、消融、负结果和局限；100 篇必须取得全文、记录本地 SHA-256、页级证据锚点、完整方法边界和代码迁移。每篇仅一个 `Pxxxx.md`，不得用无关论文、标题扫描、二手摘要、模板复述或占位字段凑数。自动验证精确数量、嵌套、去重、相关性、来源、哈希、逐篇笔记、随机抽检和关键 30 篇二次复核。最后基于代码与论文证据形成不使用任意加权总分的 Q/R/A/D 科学合同、机制优先级、最小可证伪矩阵、采集字段、统计规则和可直接施工的仓库改造规格；明确区分论文证据、专家主张、代码证据、synthetic 复现、Stage1 观察证据和仍需未来随机干预的未知项。所有事实源写入已登记实验目录，Desktop 只生成哈希一致的可读镜像。只有完成审计逐项 PASS，且另一名工程师无需本对话即可实施，目标才算完成。
