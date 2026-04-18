# Sewer-ML Gate · Data Leakage Audit & Execution Checklist

> **作用域**: 二分类门控分类器的训练/评估流水线。任何执行 AI、协作者或未来的自己在跑采样前必须逐条打勾。
> **版本**: v1 · 2026-04-18(对应 SewerML Train 单一池 + frame-level random 方案)
> **配套**: `SAMPLING_PROTOCOL.md` + `essay/docs/essay3.tex` §3.5--§3.7
> **核心原则**: **泄漏分两档,不可混称** —— (A)直接毒化 test 数字的,必堵;(B)不毒化但扭曲 CI 与 claim 的,必须量化并交代。

---

## Phase 0 — Protocol Freeze(采样开跑前必须完成)

在抽第一张样本之前把以下全部定死,写入 `sampling_protocol_v1.yaml` 并 commit。commit hash 记入 `manifest.json`。

- [ ] CJJ 类映射(6 主任务类 + holdout 清单 + Normal 定义)
- [ ] OK/PH 等图像质量过滤规则
- [ ] 数据增强参数集(mosaic/HSV/flip/scale 等)
- [ ] 字典序 rank 规则(`Spec@R99.5 ≻ Spec@R99.0 ≻ Prec@R99.0 ≻ -PTR@R99.0`)
- [ ] τ 搜索算法(二分或扫描,步长固定)
- [ ] 三子集目标规模(train 24k / val-cal 2.4k / val-op 5.6k / test 20k)
- [ ] ImageNet 固定 mean/std(推荐;避免数据驱动预处理统计量的泄漏,见 L1-6)

**冻结后的铁律**: 任何基于 val-op **或** test 结果反向修改以上规则 = 把验证集用作 meta-train。要改,必须发版到 `v2` 并在论文显式说明。

---

## Phase 1 — 采样执行(一级坑)

### L1-1 单一数据源 = SewerML Train

四子集(train / val-cal / val-op / test)均从 `SewerML_Train.csv` 抽取。理由见 `SAMPLING_PROTOCOL.md` §1:

- SewerML Val / Test 图像未本地化
- SewerML Test 标签持有于 Codalab,本地不可用
- SewerML filename 不含 inspection\_id,即使用 Val / Test 也无法恢复管段级分组

### L1-2 frame-level 随机划分

一次性无重复随机采样,固定 seed `20260417`。**不做分层、不做 groupwise、不做 per-inspection cap**。

*过去版本曾讨论过的 inspection 级 groupwise split、stratification、K 帧数上限,在本版中全部放弃*,因其前提(inspection\_id 可得)不成立。不假装具有已知不存在的保证。

### L1-3 产物落盘:ID 清单 + SHA256 + protocol hash

```
sampling_output/v1/
├── train_ids.csv
├── val_cal_ids.csv
├── val_op_ids.csv
├── test_ids.csv
├── cooccurrence_matrix.csv
├── manifest.json          # 4 个清单的 SHA256 + protocol.yaml hash
└── README.md              # test 目录警示
```

下游训练/评估脚本启动时必须校验 `manifest.json` 里的 hash,不匹配拒绝跑。

### 采样脚本断言(详见 `SAMPLING_PROTOCOL.md` §6)

核心断言:
```python
# 1. pairwise image_id disjoint across splits
# 2. no duplicate image_id within each split
# 3. y_ratio in [0.30, 0.70] per split
# 4. size matches target exactly
# 5. sampled image files physically present on disk
# 6. seed == 20260417
```

---

## Phase 2 — 训练侧样本构造(一级坑 · 最易漏)

### L1-4 反向构造集合只能读 train

**凡是会反向影响训练分布的样本选择动作,输入池必须只来自 $\mathcal{D}_{\rm train}$。**

禁止以下动作从 val / test 构造:
- hard negative mining
- candidate pool / active learning pool
- pseudo-label 生成
- sample weights / class weights
- curriculum 难度打分
- noisy-label filter
- Goldilocks / RDTC 价值打分用于筛选训练样本

实施: `sample_selection.py` 顶部:
```python
assert input_pool.origin_split == "train", \
    f"reverse construction must read train only, got {input_pool.origin_split}"
```

### L1-5 oversample / reweighting 只在 train 内、split 完成之后

绝不对 val/test 做分布重整 —— 评估集必须保留自然分布,否则 Spec 失去工程意义。

---

## Phase 3 — 预处理 & 校准(一级坑)

### L1-6 数据驱动的预处理统计量只能在 train 上算

以下全部只读 train:
- mean / std
- PCA whitening / ZCA
- histogram matching reference
- quality score 阈值
- class frequency(用于 loss weighting)
- sample reweight 频次

**推荐直接用 ImageNet 固定 mean/std**:(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225),自动满足此条。

### L1-7 T 与 τ 必须每 epoch 独立拟合

禁止跨 epoch 复用。对 200 个 ckpt 各跑一遍 `val-cal → val-op` 流水线,代价约几十分钟,能接受。

---

## Phase 4 — 模型选择(二级坑:不毒化 test,但让 val-op 变 meta-train)

### L2-1 近似重复帧 & 同 inspection 残留相关 = $n_{\rm eff}$ 虚高,不是泄漏

Wilson CI 汇报时**明示为"独立性假设下的下界"**。如需稳健区间,跑 **filename 连续块 bootstrap**($B=2000$,块大小 = 50,Künsch 1989)作为 inspection-level bootstrap 的工程代理。见 essay3 §3.7.5。

### L2-2 多标签 collapse = 任务定义污染,不可忽略

- 采样产物附:"主任务类 × holdout 类" 共现矩阵
- 正文汇报 train / val / test 多标签帧占比
- 不主观删"视觉主导"的多标签帧(主观判定不可复现)
- 若 per-class recall 翻车,在 discussion 归因到此

### L2-3 val-op 只看字典序输出,不看指标曲线做设计迭代

- 跑完 200 epoch,按字典序选 θ*,落地 `best_epoch.json`
- 若结果不理想:可重跑同协议不同 seed;**不可**回头改 Phase 0 的任一规则
- 若规则必须改:进入 `protocol v2`,全部四子集**重抽**一次

---

## Phase 5 — Test 最终评估(一级坑 · 铁律)

### L1-8 Phase 0 冻结 ~ Phase 5 之间,test 零读取

实施:
- `test/` 目录 `README.md` 显眼警示
- 所有 dev 脚本顶部:`assert os.getenv("FINAL_EVAL") == "1"` 才允许读 test
- git hook / CI 阻止 `test_ids.csv` 被任何非最终评估脚本引用

### Phase 5 执行(仅一次)

1. 冻结 $(\theta^*, T^*, \tau^*)$
2. 在 test 上推理一次
3. 输出 Spec@R99.5 / Spec@R99.0 + Wilson CI
4. 可选:filename-连续块 bootstrap CI(见 L2-1)

### Phase 5 之后

- **不允许**基于 test 结果修改任何决策(模型、阈值、采样、协议)
- 真发现 bug 必须重读:论文 appendix 显式报告"test 被读 N 次,每次原因"

---

## Phase 6 — 稳健性加强(可选;submit 前加分)

### L3-1 filename 连续块 bootstrap

$B = 2000$,重采样单元 = 连续 50 个 filename 组成的块(不是 frame,也不是不可获得的 inspection)。并列汇报 Wilson CI 与 bootstrap CI。Block bootstrap 区间通常宽 1.5--3×,反映残留相关性下的真实不确定性。

### L3-2 小规模视觉 sanity check(可选)

从 $\mathcal{D}_{\rm test}$ 抽 20 对"filename 相邻"的帧(filename 差 ≤ 3),目视判断是否属同一段 CCTV 视频。该比例量化"filename 连续是否近似 inspection"这一工程假设的有效性,可附于论文附录。

### L3-3 Tail-safe guardrail

- 按 CJJ 主类报 per-class recall
- 计算 worst-class recall
- 防止 pooled recall = 99.5% 掩盖 PF / DE 等稀少类的漏检

---

## 附:一句话总结(贴给下游执行 AI)

> **泄漏分两档**:
> (A)用 val-test 反向构造 train 样本 / test 偷看 / 跨 epoch 复用 T·τ / 数据驱动预处理碰 val-test —— 直接毒化数字,**必堵**。
> (B)同 inspection 残留相关(因 SewerML 不提供 inspection\_id,本文无法消除)/ 多标签 collapse / validation overfitting —— 不毒化但扭曲 CI 与 claim,**必须量化并交代**。
> **绝不将两类混称"泄漏"**。

---

## 版本历史

- **v1 · 2026-04-18**: 初版。对应 essay3 §3.6.5(帧间相关性与数据源约束)+ §3.7.5(统计推断局限)。
    - 采样方案: SewerML Train 单一源 + frame-level random 划分
    - 放弃: inspection 级 groupwise split / WaterLevel stratification / per-inspection K 帧数上限(前提不成立)
    - 残留相关性处理: filename 连续块 bootstrap(块大小 = 50)
