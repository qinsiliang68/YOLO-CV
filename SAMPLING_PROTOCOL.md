# Sewer-ML Gate · 采样准则 v1

> **配套文件**:
> - `LEAKAGE_AUDIT.md` — 反向清单(不能做什么)
> - `SAMPLING_PROTOCOL.md` — **本文件**,正向协议(必须怎么做)
> - `essay/docs/essay3.tex` §3.5--§3.7 — reviewer 可见的科学承诺
>
> **冻结状态**: `pending` → 本文件 commit 后改为 `frozen`,hash 记入 `manifest.json`
> **版本**: v1 · 2026-04-18
> **seed**: `20260417`(全流程固定,不改)

---

## 0. 机器可读参数块

```yaml
# sampling_protocol_v1.yaml
version: v1
seed: 20260417

source:
  csv:        "YOLOv11/datasets/sewerml_annotations/SewerML_Train.csv"
  image_dir:  "C:/baidunetdiskdownload/sewerml_train_images/"
  # 四子集均从 SewerML_Train 单一池抽取;见 §1 关于为何不用 Val/Test 的说明

targets:
  train:   24000
  val_cal:  2400
  val_op:   5600
  test:    20000
  # 合计 52000 / 1.04M ≈ 5.0% 抽取率

class_mapping:
  negative:     { column: Defect, value: 0 }
  positive_any: [PF, DE, FS, RB, AF, OB]    # 任一为 1 即 y=1
  holdout_keep: [BE, RO, IN, FO]            # 不删帧,不进正类判定
  quality_drop: [OK, PH]                    # 整帧剔除

sampling_method: frame_level_random
stratification: none    # 确认不分层;量上稀释,不故意挑

output_dir: "research/materials/stage1_formal/manifests/v1/"
```

---

## 1. 数据源 · SewerML Train 单一池

**全部四子集 (train / val-cal / val-op / test) 均从 `SewerML_Train.csv` 抽取**。

不使用 SewerML Val / Test 的理由(已在 essay3 §3.6.5 披露):

1. **Val / Test 图像未本地化**: `C:\baidunetdiskdownload\sewerml_train_images\` 仅含 Train 的 1,040,129 张;Val 与 Test 需要从 Sewer-ML 官方源下载 `valid00/01.zip, test00/01.zip`,短期内不具备网络条件
2. **Test 标签持有于 Codalab**: `SewerML_Test.csv` 只有 `Filename` 一列,没有缺陷标签;官方评估要求提交到 Codalab leaderboard\cite{haurum2021sewerml},提交次数受限,不支持本文所需的细粒度指标分解
3. **Filename 不含 inspection_id**: Sewer-ML 的 filename 是全局顺序编号 (`00000001.png` 到 `01300201.png`),无 inspection / pipe metadata;即使使用 Val/Test,本文也无法从文件名恢复管段级分组

---

## 2. 类映射

| 角色 | 判定 | 说明 |
|---|---|---|
| Normal(负类 $y=0$) | `Defect == 0` 且 `OK == 0` 且 `PH == 0` | |
| 主任务正类($y=1$) | `PF + DE + FS + RB + AF + OB ≥ 1` | 任一为 1 即正 |
| Holdout 类 | BE, RO, IN, FO 之一 = 1 | 不删帧;若同时有主任务类则归正类,否则从池中排除 |
| 质量剔除 | `OK == 1` 或 `PH == 1` | 整帧剔除,见 essay3 §3.4 |

详见 essay3 §3.2 表 3.1。

---

## 3. Filename 约定

- Sewer-ML 使用全局顺序编号: `00000001.png` 至 `01300201.png`
- **不提取 inspection_id**(该信息不可得)
- 下游脚本以 filename 作为唯一 ID

---

## 4. 采样算法 · frame-level 随机

### 4.1 预处理

```python
import pandas as pd

df = pd.read_csv(SEWERML_TRAIN_CSV, encoding="utf-8-sig")

# 质量过滤
df = df[(df.OK == 0) & (df.PH == 0)]

# 二分类标签
MAIN = ["PF", "DE", "FS", "RB", "AF", "OB"]
HOLDOUT = ["BE", "RO", "IN", "FO"]
df["has_main"] = (df[MAIN].sum(axis=1) >= 1)
df["has_holdout_only"] = (df[HOLDOUT].sum(axis=1) >= 1) & (~df["has_main"]) & (df["Defect"] == 1)

# 排除"仅 holdout"帧
df = df[~df["has_holdout_only"]]

df["y"] = df["has_main"].astype(int)
df["image_id"] = df["Filename"].str.replace(".png", "", regex=False)
```

### 4.2 一次性无重复随机划分

```python
import numpy as np

SEED = 20260417
N_TRAIN, N_VAL_CAL, N_VAL_OP, N_TEST = 24000, 2400, 5600, 20000
N_TOTAL = N_TRAIN + N_VAL_CAL + N_VAL_OP + N_TEST  # 52000

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(df))[:N_TOTAL]

shuffled = df.iloc[indices].reset_index(drop=True)

train_df   = shuffled.iloc[: N_TRAIN]
val_cal_df = shuffled.iloc[N_TRAIN : N_TRAIN + N_VAL_CAL]
val_op_df  = shuffled.iloc[N_TRAIN + N_VAL_CAL : N_TRAIN + N_VAL_CAL + N_VAL_OP]
test_df    = shuffled.iloc[N_TRAIN + N_VAL_CAL + N_VAL_OP :]
```

**不做分层,不做 groupwise,不做 per-inspection cap**。
理由:
- 总抽取率 5.0%,同 inspection 跨子集分配的期望频次随整体抽取率线性下降,量上自然稀释
- 无 inspection_id,任何分组都是编造
- 主动分层(按 y / cjj 主类 / WaterLevel)会引入"故意挑"的选择偏差,反而扭曲样本分布

### 4.3 正负比例 sanity

预期 `y` 在 Sewer-ML Train 中约 45--55%(剔除 OK/PH 后)。frame-level 随机抽取应保留此比例,每子集漂移不超过 ±2pp。断言见 §7。

---

## 5. 产物落盘

输出目录:`research/materials/stage1_formal/manifests/v1/`

```
v1/
├── train_ids.csv           # 列: image_id, y, WaterLevel, PF,DE,FS,RB,AF,OB,BE,RO,IN,FO
├── val_cal_ids.csv
├── val_op_ids.csv
├── test_ids.csv
├── cooccurrence_matrix.csv # 行=6 主任务类, 列=4 holdout 类;按 split 分表
├── manifest.json
└── README.md               # test 目录警示 + 版本说明
```

**manifest.json 结构**:
```json
{
  "protocol_version": "v1",
  "protocol_sha256": "<hash of sampling_protocol_v1.yaml>",
  "seed": 20260417,
  "freeze_commit": "<git commit hash at freeze>",
  "generated_at": "2026-04-18T...",
  "source_pool": {
    "csv_path": "SewerML_Train.csv",
    "n_raw": 1040129,
    "n_after_quality": "<populated>",
    "n_after_holdout_exclude": "<populated>"
  },
  "splits": {
    "train":   { "n_frames": 24000, "sha256": "...", "y_ratio": "..." },
    "val_cal": { "n_frames": 2400,  "sha256": "...", "y_ratio": "..." },
    "val_op":  { "n_frames": 5600,  "sha256": "...", "y_ratio": "..." },
    "test":    { "n_frames": 20000, "sha256": "...", "y_ratio": "..." }
  }
}
```

---

## 6. 采样脚本必选断言(失败 → 拒绝写入产物)

```python
def assert_integrity(splits: dict):
    # 1. pairwise image_id disjoint
    for (a_name, a), (b_name, b) in itertools.combinations(splits.items(), 2):
        a_ids = set(a.image_id); b_ids = set(b.image_id)
        assert not (a_ids & b_ids), f"image_id leak between {a_name} and {b_name}"
    # 2. no duplicate image_id within split
    for name, df in splits.items():
        assert df.image_id.is_unique, f"{name}: duplicate image_ids"
    # 3. y distribution sanity
    for name, df in splits.items():
        y_ratio = df.y.mean()
        assert 0.30 <= y_ratio <= 0.70, f"{name}: y_ratio={y_ratio:.3f} out of plausible band"
    # 4. size match target
    expected = {"train": 24000, "val_cal": 2400, "val_op": 5600, "test": 20000}
    for name, df in splits.items():
        assert len(df) == expected[name], f"{name}: size={len(df)} != {expected[name]}"
    # 5. all image files physically present
    for name, df in splits.items():
        missing = [i for i in df.image_id.head(100) if not (IMG_DIR / f"{i}.png").exists()]
        assert not missing, f"{name}: {len(missing)}/100 sampled images missing on disk"
    # 6. seed recorded
    assert SEED == 20260417
```

---

## 7. Phase 0 · 冻结 checklist(commit 本 yaml 前最后一次核对)

- [ ] 路径正确:`SewerML_Train.csv` 在 `C:/GitHub/YOLO-CV/YOLOv11/datasets/sewerml_annotations/`
- [ ] 图像目录: `C:/baidunetdiskdownload/sewerml_train_images/` 含 1,040,129 张
- [ ] `seed = 20260417` 确认
- [ ] target 规模 (24k/2.4k/5.6k/20k) 与 essay3 §3.6 表 3.2 一致
- [ ] `holdout_keep` 与 `quality_drop` 清单与 essay3 §3.2 / §3.4 一致
- [ ] `output_dir` 已建空目录,没有前一版(`extended_*_split.csv` 等)残留混入
- [ ] 本文件 commit,记录 commit hash
- [ ] 同时 commit `LEAKAGE_AUDIT.md` v1 对应更新

---

## 8. 冻结后若需修改

**禁止**:
- 基于 val-op / test 结果反向改以上任一参数
- 直接覆盖 `v1/` 目录产物

**允许**:
- 发版 `sampling_protocol_v2.yaml`,输出至 `v2/` 新目录
- 论文 appendix 并列汇报 v1 / v2 两版结果
- commit message 显式说明改版动机(必须不基于 test)

---

## 附:与 essay3 的对应关系

| 本文件节 | essay3 节 |
|---|---|
| §1 数据源(单一 SewerML Train) | §3.6.5 帧间相关性与数据源约束下的独立性处理 |
| §2 类映射 | §3.2 CJJ 对齐 + 表 3.1 |
| §4 frame-level 随机采样 | §3.5 采样 · §3.6.2 val 二次划分 |
| §5.cooccurrence_matrix | §3.5.2 多标签保留(任务定义污染披露) |
| §8 冻结与版本化 | §3.7.5 协议冻结(防 validation overfitting) |

残留相关性由 `essay3.tex` §3.7.5 的 **filename 连续块 bootstrap**(块大小 = 50)作稳健性处理,作为 inspection-level bootstrap 的工程代理。
