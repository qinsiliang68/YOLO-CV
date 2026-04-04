# Stage-1 逐表对账清单

## 目的

这份清单用于统一论文 stage-1 各张表格的正式来源，避免同一概念在不同表中指向不同 CSV/JSON。

## 对账规则

1. 先区分“模型选型表”和“主线增强表”。
2. 同属 `0% HN` 的基线，如果来自不同重算批次，只允许在各自链路内部比较。
3. 第二模型补充表不参与主线 `G` 编号。

## 正式映射

### 六类 source 容量扫描

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-capacity-scan-cls6}`
  - 第 6 章第一阶段 baseline 收束段
- 正式来源：
  - `research/materials/run_master.csv`
- 正式口径：
  - 只使用统一重跑的 `yolo11*_cls6_train7200_uniform` 行
  - `yolo11x-cls` 是 accuracy leader
  - `yolo11n-cls` 是 AUROC / AUPRC leader
- 备注：
  - 旧版 `yolo11l_cls6_train7200` 结果只作历史留档，不再用于正文 leader 表述

### direct binary gate 五模型 baseline

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-capacity-scan-gate2}`
- 正式来源：
  - `research/materials/run_master.csv`
- 正式口径：
  - `yolo11s-cls` 是默认阈值 leader
  - `yolo11l-cls` 是高召回锚点 leader
  - `yolo11m-cls` 是 AUPRC 参考模型

### 五模型 calibration 结果表

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-calibration-results}`
  - 第 6 章 calibration 收束段
- 正式来源：
  - `research/materials/stage1_gate_calibration_all_models.csv`
- 用途：
  - 只用于五模型 calibration 选型与温度缩放分析
- 备注：
  - 其中的 `yolo11l-cls + calibration + 0% HN` 不能拿去与 `hn00` 做跨表绝对值逐项比较

### HN 回流比例扫描

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-hn-ratio-sweep}`
  - 附录 HN 全量表
- 正式来源：
  - `research/results/stage1_gate_hn_ratio_sweep/hn_ratio_sweep_summary.csv`
- 用途：
  - 作为 HN / PTSG / max-filter 主线的内部 baseline 链
- 备注：
  - `hn00` 是后续主线内部基线，不回头替代 calibration 五模型表中的 `0% HN`

### PTSG 主结果

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-ptsg-results}`
- 正式来源：
  - `research/materials/stage1_ptsg/yolo11l_gate2_hn02/ptsg_summary.csv`
  - 各变体目录下 `threshold_summary.json`
- 用途：
  - 比较 `P0~P4`
- 备注：
  - 与 HN 比例扫描表只作同轮内部相对比较，不作跨表绝对值逐项比较

### PTSG next-wave

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-ptsg-nextwave}`
- 正式来源：
  - `research/results/stage1_ptsg_nextwave/ptsg_nextwave_summary.csv`
- 用途：
  - 比较 `P2 / P5a / P5b / P6a / P6b`

### Max-filter suite

- 论文位置：
  - 第 4 章 / 第 6 章主线消融与工程增强表
  - 第 6 章图 `\ref{fig:stage1-maxfilter-suite-best}`
- 正式来源：
  - `research/results/stage1_gate_maxfilter_suite/stage1_maxfilter_suite_summary.csv`
  - `research/results/stage1_gate_maxfilter_suite/stage1_maxfilter_suite_summary.json`
- 当前正式口径：
  - 新的 stage-1 训练侧综合最优候选为 `HardMix`
  - 最优后处理变体为 `P0`

### 第二模型跨容量补充表

- 论文位置：
  - 第 4 章 `表\ref{tab:stage1-second-model-check}`
  - 第 6 章 `表\ref{tab:stage1-second-model-results}`
- 正式来源：
  - `research/results/stage1_gate_hn_ratio_sweep/second_model_summary.csv`（若后续保留）
  - 或当前正文中对应 `yolo11s-cls` 校准基线与 `hn02` 数值来源文件
- 用途：
  - 只证明 `hn02` 存在跨容量稳定收益
- 备注：
  - 不再使用 `G3/G4` 编号，避免与主线消融组混淆

## 当前最终收束口径

- six-class source：
  - `yolo11x-cls` 为 accuracy leader
  - `yolo11n-cls` 为 AUROC / AUPRC leader
- stage-1 mainline：
  - `yolo11l-cls + calibration + hn02 + HardMix`
- stage-1 best post-hoc：
  - `P0 = calibrated p_abnormal`
