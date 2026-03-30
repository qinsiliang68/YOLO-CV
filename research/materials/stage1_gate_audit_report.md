# Stage-1 Gate 数值审计报告

## 审计范围
- 论文主文件：`essay/docs/essay.tex`
- 重点章节：第 4 章 `4.3/4.4/4.5`，第 6 章 `6.2/6.3`
- 原始材料来源：
  - `research/materials/yolo11*_gate2_train7200/*`
  - `research/materials/stage1_gate_calibration_all_models.csv`
  - `research/materials/run_master.csv`
  - 旧六类 source baseline 的 `val_summary.json` 历史版本

## 审计结论
- 第一阶段 direct binary gate baseline 数值已按 raw materials 统一核对。
- 第一阶段 unified calibration 五模型结果已按统一协议重算，并写回文稿。
- 第一阶段主模型与第二模型已经按统一 calibration 后的指标重新确定。
- 旧的口径错误已修正，不再保留“`yolo11m-cls` 是默认阈值最强 gate 模型”的错误说法。

## 已修正的问题
1. 默认阈值 leader 口径错误
   - 原问题：正文曾把 `yolo11m-cls` 写成默认阈值下最强 gate 模型。
   - 原始依据：`yolo11s_gate2_train7200/val_summary.json`
   - 正确事实：`yolo11s-cls` 才是默认阈值 leader，`accuracy=0.934722`，`macro_f1=0.876214`。
   - 修正动作：第 4 章、第 6 章相关 prose 已统一改为 `s` 默认阈值最强、`l` 高召回锚点最强、`m` 为 AUPRC 参考模型。

2. 第 6 章六类 source baseline 表存在占位符
   - 原问题：`表 6.2` 仍保留 `TODO`，与 raw materials 不一致。
   - 修正动作：已替换为完整五模型 source baseline 数值。
   - 当前正式口径：六类 source 分类视图下 `yolo11l-cls` 最强，`accuracy=0.7139`，`macro_f1=0.7151`，`AUPRC=0.9879`。

3. 第 6 章 calibration 只写了 `m/l`
   - 原问题：文稿只展示 `m/l` 两组 calibration，对五模型统一比较不完整。
   - 修正动作：已替换为五模型统一 Temperature Scaling 结果表，并改用五模型汇总图。

## 原本正确、保持不变的内容
- 二分类 gate baseline 表中的五模型数值与 raw materials 一致。
- 六类间接 gate 默认阈值过滤统计表与旧 source raw materials 一致。
- `yolo11l-cls` 在高召回锚点上的 leader 结论保持不变。
- `yolo11m-cls` 作为 AUPRC 参考模型的结论保持不变。

## 当前第一阶段正式口径
- 六类 source 分类 leader：`yolo11l-cls`
- 默认阈值 gate leader：`yolo11s-cls`
- 高召回锚点 gate leader：`yolo11l-cls`
- AUPRC 参考模型：`yolo11m-cls`
- unified calibration 后主模型：`yolo11l-cls`
- unified calibration 后第二对照模型：`yolo11s-cls`

## 备注
- 新一轮统一超参数六类 source 重跑已收到 `n/s/m/l`，但 `x` 尚未齐备，因此本次 stage-1 正式 source 排名仍以完整五模型旧 raw materials 为准。
