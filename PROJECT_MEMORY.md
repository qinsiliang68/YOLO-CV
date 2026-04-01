# Project Memory

## 1. 项目定位

本项目面向地下排水管道 CCTV 巡检场景，目标不是单纯追求某个默认阈值下的总体分类或检测分数，而是构建一套面向工程部署的两阶段缺陷智能检测框架，形成：

`工程问题 -> 两阶段架构 -> 支撑链增强 -> 改进 detector -> 跨域迁移 -> 系统验证`

的完整闭环。

## 2. 当前主线

当前论文与代码主线已经收敛为：

`源域分类预训练 -> 第一阶段高召回 gate -> 弱监督候选区域/伪框增强 -> 第二阶段改进 YOLOv11 精检 -> 跨域迁移适应 -> 系统级验证`

其中：

- 第一阶段：高召回门控，尽量不过漏异常，最大化过滤正常背景。
- 第二阶段：六类结构性缺陷检测，完成最终类别判定与空间定位。
- 跨域迁移：通过源域预训练与目标域微调，缓解目标域标注不足与域差问题。

## 3. 创新点与支撑链

### 创新点 1

面向地下管网场景的改进 YOLOv11 检测方法，围绕三类定向优化展开：

- 主优化：注意力/特征建模模块
- 辅助优化 1：检测头改进
- 辅助优化 2：损失函数定向设计

### 创新点 2

源域预训练与目标域微调相结合的跨域迁移策略：

`源域分类预训练 -> 目标域分类微调 / gate 迁移 -> 目标域检测微调`

### 关键支撑链

以下内容是关键工程支撑链，不单独升级为独立核心创新点：

- 第一阶段高召回 gate
- HN 回流策略
- CAM / Grad-CAM 弱监督候选区域与伪框增强
- 附录级补充表格与全量实验材料

## 4. 第一阶段稳定结论

### 4.1 五模型 baseline 角色定位

- 六类 source 分类 leader：`yolo11l-cls`（旧完整口径）
- direct binary gate 默认阈值 leader：`yolo11s-cls`
- direct binary gate 高召回锚点 leader：`yolo11l-cls`
- direct binary gate AUPRC 参考模型：`yolo11m-cls`

重要约束：

- 不能把 `yolo11m-cls` 写成默认阈值 leader。
- prose 必须服从 raw materials，不允许“图里是 s 强，正文写成 m 强”。

### 4.2 calibration 结论

第一阶段五个二分类 gate 模型已完成统一 Temperature Scaling，比对口径为：

- `val-cal = 30%`
- `val-op = 70%`
- `seed = 20260330`
- 基于现有逐图 `p_abnormal` 反推二分类 logits

统一 calibration 的作用是：

- 改善分数刻度
- 让 operating point 比较更干净
- 让模型选型与阈值分析建立在统一校准后的分数体系上

calibration 改的是分数刻度，不是模型本体。

### 4.3 HN 结论

主模型 `yolo11l-cls` 上已完成 `0%~20%` HN 回流比例扫描，稳定结论是：

- `hn02`（2%）是当前最优回流强度
- `Spec@R99.5`：`0.4405 -> 0.5119`
- `Spec@R99.0`：`0.5357 -> 0.5476`
- `Prec@R99.0`：`0.9143 -> 0.9165`
- `PTR@R99.0`：保持不变

解释口径：

- 小比例高置信误报 `normal` 回流不会提升默认阈值下的总体分类分数
- 但会显著提升第一阶段 gate 在固定高召回约束下的 `normal` 过滤能力
- 当前最优回流强度为 `2%`

## 5. 第一阶段当前实验顺序

当前 stage-1 主线顺序为：

1. `yolo11l-cls + calibration + hn02`
2. `yolo11s-cls + hn02` 作为第二容量对照
3. `yolo11l-cls + hn02 + Weighted BCE`
4. `yolo11l-cls + hn02 + Focal`

`yolo11s-cls + hn02` 的定位不是重新给 `s` 找全局最优 HN 比例，而是验证：

> 在非主模型容量下，主模型选出的 `hn02` 回流方案是否仍具有一定稳定增益。

## 6. 统一入口与运行规则

### 6.1 统一入口

当前根目录统一入口为：

```powershell
uv run main.py
```

实际任务由：

- `YOLOv11/configs/runtime/main_entry.json`

控制。

当前默认任务是：

- `stage1_gate_s_hn`

也就是：

- 先给训练侧 `Normal` 打分
- 自动构建 `HN 2%` 数据集
- 再启动 `yolo11s-cls + HN 2%` 训练

### 6.2 特殊任务

如果要跑六类统一容量扫描，用：

```powershell
uv run main.py --task cls6_sweep --rerun
```

## 7. 仓库协作规则

当前仓库采用单一 `main` 分支协作：

- 本地工作机：
  - 改代码
  - 改论文
  - 推送 `main`
- 训练机：
  - 同步 `main`
  - 跑实验
  - 只推送 `research/materials/` 和 `research/results/`

### 强制规则

- 训练机不要改论文
- 训练机不要 force push
- 训练机不要改仓库结构
- 长训练只在训练机执行
- 本地机负责整理表格、图表、脚本和论文

## 8. 数据与删除规则

- `data/`、`.venv/`、`YOLOv11/runs/`、`YOLOv11/weights/`、`$out/` 不进 Git
- `research/materials/`、`research/results/` 保留关键实验材料
- `dataset_manifest.csv / split_train.csv / split_val.csv` 全量保留
- `raw_run_artifacts/*.jpg` 只保留少量代表图
- 重复的 `pip_freeze.txt / runtime_profile.json / model_profile.json` 只保留代表性实验

删除规则：

- 用户说“删除”，默认送 `_recycle_bin/`
- 不直接物理删除

## 9. AI 助手工作约束

所有 AI / 自动化助手在继续此项目时，必须遵守：

1. 只信 raw materials，不信旧 prose
2. 不把 `m` 写成默认阈值 leader
3. stage-1 结论必须围绕：
   - `s`：默认阈值 leader
   - `l`：高召回锚点 leader
   - `m`：AUPRC 参考模型
4. HN 主线固定为：
   - 主模型 `l`
   - 第二模型 `s`
   - HN 比例固定 `2%`
5. 默认对人类友好的入口应优先做成：
   - `uv run main.py`

## 10. 优先查看文件

任何接手本项目的人或 AI，优先看这几个文件：

- `PROJECT_MEMORY.md`
- `research/project_memory/stage1_memory.md`
- `research/project_memory/decision_log.md`
- `research/training_machine_runbook.md`
- `research/experiment_handoff_workflow.md`
- `essay/docs/stage1_next_run_checklist.md`
