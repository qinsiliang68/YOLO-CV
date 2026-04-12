# 滑窗定峰实验部署说明

## 我们在干什么

我们正在做一件事:**精确定位高价值训练样本的最优区间**。

### 背景

之前的实验已经完成了三步:
1. **容量扫描**(10 run): 从 5 个 YOLO11 模型中选出 yolo11m 作为主力
2. **HN 全量扫描**(44 run): 发现回流增益是非单调的 — 不是越多越好,存在最优比例点(hn14 = 151 张)
3. **分位数分层消融**(16 run): 把 250 个候选样本按 R/D/C 三个信号分成 5 桶,发现了 **Goldilocks 效应** — 信号值最极端的样本(Q1)反而不如中间层(Q2-Q4)有价值

### 现在要做什么

第三步的 5 桶只是粗定位,告诉我们"大概在 Q2 附近最好"。现在需要**精确找到峰值中心和宽度**。

方法: 用滑动窗口扫描(窗口 50 样本, 步长 10)在 250 个候选样本上滑过,每个窗口位置都独立训练 + 评估,最终画出一条连续的**信号值 → 门控性能**响应曲线。

### 四个信号是什么

| 信号 | 英文 | 一句话 | 镜像对 |
|------|------|--------|--------|
| R | Risk (边界性) | 样本离门控决策边界多远 | 空间维度 |
| D | Density (抱团性) | 样本在特征空间有没有相似邻居 | 空间维度 |
| T | Trajectory (训练动力学) | 样本跨 epoch 是否持续帮助训练 | 动态维度 |
| C | Consistency (扰动稳定性) | 同一张图扰动后预测跳不跳 | 动态维度 |

R 和 D 是空间维度的一对正交镜像,T 和 C 是动态维度的一对正交镜像。

## 部署方式

### 前置条件

1. 仓库代码已 `git pull` 到最新(分支 `push-info-sampling-lite`)
2. `YOLOv11/datasets/sewerml_gate2_train7200/` 数据集在位
3. `research/materials/stage1_formal/gate_bucket_pilot/score_inputs/candidate_pool_master.csv` 存在
4. yolo11m hn00 best checkpoint (epoch_078.pt) 可访问
5. Python 环境有 ultralytics + torch + numpy

### 两台机器分别执行

**机器 A** — 跑空间维度 R + D (42 run, ~4.5 天):

```powershell
cd C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV
git pull origin push-info-sampling-lite
uv run main_sw_machineA.py
```

**机器 B** — 跑动态维度 T + C (42 run, ~4.5 天):

```powershell
cd C:\path\to\YOLO-CV
git pull origin push-info-sampling-lite
uv run main_sw_machineB.py
```

### 注意事项

1. **T 信号依赖 per_epoch_gate 数据**: 需要 `research/materials/stage1_formal/gate_hn_m_sweep/hn00/per_epoch_gate/` 目录下有逐 epoch 的 `val_op_predictions_calibrated.csv`。如果不存在,T 信号会自动退化为 R 的代理(仍然会跑,但结果不是真正的训练动力学信号)。

2. **已完成的实验会自动跳过**: 检测到 `last.pt` 和 `best_epoch_manifest.json` 就跳过,不重复训练。

3. **中途中断可以恢复**: 直接重新运行同一命令,已完成的 run 会跳过,从断点继续。

4. **如果只想跑单个信号**: 
```powershell
uv run main_sw_R.py   # 只跑 R
uv run main_sw_D.py   # 只跑 D
uv run main_sw_T.py   # 只跑 T
uv run main_sw_C.py   # 只跑 C
```

## 实验列表 (84 run)

每个信号 21 个滑动窗口:

| 窗口 | Rank 范围 | 中心位置 |
|------|----------|---------|
| w00 | 1-50 | 10% |
| w01 | 11-60 | 14% |
| w02 | 21-70 | 18% |
| w03 | 31-80 | 22% |
| w04 | 41-90 | 26% |
| w05 | 51-100 | 30% |
| w06 | 61-110 | 34% |
| w07 | 71-120 | 38% |
| w08 | 81-130 | 42% |
| w09 | 91-140 | 46% |
| w10 | 101-150 | 50% |
| w11 | 111-160 | 54% |
| w12 | 121-170 | 58% |
| w13 | 131-180 | 62% |
| w14 | 141-190 | 66% |
| w15 | 151-200 | 70% |
| w16 | 161-210 | 74% |
| w17 | 171-220 | 78% |
| w18 | 181-230 | 82% |
| w19 | 191-240 | 86% |
| w20 | 201-250 | 90% |

## 输出

- 训练权重: `YOLOv11/runs/stage1_sliding_window/sw_{signal}_w{XX}_r{start}_{end}/`
- 评估结果: `research/materials/stage1_formal/gate_sliding_window/sw_{signal}_w{XX}_r{start}_{end}/`
- 汇总 CSV: `research/results/stage1_formal/gate_sliding_window/sliding_window_summary.csv`

## 结果怎么用

84 run 跑完后,每个信号得到一条 21 点的响应曲线。用二次或三次多项式拟合,可以得到:
- **峰值中心 μ**: 最优样本区间的中心位置
- **峰值宽度 w**: Goldilocks zone 的范围
- **是否有次峰**: 是否存在第二个高价值区间

四个信号的 μ 和 w 确定后,就可以组合成最终的**非单调价值评分函数**:
$$V_i = U_R(R_i) \cdot U_D(D_i) \cdot U_T(T_i) \cdot U_C(C_i)$$
其中每个 $U$ 是以 μ 为中心、w 为宽度的高斯窗(中间高两端低)。
