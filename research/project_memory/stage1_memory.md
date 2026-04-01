# Stage-1 Memory

## 1. 任务定义

stage-1 不是最终六类检测器，而是两阶段系统中的高召回 gate。

核心目标：

- 尽量不过漏异常
- 在固定高召回约束下尽可能拦截 `normal`

因此 stage-1 的核心指标不是默认阈值 `Accuracy`，而是：

- `Spec@R99.5`
- `Spec@R99.0`
- `Prec@R99.0`
- `PTR@R99.0`

## 2. 为什么采用 direct binary gate

第一阶段并非直接凭经验选定二分类任务和具体模型，而是通过：

- 五组六类 source baseline
- 五组 direct binary gate baseline

的系统比较，先验证任务重定义的必要性，再在统一评价口径下完成模型选型。

稳定结论：

- source 分类排序与 direct binary gate 排序并不一致
- 第一阶段最应优化的是高召回约束下的 `normal` 过滤能力
- 因此 stage-1 应以 direct binary gate 结果为主

## 3. 模型角色

- `yolo11l-cls`：主模型，高召回锚点最强
- `yolo11s-cls`：第二对照模型，默认阈值整体最强
- `yolo11m-cls`：AUPRC 参考模型

## 4. calibration 共识

Temperature Scaling 已统一做完。

固定协议：

- `val-cal = 30%`
- `val-op = 70%`
- `seed = 20260330`

它的作用是：

- 让分数更可信
- 让阈值选择更稳
- 让模型比较不再基于过度自信 raw score

## 5. HN 共识

### 5.1 HN 的意义

高置信误报 `normal` 是当前最有信息量的 hardest normal。

stage-1 中真正限制 `Spec@R99.5 / Spec@R99.0` 的，往往不是 easy normal，而是：

- 反光
- 水膜
- 污渍
- 接缝纹理
- 阴影
- 纹理伪异常

### 5.2 为什么选 2%

`0%~20%` 比例扫描已证明：

- `2%` 是当前最优回流强度
- 更大比例未继续带来稳定收益

因此后续主线固定为：

- `0%` 对照
- `2%` 主线

完整比例扫描继续保留在正文/附录中作为证据链。

## 6. 当前 stage-1 正式实验序列

1. `s + HN 2%`
2. `l + HN 2% + Weighted BCE`
3. `l + HN 2% + Focal`

如果后续还扩展：

- `BestLoss`
- `BestLoss + LightAttn`

它们应被归类为：

- 第一阶段关键消融与工程增强实验

而不应混写成同一种严格单因素消融。
## 7. PTSG Candidate Line

- Framework-level framing: `SNSG / selective safe-normal gate`
- Immediate implementation path: `PTSG`
- Current execution rule:
  - keep `G2 = yolo11l-cls + calibration + hn02`
  - do post-hoc evaluation first
  - do not retrain the backbone first

Fixed first-round comparison set:

- `P0`: calibrated `p_abnormal`
- `P1`: `p_abnormal + uncertainty`
- `P2`: `p_abnormal + trust`
- `P3`: `p_abnormal + trust + uncertainty`
- `P4`: `P3 + HN-aware normal bank`

The ranking rule does not change:

- `Spec@R99.5`
- `Spec@R99.0`
- `Prec@R99.0`
- `PTR@R99.0`
