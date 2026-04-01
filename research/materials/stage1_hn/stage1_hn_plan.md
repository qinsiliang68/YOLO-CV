# Stage-1 Hard Negative Plan

## 原则
- 本目录下的 `top_false_positive_normals.csv` 与 `hardest_normal_gallery/` 来自现有验证侧误报 `normal`，仅用于展示、分析与误报模式归纳。
- 验证侧误报样本不得直接回流训练，避免污染 `val-op`。
- 真正可回流的 hard negative 必须来自训练侧 `Normal` 池或额外 `Normal` 池；后续统一通过 `stage1_score_train_normals.py` 在 `train/Normal` 上重新打分生成。

## 当前已确定口径
- 主模型：`yolo11l-cls`
- 第二模型：`yolo11s-cls`
- HN 回流强度：`hn02 = 2%`
- 对应回流样本数：`22`

## 当前已准备材料
- `yolo11l_gate2_train7200`：验证侧误报 `normal` 样本与训练侧 `Normal` 打分脚本均已就绪。
- `yolo11s_gate2_train7200`：验证侧误报 `normal` 样本与训练侧 `Normal` 打分脚本均已就绪。
- `hn02` 的最佳性已经由比例扫描确认：
  - `Spec@R99.5 = 0.5119`
  - `Spec@R99.0 = 0.5476`
  - `Prec@R99.0 = 0.9165`
  - `PTR@R99.0 = 0.9028`

## 回流建议
- 可回流：后续在 `train/Normal` 中重新打分后选出的高置信 `Normal` 误报样本。
- 仅展示或分析：当前 `fp_normal.csv` 导出的验证侧 hardest normal 画廊。
- 若本地工作区缺少原始图像，当前 `gallery` 可能为空；这不影响数值筛选与训练机构建。

## 后续执行顺序
1. 用主模型 `l` 和第二模型 `s` 在 `train/Normal` 上重新打分。
2. 各自按分数排序截取 `top-22` 训练侧 hard negatives。
3. 通过 `stage1_build_hn_dataset.py` 生成 `hn02` 数据集视图。
4. 先跑 `l + HN 2%`，再跑 `s + HN 2%`。
5. 若 `l + HN 2%` 继续有效，再在 `l` 上追加 `Weighted BCE` 与 `Focal`。
