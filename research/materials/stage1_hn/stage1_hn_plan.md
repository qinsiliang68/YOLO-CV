# Stage-1 Hard Negative Plan

## 原则
- 本目录下的 `top_false_positive_normals.csv` 与 `hardest_normal_gallery/` 来自现有验证侧误报 normal，仅用于展示、分析与误报模式归纳。
- 验证侧误报样本不得直接回流训练，避免污染 val-op。
- 真正可回流的 hard negative 必须来自训练侧 normal 池或额外 normal 池；后续统一通过 `stage1_score_train_normals.py` 在 `train/Normal` 上重新打分生成。

## 当前已准备材料
- `yolo11l_gate2_train7200`：导出验证侧误报 normal 31 张，训练侧 normal 池 1080 张。
- `yolo11s_gate2_train7200`：导出验证侧误报 normal 31 张，训练侧 normal 池 1080 张。

## 回流建议
- 可回流：后续在 `train/Normal` 中重新打分后选出的高置信 normal 误报样本。
- 仅展示或分析：当前 `fp_normal.csv` 导出的验证侧 hardest normal 画廊。
- 若本地工作区缺少原始图像，当前 gallery 可能为空；不影响数值筛选与训练机上的正式构建。

## 后续执行顺序
1. 用确定的主模型和第二模型在 `train/Normal` 上重新打分。
2. 按分数排序选择 top-k 训练侧 hard negatives。
3. 通过 `stage1_build_hn_dataset.py` 生成带 HN 重复样本的新数据集。
4. 再启动 HN、Weighted BCE 和 Focal 版本训练。
