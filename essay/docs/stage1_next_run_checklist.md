# Stage-1 下一轮运行清单

## 先跑哪两个实验
1. `l + HN`
2. `s + HN`

这两个实验优先级最高，因为它们直接回答：
- 第一阶段主模型 `l` 在 HN 回流后能否继续推高高召回锚点过滤能力；
- 第二对照模型 `s` 在相同 HN 口径下是否仍值得保留。

## 最高优先级实验
- **最高优先级：`l + HN`**
- 原因：`l` 是 unified calibration 后的主模型，且是唯一在 calibration 后同时推高 `Spec@R99.5`、`Spec@R99.0` 和 `Prec@R99.0` 的模型。

## 运行前必须检查
1. 六类 sweep 的 `main.py` 已经切回你需要的版本，不要误用。
2. 训练机数据集路径存在：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/datasets/sewerml_gate2_train7200`
3. 基线权重存在：
   - `.../yolo11l_gate2_train7200/weights/best.pt`
   - `.../yolo11s_gate2_train7200/weights/best.pt`
4. 输出目录可写：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/research/materials/stage1_hn`
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/runs/stage1_gate_hn`

## 直接运行命令

### 1. 用主模型 `l` 给训练侧 normal 池重新打分
```powershell
uv run python .\scripts\stage1_score_train_normals.py --weights C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11l_gate2_train7200\weights\best.pt --data-root C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --output-dir C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11l_gate2_train7200 --device 0 --imgsz 640 --batch 16 --top-k 200
```

### 2. 构建 `l` 的 HN 数据集
```powershell
uv run python .\scripts\stage1_build_hn_dataset.py --source-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --scores-csv C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11l_gate2_train7200\top_false_positive_normals.csv --output-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate2_l_hn_top200_r2 --top-k 200 --repeat 2 --link-mode hardlink
```

### 3. 先跑最高优先级实验 `l + HN`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn.json
```

### 4. 用第二模型 `s` 给训练侧 normal 池重新打分
```powershell
uv run python .\scripts\stage1_score_train_normals.py --weights C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11s_gate2_train7200\weights\best.pt --data-root C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --output-dir C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11s_gate2_train7200 --device 0 --imgsz 640 --batch 16 --top-k 200
```

### 5. 构建 `s` 的 HN 数据集
```powershell
uv run python .\scripts\stage1_build_hn_dataset.py --source-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --scores-csv C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11s_gate2_train7200\top_false_positive_normals.csv --output-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate2_s_hn_top200_r2 --top-k 200 --repeat 2 --link-mode hardlink
```

### 6. 跑 `s + HN`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_s_hn.json
```

### 7. 如果 `l + HN` 明显有效，再跑 `l + HN + Weighted BCE`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn_wbce.json
```

### 8. 如果 `l + HN + Weighted BCE` 仍值得继续，再跑 `l + HN + Focal`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn_focal.json
```

## 跑完后优先看哪四个指标
1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0`

## 如何判断要不要继续
- 如果 `l + HN` 没有推高 `Spec@R99.5` 和 `Spec@R99.0`，先不要继续更重的 loss 变体。
- 如果 `l + HN` 推高了两项高召回锚点特异度，再看 `Prec@R99.0` 与 `PTR@R99.0` 是否仍可接受。
- 如果 `s + HN` 仍明显落后于 `l + HN`，后续 loss 变体优先继续压在 `l` 上，不必给 `s` 继续扩实验。
- 如果 `l + HN + Weighted BCE` 相比 `l + HN` 没能继续改善 `Spec@R99.5 / Spec@R99.0`，则不建议继续上更重的 Focal 版本。
