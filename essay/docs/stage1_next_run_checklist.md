# Stage-1 下一轮运行清单

## 先跑哪两个实验
1. `l + HN 2%`
2. `s + HN 2%`

这两个实验优先级最高，因为它们直接回答两件事：
- 主模型 `l` 在最优 `hn02` 口径下能否继续稳定保持高召回锚点优势。
- 第二模型 `s` 在同一 `hn02` 口径下是否仍值得保留为后续对照模型。

## 最高优先级实验
- **最高优先级：`l + HN 2%`**
- 原因：`l` 是当前高召回锚点主模型，且 `hn02` 已经证明能够把 `Spec@R99.5` 从 `0.4286` 提升到 `0.5119`。

## 运行前必须检查
1. 根目录 `main.py` 已切回你当前需要的版本，不要误用六类 sweep 入口。
2. 二分类原始数据集存在：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/datasets/sewerml_gate2_train7200`
3. 基线权重存在：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/runs/cls_gate_source/yolo11l_gate2_train7200/weights/best.pt`
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/runs/cls_gate_source/yolo11s_gate2_train7200/weights/best.pt`
4. HN 2% 数据集目录可写：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/datasets/stage1_gate_hn_backflow`
5. 训练输出目录可写：
   - `C:/Users/ASUS/Desktop/YOLOv11/YOLO-CV/YOLOv11/runs/stage1_gate_hn`

## 直接运行命令

### 1. 用主模型 `l` 给训练侧 normal 池重新打分
```powershell
uv run python .\scripts\stage1_score_train_normals.py --weights C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11l_gate2_train7200\weights\best.pt --data-root C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --output-dir C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11l_gate2_train7200 --device cpu --imgsz 640 --batch 2 --top-k 22
```

### 2. 构建 `l` 的 HN 2% 数据集
```powershell
uv run python .\scripts\stage1_build_hn_dataset.py --source-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --scores-csv C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11l_gate2_train7200\top_false_positive_normals.csv --output-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate_hn_backflow\yolo11l_gate2_hn02 --top-k 22 --repeat 1 --link-mode hardlink
```

### 3. 先跑最高优先级实验 `l + HN 2%`
也可以直接用一键脚本：
```powershell
.\scripts\run_stage1_gate_l_hn.ps1
```

```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn.json
```

### 4. 用第二模型 `s` 给训练侧 normal 池重新打分
```powershell
uv run python .\scripts\stage1_score_train_normals.py --weights C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11s_gate2_train7200\weights\best.pt --data-root C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --output-dir C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11s_gate2_train7200 --device cpu --imgsz 640 --batch 2 --top-k 22
```

### 5. 构建 `s` 的 HN 2% 数据集
```powershell
uv run python .\scripts\stage1_build_hn_dataset.py --source-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200 --scores-csv C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11s_gate2_train7200\top_false_positive_normals.csv --output-dataset C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate_hn_backflow\yolo11s_gate2_hn02 --top-k 22 --repeat 1 --link-mode hardlink
```

### 6. 跑 `s + HN 2%`
也可以直接用一键脚本：
```powershell
.\scripts\run_stage1_gate_s_hn.ps1
```

```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_s_hn.json
```

### 7. 如果 `l + HN 2%` 有效，再跑 `l + HN 2% + Weighted BCE`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn_wbce.json
```

### 8. 如果 `l + HN 2% + Weighted BCE` 仍值得继续，再跑 `l + HN 2% + Focal`
```powershell
uv run python .\scripts\stage1_gate_train.py --config .\YOLOv11\configs\runtime\stage1_gate_l_hn_focal.json
```

## 跑完后优先看哪四个指标
1. `Spec@R99.5`
2. `Spec@R99.0`
3. `Prec@R99.0`
4. `PTR@R99.0`

## 如何判断要不要继续
- 如果 `l + HN 2%` 没能推高 `Spec@R99.5` 和 `Spec@R99.0`，先不要继续更重的 loss 变体。
- 如果 `l + HN 2%` 推高了两项高召回锚点特异度，再看 `Prec@R99.0` 与 `PTR@R99.0` 是否仍可接受。
- 如果 `s + HN 2%` 明显落后于 `l + HN 2%`，后续 loss 变体优先继续压在 `l` 上，不必给 `s` 扩更多实验。
- 如果 `l + HN 2% + Weighted BCE` 相比 `l + HN 2%` 没能继续改善 `Spec@R99.5 / Spec@R99.0`，则不建议继续上更重的 `Focal` 版本。
