# Checkpoint 提取任务

## 背景

我需要从本机提取 55 个 YOLO11 分类模型的 checkpoint (.pt 文件)，用于在另一台机器上做推理评估。这些 checkpoint 分布在两个目录下，每个都是特定 run 的特定 epoch。

## 输入清单

仓库内有一份 CSV 清单：`research/checkpoint_extract_manifest.csv`

格式：`model,ratio,epoch,run_parent,run_name,checkpoint_file`

其中：
- `run_parent` = `stage1_formal_gate`（5 个 baseline）或 `stage1_formal_gate_hn`（50 个 HN run）
- `checkpoint_file` = 形如 `epoch_076.pt` 的文件名

## checkpoint 存储位置

checkpoint 分散在以下位置（请根据本机实际路径调整）：

1. **n / s / m baseline (hn00)**：`YOLOv11/runs/stage1_formal_gate/{run_name}/weights/{checkpoint_file}`
2. **n / s / m HN runs (hn02-hn20)**：`YOLOv11/runs/stage1_formal_gate_hn/{run_name}/weights/{checkpoint_file}`
3. **l baseline + HN runs**：同上路径，或在外部硬盘/归档目录中
4. **x baseline + HN runs**：可能在 `D:\` 盘归档目录中（之前通过 `archive_and_export.py` 导出过）

## 要做的事

1. 读取 `research/checkpoint_extract_manifest.csv`
2. 逐行查找对应的 .pt 文件（先在仓库内 runs 目录找，找不到再搜归档目录）
3. 将 55 个 .pt 文件统一复制到一个输出目录，结构如下：

```
checkpoint_export/
  yolo11n/
    hn00_epoch_076.pt
    hn02_epoch_043.pt
    ...
  yolo11s/
    hn00_epoch_077.pt
    ...
  yolo11m/
    ...
  yolo11l/
    ...
  yolo11x/
    ...
```

文件命名规则：`{ratio}_epoch_{epoch:03d}.pt`

4. 生成一份验证报告：
   - 55 个文件是否全部找到
   - 每个文件的大小
   - 缺失文件列表（如果有）

5. 输出目录打包为 zip 或直接上传百度网盘。

## 注意

- 只需要清单中指定的 epoch，不要复制其他 epoch
- x 模型的 checkpoint 可能在 `D:\stage1_formal_gate_hn_yolox-cls\pt_files\` 或 `D:\BaiduNetdiskDownload\` 下
- 如果某个 .pt 文件找不到，记录下来并继续处理其余文件
- 总共约 55 个文件，预计总大小 2-5 GB
