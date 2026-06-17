# Stage-1 OOF 10 Fold Training Plan

本文档记录 Stage-1 分类门控的 10 折 OOF 训练清单生成方法和训练机执行方法。

当前目的不是重新做最终测试集评估，而是给训练集内每一张图生成一次“模型没有见过它时”的预测分数。后续会用这些 out-of-fold 分数判断哪些正常样本、缺陷样本、边界样本、误分类样本更有训练价值，再设计样本加权和重采样实验。

## 核心思路

训练集一共有 120000 张：

- 缺陷图：60000 张
- 正常图：60000 张

10 折 OOF 的含义是：

- 每一折拿约 9/10 训练，也就是约 108000 张。
- 每一折留下约 1/10 做本折验证，也就是约 12000 张。
- 10 折跑完后，每张训练图都会被某一个“没见过它”的模型预测一次。
- 这些预测结果后面用于困难样本、边界样本、误分类样本、高置信错误样本等变量的量化。

这不是普通随机切行。脚本会先尽量按 `pipe_id`、`inspection_id`、`video_id` 之类字段做 group-disjoint。如果当前 manifest 没有这些字段，就退而用文件名数字桶做近邻分组，默认每 1000 个编号一个桶。这样可以降低连续视频帧或近邻帧同时进入训练和验证的风险。

当前数据的训练 manifest 没有明确的 `pipe_id`、`inspection_id`、`video_id` 字段，所以实际采用：

- 分组方式：`numeric_filename_bucket`
- 桶大小：`1000`
- 随机种子：`20260606`
- 折数：`10`

## 生成 10 折清单

在仓库根目录运行：

```powershell
uv run python scripts\build_stage1_oof_folds.py --folds 10 --dataset-root data\final_sewerml_dataset --output-root artifacts\stage1_oof_folds_10fold_20260617 --overwrite
```

输出目录：

```text
artifacts/stage1_oof_folds_10fold_20260617/
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `train_oof_assignments.csv` | 120000 张训练图的 fold 分配总表 |
| `fold_summary.csv` | 每折正负样本数、训练样本数 |
| `group_summary.csv` | 每个 group 被分到哪个 fold |
| `metadata.json` | 生成参数、分组来源、样本总数 |
| `fold_jobs.csv` | 每折对应的建议训练命令 |
| `folds/fold_00/manifests/` 到 `folds/fold_09/manifests/` | 每折实际训练用 manifest |

每个 fold 的 manifest 目录里有：

| 文件 | 作用 |
| --- | --- |
| `train_manifest.csv` | 本折训练用缺陷图，来自其他 9/10 |
| `normal_train_manifest.csv` | 本折训练用正常图，来自其他 9/10 |
| `holdout_manifest.csv` | 本折留下的缺陷图，约 1/10 |
| `normal_holdout_manifest.csv` | 本折留下的正常图，约 1/10 |
| `val_model_manifest.csv` | 等同于 `holdout_manifest.csv` |
| `normal_val_model_manifest.csv` | 等同于 `normal_holdout_manifest.csv` |

也就是说，训练脚本加载某一折时，只会用其他 9/10 训练，并用本折 1/10 作为验证集。

## 当前本地校验结果

当前本地生成结果的关键校验：

```text
split_groups=0
fold_00: total=12036 pos=6020 neg=6016
fold_01: total=12003 pos=6012 neg=5991
fold_02: total=12039 pos=6039 neg=6000
fold_03: total=12004 pos=6007 neg=5997
fold_04: total=11987 pos=5987 neg=6000
fold_05: total=11959 pos=5976 neg=5983
fold_06: total=11997 pos=5994 neg=6003
fold_07: total=11945 pos=5925 neg=6020
fold_08: total=12001 pos=6007 neg=5994
fold_09: total=12029 pos=6033 neg=5996
```

解释：

- `split_groups=0` 表示没有任何一个 group 被拆到多个 fold。
- 每折大约 12000 张。
- 每折正负样本基本都是 6000/6000。
- 因为训练集本身不是长尾分布，这里的重点不是处理长尾，而是防止近邻泄漏，同时保持每折比例稳定。

还校验过 fold 00：

```text
holdout_manifest.csv == val_model_manifest.csv: True
normal_holdout_manifest.csv == normal_val_model_manifest.csv: True
```

## 单折训练命令

每折推荐训练 `yolo11l`，200 epoch，224 分辨率。以 fold 00 为例：

```powershell
uv run python scripts\train_stage1_cls_sweep.py --mode full --models l --epochs 200 --imgsz 224 --batch 128 --workers 4 --save-period 1 --keep-data --device 0 --dataset-root data\final_sewerml_dataset --manifest-dir artifacts\stage1_oof_folds_10fold_20260617\folds\fold_00\manifests --work-root data\stage1_oof_workdir\fold_00 --runs-root YOLOv11\runs\stage1_oof_10fold\fold_00
```

参数重点：

- `--manifest-dir` 指向当前 fold 的 manifest 目录。
- `--work-root` 每折单独给目录，避免不同 fold 互相覆盖。
- `--runs-root` 每折单独给目录，方便中途失败后查日志和恢复。
- `--save-period 1` 每个 epoch 存一次，减少中断损失。
- `--keep-data` 保留临时分类数据目录，方便排查。

`fold_jobs.csv` 已经写好了 10 条建议命令。分机器跑时，可以给每台机器分配一个或多个 fold。

## 多机器分配建议

如果有 10 台机器：

- 每台机器跑 1 个 fold。
- 理想情况下，墙钟时间接近单个 `yolo11l` 200 epoch 的训练时长。

如果少于 10 台机器：

- 5 台机器：每台跑 2 个 fold。
- 2 台机器：每台跑 5 个 fold。
- 1 台机器：顺序跑 10 个 fold。

已知参考时间：

- `yolo11l` 200 epoch，224，batch 128，单张 RTX 3090 约 13 小时 1 分。
- 10 折总 GPU 时间约 130 小时。
- 10 台机器并行时，墙钟时间大约就是 13 小时级别。

## 快速 dry-run 检查

正式跑之前，建议先在训练机做一个极小 dry-run，确认路径、manifest、权重文件都能读到：

```powershell
uv run python scripts\train_stage1_cls_sweep.py --mode smoke --models n --epochs 1 --train-per-class 2 --val-per-class 2 --dry-run --dataset-root data\final_sewerml_dataset --manifest-dir artifacts\stage1_oof_folds_10fold_20260617\folds\fold_00\manifests --work-root data\stage1_oof_workdir_smoke_test --runs-root YOLOv11\runs\stage1_oof_manifest_smoke_test --yolo-root YOLOv11 --keep-data
```

本地已经验证过该 dry-run 可以走通到训练入口。

## 中断和恢复

每个 fold 独立，互不影响。某一折失败时：

1. 先保留该 fold 的 `runs-root` 和 `work-root`。
2. 查看该 fold 的日志和 checkpoint。
3. 只重跑失败的 fold，不需要重跑其他 fold。

如果要彻底重跑某一折，可以先移动或删除该 fold 对应目录：

```text
data/stage1_oof_workdir/fold_XX
YOLOv11/runs/stage1_oof_10fold/fold_XX
```

不要改动 `train_oof_assignments.csv` 或 `folds/fold_XX/manifests/`，否则 OOF 分配会变。

## 后续需要补的步骤

10 折训练完成后，还需要补一个预测汇总脚本：

1. 对每个 fold 的 best checkpoint，预测该 fold 的 `val_model_manifest.csv` 和 `normal_val_model_manifest.csv`。
2. 合并 10 折预测结果，得到 120000 张训练图的 OOF 预测表。
3. 用 OOF 预测表计算样本价值变量，例如：
   - 缺陷误判为正常，尤其是高置信错误。
   - 正常误判为缺陷。
   - 低置信边界样本。
   - 近邻抱团样本。
   - 可能噪声样本。
4. 再把这些变量用于后续重采样、加权训练、消融实验。

当前提交只负责可靠地产生 10 折 OOF 训练清单，并让训练脚本可以按 fold manifest 启动训练。
