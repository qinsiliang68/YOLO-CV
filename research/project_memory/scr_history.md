# SCR History

本文件用于记录仓库里曾经实际使用过的训练入口脚本与关键 `main.py` 任务演变，避免后续重构后丢失旧流程。

时间说明：

- 时间默认取 `git` 历史里的首次纳入时间或最近一次变更时间。
- 时区沿用提交时间里的 `+0800`。
- 本文件记录的是“曾被纳入并作为训练入口使用过的脚本/任务”，不是“今天仍推荐直接运行的全部入口”。

记录更新时间：

- `2026-04-09`

## 1. 训练入口脚本历史

| 首次出现 | 最近变更 | 入口脚本 | 代表命令 | 主要用途 | 当前状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-27 12:11:46 +0800 | 2026-03-28 15:37:44 +0800 | `scripts/cls_pretrain.ps1` | `.\scripts\cls_pretrain.ps1 -Config YOLOv11/configs/runtime/cls_source_cls6.json` | 源域分类预训练入口 | 仍在仓库 | PowerShell 包装，底层调用 `scripts/cls_pretrain.py` |
| 2026-03-27 12:11:46 +0800 | 2026-03-27 12:11:46 +0800 | `scripts/cls_finetune_target.ps1` | `.\scripts\cls_finetune_target.ps1 -Config YOLOv11/configs/runtime/cls_target_struct6.json` | 目标域分类微调入口 | 仍在仓库 | PowerShell 包装，底层调用 `scripts/cls_finetune_target.py` |
| 2026-03-28 15:37:44 +0800 | 2026-04-08 15:19:32 +0800 | `main.py` | `uv run main.py` / `uv run main.py --task <task>` | 统一训练入口 | 仍在仓库 | 历史上承载过 source、HN、PTSG、RCIS、formal capacity、formal HN、info-sampling-lite 等任务 |
| 2026-03-29 18:45:30 +0800 | 2026-03-30 13:29:16 +0800 | `scripts/history/main_pipeline_full_174f329.py` | `uv run python scripts/history/main_pipeline_full_174f329.py` | 旧版全流程入口 | 已归档移除 | 覆盖 source 分类、target 微调、CAM 导出和 pseudo-box 生成；仅保留历史参考意义 |
| 2026-03-30 13:03:25 +0800 | 2026-04-08 11:27:45 +0800 | `main_cls6_sweep.py` | `uv run main_cls6_sweep.py` | 六分类 sweep 兼容入口 | 仍在仓库 | 兼容包装，当前实际走 `main.py` |
| 2026-04-01 11:34:01 +0800 | 2026-04-01 11:54:15 +0800 | `scripts/run_stage1_gate_l_hn.ps1` | `powershell -ExecutionPolicy Bypass -File scripts\run_stage1_gate_l_hn.ps1` | `yolo11l-cls` 的 HN 2% 一键回流训练 | 仍在仓库 | 包含打分、构建 HN 数据集、训练三步 |
| 2026-04-01 11:34:01 +0800 | 2026-04-01 11:54:15 +0800 | `scripts/run_stage1_gate_s_hn.ps1` | `powershell -ExecutionPolicy Bypass -File scripts\run_stage1_gate_s_hn.ps1` | `yolo11s-cls` 的 HN 2% 一键回流训练 | 仍在仓库 | 与 `l` 版对应 |
| 2026-04-04 19:04:49 +0800 | 2026-04-08 11:27:45 +0800 | `main_A.py` | `uv run main_A.py` | formal 二分类 gate capacity 入口 | 仍在仓库 | 当前委托到 `stage1_formal_gate_capacity` |
| 2026-04-04 19:04:49 +0800 | 2026-04-08 15:27:58 +0800 | `main_B.py` | `uv run main_B.py` | 机器 B 人类友好入口 | 仍在仓库 | 最初对应 formal cls6 capacity；当前分支上已改为 `stage1_formal_gate_hn_ns_all` |
| 2026-04-06 02:31:50 +0800 | 2026-04-08 11:27:45 +0800 | `main_HN_A.py` | `uv run main_HN_A.py` | formal `yolo11m` HN sweep 入口 | 仍在仓库 | 当前委托到 `stage1_formal_gate_hn_m_sweep` |
| 2026-04-06 02:31:50 +0800 | 2026-04-08 11:27:45 +0800 | `main_HN_B.py` | `uv run main_HN_B.py` | formal `yolo11x` HN cross-check 入口 | 仍在仓库 | 当前委托到 `stage1_formal_gate_hn_x_crosscheck` |
| 2026-04-06 02:34:06 +0800 | 2026-04-08 11:27:45 +0800 | `main_HN.py` | `uv run main_HN.py` | formal HN 总入口 | 仍在仓库 | 当前委托到 `stage1_formal_gate_hn_all` |
| 2026-04-08 15:19:32 +0800 | 2026-04-08 15:19:32 +0800 | `scripts/run_stage1_hn_ns_all.ps1` | `powershell -ExecutionPolicy Bypass -File scripts\run_stage1_hn_ns_all.ps1` | `yolo11n/s` 的 formal HN 全 sweep 启动脚本 | 仍在仓库 | 用于机器侧串行跑 `n` 和 `s` 的 `hn00, hn02, ..., hn20` |

## 2. `main.py` 关键训练任务演变

| 时间 | 提交 | 任务/变化 | 说明 |
| --- | --- | --- | --- |
| 2026-03-28 15:37:44 +0800 | `0e6ee06` | 初始统一训练入口 | `main.py` 进入仓库，开始承载统一训练工作流 |
| 2026-03-30 13:03:25 +0800 | `28d6c99` | `cls6_sweep` | 六分类统一容量扫描进入主入口体系 |
| 2026-04-01 11:41:37 +0800 | `8f0008b` | `main.py` 成为统一人类友好训练入口 | 后续默认通过 `main_entry.json` 或显式 `--task` 分派 |
| 2026-04-01 16:42:14 +0800 | `07b0351` | `stage1_gate_ptsg_eval` | stage-1 PTSG / post-hoc selective gate 路线进入主入口 |
| 2026-04-01 18:20:58 +0800 | `177abe6` | `stage1_gate_ptsg_nextwave` | PTSG next-wave 入口加入 |
| 2026-04-01 21:24:54 +0800 | `832edb7` | `stage1_gate_embedding_supcon_eval` | strong-embedding / SupCon gate 路线加入 |
| 2026-04-03 14:39:56 +0800 | `3855ba6` | `stage1_gate_maxfilter_suite` | 一键 max-filter / HardMix / selective / WBCE / focal 套件加入 |
| 2026-04-04 12:35:37 +0800 | `3ce4a79` | `stage1_gate_rcis_suite` | RCIS information-sampling 第一波加入 |
| 2026-04-04 19:04:49 +0800 | `781227c` | `stage1_formal_gate_capacity` / `stage1_formal_cls6_capacity` | formal capacity scan 进入主入口 |
| 2026-04-06 02:31:50 +0800 | `01dac23` | `stage1_formal_gate_hn_m_sweep` / `stage1_formal_gate_hn_x_crosscheck` | formal HN 的 `m/x` 入口加入 |
| 2026-04-06 02:34:06 +0800 | `7766de5` | `stage1_formal_gate_hn_all` | formal HN 总入口加入 |
| 2026-04-08 13:20:45 +0800 | `5bdea94` | `stage1_formal_gate_info_sampling_lite` | lite 版有效信息量非线性重加权任务加入 |
| 2026-04-08 15:19:32 +0800 | `ffb8452` | `stage1_formal_gate_hn_n_sweep` / `stage1_formal_gate_hn_s_sweep` / `stage1_formal_gate_hn_ns_all` | `n/s` HN 全 sweep 相关任务加入 |

## 3. 最新 lite 资料时间线

| 时间 | 提交 | 内容 | 文件 |
| --- | --- | --- | --- |
| 2026-04-08 13:20:45 +0800 | `5bdea94` | info-sampling-lite suite 进入主入口 | `main.py`、`stage1_formal_gate_info_sampling_lite.json`、4 个主脚本 |
| 2026-04-08 13:53:43 +0800 | `92b4864` | preflight 检查加固 | `scripts/stage1_formal_gate_info_sampling_lite.py` |
| 2026-04-08 14:46:42 +0800 | `b900afd` | smoke preflight 范围修复 | `scripts/stage1_formal_gate_info_sampling_lite.py` |
| 2026-04-09 11:48:10 +0800 | `5b66b2d` | info-sampling-lite png rebuild 映射清单 | `stage1_info_sampling_lite_png_file_manifest.csv`、`stage1_info_sampling_lite_png_manifest_usage.md`、`stage1_info_sampling_lite_png_rebuild_map.csv` |

## 4. 备注

- `main_B.py` 的角色历史上发生过变化，因此查旧实验时不要只看今天文件内容，必须同时看对应提交时间。
- `scripts/history/main_pipeline_full_174f329.py` 已从当前仓库删除，但可通过 `git show 216bcb1:scripts/history/main_pipeline_full_174f329.py` 找回。
- 本文件优先记录“用户会直接运行的脚本入口”。底层辅助脚本如 `stage1_gate_train.py`、`stage1_build_hn_dataset.py`、`stage1_formal_gate_epoch_eval.py` 等不在此表展开。
