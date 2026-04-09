# Ingest Manifest

日期：

- `2026-04-09`

来源：

- 下载目录：`C:\baidunetdiskdownload`
- 输入压缩包：`stage1_info_sampling_lite_materials.zip`
- 同批文件：
  - `stage1_info_sampling_lite_compress.log`
  - `stage1_info_sampling_lite_compress_done.json`

吸收目标：

- `research/materials/stage1_formal/gate_info_sampling_lite/`
- `research/results/stage1_formal/gate_info_sampling_lite/`
- `YOLOv11/runs/stage1_formal_gate_info_sampling_lite/`
- `$out/generated_configs/stage1_formal/gate_info_sampling_lite/`
- `research/materials/stage1_formal/manifests/`

本次吸收后的文件统计：

- `research/materials/stage1_formal/gate_info_sampling_lite/`：`3696` files
- `research/results/stage1_formal/gate_info_sampling_lite/`：`59` files
- `YOLOv11/runs/stage1_formal_gate_info_sampling_lite/`：`85` files
- `$out/generated_configs/stage1_formal/gate_info_sampling_lite/`：`4` files
- `research/materials/stage1_formal/manifests/` 中与 lite 相关的新清单：`6` files

本批资料包含：

- A2/A3/A4 的 `candidate_pool_scores.csv` 与 `score_stats.*`
- smoke `A4 3ep` 的材料、结果与 non-pt run 资料
- full `A2/A3/A4` 的 formal materials
- lite 结果目录下的：
  - `PREFLIGHT_gate_info_sampling_lite.*`
  - `SUMMARY_gate_info_sampling_lite.md`
  - `REPRODUCE_gate_info_sampling_lite.md`
  - `ARTIFACTS_gate_info_sampling_lite.md`
  - `suite_context.*`
  - `tables/`, `figures/`, `captions/`, `appendix/`, `manifests/`
- png rebuild / usage / verification manifests

本批资料不包含：

- `.pt` 权重文件

清理动作：

- 下载目录中的以下文件已计划删除：
  - `stage1_info_sampling_lite_materials.zip`
  - `stage1_info_sampling_lite_compress.log`
  - `stage1_info_sampling_lite_compress_done.json`
- 临时解压目录已计划删除：
  - `C:\z1`

备注：

- 本次吸收优先保留 formal materials、results、generated configs 与 non-pt run source-of-truth。
- 这意味着第 6 章现在已经具备从真实结果目录回填 A0--A4 lite 图表与数值的基础。
