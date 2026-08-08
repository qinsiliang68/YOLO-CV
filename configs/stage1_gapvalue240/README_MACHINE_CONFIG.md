# Machine configuration boundary

Only paths and hardware/resource fields in these YAML files may be edited. Scientific fields such as method, budget, seed, epochs, batch size, guard ratio, learning rate, threshold rules, or arm are rejected by the loader.

Use one file per machine. `machine_11` and `machine_12` are reserve nodes and initially receive empty shards.

Dynamic replay scientific queues use neutral planning slots (`M01` through `M10`).
`DYNAMIC_MACHINE_SLOT_MAP_v1.csv` maps those slots to the initial real machine
identities. Formal workers are authorized by a separate, versioned assignment
manifest; the queue slot is not execution authorization. Reassign a complete
cycle/seed block by generating a new assignment version. Do not edit training
code or the frozen queue on a training node.

The current repository trainer consumes one manifest directory containing four canonical files. Each machine YAML must therefore bind all of:

```text
train_manifest
normal_train_manifest
val_model_defect_manifest
val_model_normal_manifest
```

Evaluation additionally requires the four `val_cal_*` and `val_op_*` paths. `dataset_root` must be the directory against which `canonical_image_relpath` resolves; stale absolute `source_image_path` values are not used as the primary identity path.

Runtime v1.2 additionally requires:

```text
staging_root                 same volume as dataset_root; hardlink-only cache
machine_asset_report         one-time PASS report covering all 384k manifest rows/images
minimum_staging_free_gib     operational guard, default template 2 GiB
minimum_output_free_gib      operational guard, default template 20 GiB
maximum_staging_files        file-entry ceiling, default template 151000
gpu_memory_release_threshold_mib  resource-canary reference value
```

`cache_root`, `local_scratch_root`, and failed/incomplete staging are reproducible and may be cleared by AIOps when disk is low. Frozen inputs, selection CSVs, contracts, machine reports, VALIDATED predictions/metrics, and artifact manifests are not cache.

## v2 正式训练前顺序

机器 YAML 只允许路径与硬件字段。完成配置后依次运行：磁盘/GPU preflight、共享 root canary、one-job real-data canary，再由中央生成 gate/release/assignment。训练节点只复制 assignment 中为本机生成的 single-job 命令；不得添加 method、arm、seed、schedule、batch、workers 或自动下一任务参数。

完整命令、改派、lease/fencing、热备和故障恢复见：

```text
docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATOR_BRIEFING_V2.md
docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md
docs/stage1_gapvalue240/RUN_QUEUE_V2_README.md
```
