# Machine configuration boundary

Only paths and hardware/resource fields in these YAML files may be edited. Scientific fields such as method, budget, seed, epochs, batch size, guard ratio, learning rate, threshold rules, or arm are rejected by the loader.

Use one file per machine. `machine_11` and `machine_12` are reserve nodes and initially receive empty shards.

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
