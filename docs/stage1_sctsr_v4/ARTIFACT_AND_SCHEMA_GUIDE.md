# SCTSR v4 产物、字段和消费关系

## 1. 设计原则

- 大表：按 run/epoch 分区的 canonical Zstd Parquet；
- 小合同/receipt/index：canonical UTF-8 JSON、sorted keys、LF、禁止 NaN/Infinity；
- checkpoint：原子 PyTorch 文件并有 payload digest；
- 每项产物都记录 owner、source、consumer、lifecycle、bytes、SHA；
- synthetic portable columnar 不得进入 formal run；
- 目录存在和文件数量不等于语义完整。

## 2. Run 根目录

```text
<run>/
  00_contract/                 immutable contract/release/source snapshots
  01_assets/                   registry, identity, pools, dataset binding
  02_lineage/                  parent/branch lineage and schedule
  03_epoch_transactions/       immutable complete generations
  08_receipts/                 epoch and resume append-only chains
  09_quarantine/               append-only failed/orphan generations
  10_resume_setup/             isolated upstream setup for RESUME
  06_predictions/              E200 formal predictions
  07_evaluation/               frontier and summary
  PREPARED_TRAINER_BINDING.json
  FORMAL_IDENTITY.json
  FORMAL_AUTHORIZATION_BINDING.json
  RUN_MANIFEST.json
  PARENT_RECEIPT.json or BRANCH_RECEIPT.json
  ARTIFACT_INDEX*.json
```

Common parent 没有正式 val_op endpoint；branch 必须有。Logical child index 对 E1-E120
只引用 parent 的物理产物，不复制后伪装成 child 自己生成。

### 2.1 R2 addendum 构造证据

R2 `POOL_MANIFEST.json` 和 `QUOTA_AUDIT.json` 中的 `pool_build_audit` 必须逐字段
相同并包含：policy、selection seed、candidate/excluded/selected count、T overlap、
三字段 exact 布尔值、四字段非 exact 布尔值、唯一 relaxed field、378 条
`displacement_records` 和 ledger digest。每条 displacement 保存：

```text
selected_sample_id, y_true, historical_dynamic_bucket, oof_fold,
requested_oof_group_id, selected_oof_group_id, selection_counter_hash
```

机器 validator 必须重算 displacement ledger digest、row count、requested/selected
group 不同、pool identity/content digest、displacement=379 和
group TV=`0.12633333333333333`。`R2_U`、`R2_F` 和 fallback 的
schedule/pool binding 必须回指同一 R2 `POOL_MANIFEST.json`；复制后改路径但保持不同
manifest SHA 不能冒充共享 pool。

## 3. Occurrence ledger（49 fields）

每个 optimizer-visible base/replay occurrence 一行：

```text
run_id, parent_id, arm_id, training_seed, epoch, base_batch_index,
global_step_before, occurrence_role, occurrence_index_in_step, sample_id,
y_true, replay_role, identity_pool_id, identity_group, selection_policy,
selection_reason_code, oof_fold, oof_group_id, oof_group_semantic,
historical_dynamic_bucket, augmentation_seed, augmentation_trace_digest,
replay_count_before, replay_count_after, last_replay_epoch,
last_replay_epoch_reason, epochs_since_last_replay,
epochs_since_last_replay_reason, logit_normal, logit_defect, p_defect_raw,
ce_unreduced, margin_defect_minus_normal, predicted_label_argmax,
correct_argmax, oof_reference_probability, oof_reference_reason,
rho_candidate_signal, rho_reason, row_generation, planned_replay_epoch,
planned_replay_epoch_reason, planned_step_slot, planned_step_slot_reason,
cumulative_replay_count_before, cumulative_replay_count_after,
pool_multiplicity_target, schedule_family, fallback_state
```

Base rows 的 replay-only 字段必须使用明确 reason，不留空。RHO/OOF reference 是
candidate signal，不是 utility。Formal 每 epoch base occurrence 恰好 120,000；
replay occurrence由 arm schedule决定。

## 4. Optimizer-step ledger（51 fields）

每个 base step 一行，正式每 epoch 恰好 938：

```text
run_id, parent_id, arm_id, training_seed, epoch, base_batch_index,
global_step_before, global_step_after, base_batch_size,
replay_microbatch_size, replay_rate_numerator, replay_rate_denominator,
base_loss, replay_loss, combined_loss_for_reporting, base_loss_items,
parameter_grad_norm_before_clip, parameter_grad_norm_after_clip,
clip_max_norm, clip_reason, optimizer_step_count_delta, learning_rates,
optimizer_hyperparameters, amp_scale_before, amp_scale_after, amp_reason,
overflow_or_step_skipped, ema_updates_before, ema_updates_after,
scheduler_state_digest, warmup_progress, bn_digest_before_replay,
bn_digest_after_replay_restore, bn_reason, rng_digest_before_base,
rng_digest_before_replay, rng_digest_after_replay_restore, rng_reason,
replay_rng_fork_digest, replay_rng_fork_reason, base_augmentation_digest,
replay_augmentation_digest, replay_augmentation_reason,
dataloader_wait_seconds, base_forward_seconds, replay_forward_seconds,
backward_seconds, optimizer_seconds, write_buffer_bytes, status,
row_generation
```

`optimizer_step_count_delta` 必须为 1；EMA before/after 必须增 1；overflow/skip
不能被静默计作完成 step。

## 5. Exposure ledger（42 fields）

每 epoch 一行，汇总计划/实际守恒：

```text
run_id, parent_id, arm_id, training_seed, epoch, denominator_role,
base_denominator_planned, base_denominator_actual, rate_numerator,
rate_denominator, replay_numerator_planned, replay_numerator_actual,
unique_replay_ids, repeat_occurrences, cumulative_occurrences,
multiplicity_min, multiplicity_max, multiplicity_mean, multiplicity_q0,
multiplicity_q25, multiplicity_q50, multiplicity_q75, multiplicity_q100,
base_optimizer_steps_planned, base_optimizer_steps_actual,
ema_updates_delta, scheduler_epoch_transitions_delta, base_order_digest,
base_augmentation_digest, replay_schedule_digest, identity_pool_digest,
occurrence_partition_sha256, step_partition_sha256,
telemetry_partition_sha256, checkpoint_sha256, write_seconds,
dataloader_wait_seconds, training_seconds, evaluation_seconds,
disk_bytes_written, transaction_generation, validation_status
```

`repeat_occurrences` 是本 epoch pool 内重复；累计逐 ID repeat 需从 occurrence history
与 multiplicity summary联合解释，不能只看该字段。

## 6. Telemetry（50 fields）

每秒采样：UTC/monotonic、run/arm/seed/epoch、PID、process CPU/RSS/VMS/read/write、
system CPU/memory、GPU index/UUID/name/utilization/memory/temp/power、CUDA
allocated/reserved/max、run/artifact volume total/free/used，以及每个 provider 的
status/reason/error。缺 provider 不能填 0 冒充真实值，必须写 reason；正式所需 provider
缺失时 validation失败。

## 7. Checkpoint

每 epoch 保存：

- schema/epoch/global step；
- model state；
- EMA state + update count；
- optimizer state；
- scheduler state；
- AMP scaler；
- Python/NumPy/Torch CPU/all CUDA RNG；
- source/contract/asset/runtime/base/seed identity；
- checkpoint payload digest。

不得只保存 weights。E120 parent是 branch因果共同起点；E200固定 endpoint；中间全部
checkpoint保留用于恢复和预注册轨迹，不用于选 best。

## 8. Prediction（18 fields）

正式 E200/EMA/val_op 每样本一行：

```text
run_id, arm_id, training_seed, split_role, split_manifest_path,
split_manifest_sha256, sample_id, y_true, logit_normal, logit_defect,
p_defect_raw, checkpoint_epoch, checkpoint_sha256, model_variant,
source_tree_digest, prediction_generation, sample_label_identity_digest,
artifact_row_count
```

`p_defect_raw` 必须等于两 logits 的 softmax，不允许保存显示层四舍五入值。全体 sample
和 label digest 必须与 val_op split bundle一致。

## 9. Frontier（14 fields）

```text
fn_budget, actual_fn, tn, fp, tp, threshold, threshold_rule, tie_size,
reachable, defect_count, normal_count, normalized_tn, checkpoint_sha256,
prediction_artifact_sha256
```

`fn_budget=0..95` 恰好 96 行。相同 score 不拆 tie；不可达预算必须显式
`reachable=false`，不得插值伪造。Summary另外记录 normalized AUC、TN_at_FN95、
FN_at_TN68253 及两个独立 threshold。

## 10. Generation manifest/receipt/index

Generation manifest列出 transaction内每个文件 relative path、bytes、SHA，绑定 epoch、
generation、parent/source/contract/asset/RNG/previous-generation identity。Receipt chain
每行绑定 generation manifest SHA 和上一 receipt digest。Pointer/index只是可重建
secondary metadata，不能凌驾于 receipt commit point。

## 11. 消费者

| producer | direct consumer |
| --- | --- |
| asset/dataset validators | release authority, prepared trainer, closeout |
| pool builder | schedule builder, identity manifest, branch validator |
| schedule | runtime, occurrence/exposure audit, paired analysis |
| occurrence/step | epoch finalize, resume history, fairness audit |
| telemetry | disk/resource audit, failure diagnosis |
| checkpoint | next epoch/resume, endpoint publisher |
| prediction | frontier evaluator |
| frontier/summary | paired statistics only after run closeout |
| receipts/index | recovery, validate_run, closeout |

下游消费者必须验证上游 SHA/schema/identity，不得只打开文件并读取可用列。
