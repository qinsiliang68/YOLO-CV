# Stage1 campaign field-gap report

## Decision

The current evidence is broad but not sufficient to explain same-selection seed reversals causally.
The 10-fold OOF trajectory is complete for static sample dynamics, while the replay runs lack
realized exposure, role-separated loss, key-epoch raw predictions, a no-replay arm, and blind evaluation.

## Inventory scope

- Field rows: 1,493
- Logical sources: 8
- Currently collected rows: 1,366
- P0 unresolved rows: 18
- P1 pilot or unresolved rows: 14

A row is a source-field identity, not a unique physical file. Storage must be summed by path only.

## P0 gaps before confirmatory launch

| Field | Granularity | Why required |
|---|---|---|
| `base_defect_loss_per_epoch` | RUN_EPOCH_CLASS_ROLE | Required to test whether normal replay harms the weak-defect tail. |
| `base_normal_loss_per_epoch` | RUN_EPOCH_CLASS_ROLE | Required to distinguish configured replay from realized optimization exposure. |
| `base_vs_replay_loss_split` | RUN_EPOCH_ROLE | Required to distinguish configured replay from realized optimization exposure. |
| `blind_or_external_test` | DATASET_SPLIT | Required to separate mechanism discovery from final generalization evidence. |
| `difficult_normal_tail_score_trajectory` | RUN_SAMPLE_CHECKPOINT | Required to verify that dynamic replay actually lowers the target normal tail. |
| `epoch_150_checkpoint` | RUN_CHECKPOINT | The prior campaign did not retain the checkpoint needed to inspect the 150-160 epoch window. |
| `guard_defect_loss_per_epoch` | RUN_EPOCH_CLASS_ROLE | Required to test whether normal replay harms the weak-defect tail. |
| `guard_weight_per_epoch` | RUN_EPOCH | Required to test whether normal replay harms the weak-defect tail. |
| `key_epoch_checkpoint` | RUN_CHECKPOINT | Required for trajectory reconstruction, recovery, and key-epoch gradient probes. |
| `key_epoch_val_op_raw_predictions` | RUN_SAMPLE_CHECKPOINT | Required before confirmatory launch because it identifies the causal replay mechanism. |
| `no_replay_arm` | RUN | Required before confirmatory launch because it identifies the causal replay mechanism. |
| `per_sample_replay_exposure_count` | RUN_SAMPLE_EPOCH | Required to distinguish configured replay from realized optimization exposure. |
| `replay_normal_loss_per_epoch` | RUN_EPOCH_CLASS_ROLE | Required to distinguish configured replay from realized optimization exposure. |
| `replay_weight_per_epoch` | RUN_EPOCH | Required to distinguish configured replay from realized optimization exposure. |
| `run_arm_schedule_manifest` | RUN_EPOCH | Required to distinguish configured replay from realized optimization exposure. |
| `selected_checkpoint_and_threshold_provenance` | RUN_EVALUATION | Prevents implicit checkpoint or threshold cherry-picking. |
| `tail_probe_membership_manifest` | SAMPLE | Probe membership must remain fixed across arms and seeds. |
| `weak_defect_tail_score_trajectory` | RUN_SAMPLE_CHECKPOINT | Required to test whether normal replay harms the weak-defect tail. |

## P1 pilot fields

P1 fields are scientifically useful but must not delay the P0 six-arm pilot. Gradient payloads remain
unmeasured and therefore require a one-checkpoint benchmark before fleet-wide collection.

| Field | Cost class | Hypothesis |
|---|---|---|
| `augmentation_realization_digest_per_epoch` | MEDIUM | H4_SEED_SENSITIVITY |
| `batch_role_composition_per_epoch` | LOW | H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD;H4_SEED_SENSITIVITY |
| `diverse_grad_align_score` | MEDIUM | H5_GRADIENT_ALIGNMENT;H6_DIVERSITY_REDUNDANCY |
| `grad_align_defect_tail` | PILOT_REQUIRED | H3_WEAK_DEFECT_GUARD;H5_GRADIENT_ALIGNMENT |
| `grad_align_guard_score` | MEDIUM | H3_WEAK_DEFECT_GUARD;H5_GRADIENT_ALIGNMENT |
| `grad_align_normal_tail` | PILOT_REQUIRED | H5_GRADIENT_ALIGNMENT |
| `grad_align_score` | MEDIUM | H5_GRADIENT_ALIGNMENT |
| `grad_mag_align_score` | MEDIUM | H5_GRADIENT_ALIGNMENT |
| `grad_mag_score` | MEDIUM | H5_GRADIENT_ALIGNMENT |
| `grad_mag_score` | PILOT_REQUIRED | H5_GRADIENT_ALIGNMENT |
| `gradient_diversity_embedding` | PILOT_REQUIRED | H5_GRADIENT_ALIGNMENT;H6_DIVERSITY_REDUNDANCY |
| `gradient_outlier_score` | LOW_AFTER_GRADIENT_EXPORT | H5_GRADIENT_ALIGNMENT;H6_DIVERSITY_REDUNDANCY |
| `minibatch_order_digest_per_epoch` | LOW | H4_SEED_SENSITIVITY |
| `optimizer_step_summary_per_epoch` | LOW | H4_SEED_SENSITIVITY;H5_GRADIENT_ALIGNMENT |

## Storage lower bounds

The forecast excludes gradient payloads until the pilot measures dimensions, compression, and runtime.

| Scenario | Runs | Projected GiB | Estimate class |
|---|---:|---:|---|
| 6arms_x_14seeds | 84 | 47.44 | LOWER_BOUND_EXCLUDES_GRADIENT_PAYLOAD |
| 6arms_x_22seeds | 132 | 74.48 | LOWER_BOUND_EXCLUDES_GRADIENT_PAYLOAD |
| 6arms_x_30seeds | 180 | 101.51 | LOWER_BOUND_EXCLUDES_GRADIENT_PAYLOAD |

## Highest-value investigation order

1. Establish causal replay effect with NR_NO_REPLAY and matched seeds.
2. Test dynamic replay decay while measuring realized replay exposure.
3. Test weak-defect guard protection on fixed tail probes at key epochs.
4. Quantify seed reversal through exposure, loss, prediction, and optimizer trajectories.
5. Pilot true last-layer gradient direction and outlier metrics; scale only if predictive.
6. Freeze the protocol, then open the blind holdout once.
