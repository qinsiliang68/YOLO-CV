# Documentation Scope

Current documentation:

| File | Status | Meaning |
| --- | --- | --- |
| `stage1_oof_10fold.md` | CURRENT_OOF_PLAN | How the current 10-fold OOF split is generated and trained. |
| `stage1_oof_200epoch_archives_20260621.md` | CURRENT_OOF_ARCHIVES | Which OOF folds have completed, where their archive indexes are, and how to export OOF predictions. |
| `stage1_sample_value_oof_dynamics_20260708.md` | CURRENT_SAMPLE_VALUE_RUNBOOK | Operational runbook and directory layout for the OOF dynamics sample-value experiment. |
| `stage1_oof_gap_value_experiment_design_20260708.md` | CURRENT_SAMPLE_VALUE_DESIGN | Human handoff design for OOF gap-value sample selection, metrics, replay matrix, and success/failure criteria. |
| `stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATOR_BRIEFING_V2.md` | CURRENT_OPERATOR_BRIEFING | Scientific background, frozen experiment matrix, output interpretation, monitoring rules, and failure boundaries for the current ten-machine campaign. |
| `stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md` | CURRENT_CAMPAIGN_OPERATIONS | Formal v2 release, assignment, single-job worker, lease, recovery, and ten-machine canary procedure. |
| `stage1_sctsr_v4/IMPLEMENTATION_GUIDE.md` | IMPLEMENTATION_ONLY_HELD | Isolated SCTSR v4 code, evidence, validation and formal-release boundaries. No formal SCTSR training has been authorized. |
| `stage1_sctsr_v4/KNOWN_BLOCKERS.md` | CURRENT_SCTSR_BLOCKERS | R2 feasibility, v3 regression, val_target, release/seed and scientific non-claim blockers. |

Top-level scope document:

| File | Status | Meaning |
| --- | --- | --- |
| `../CURRENT_RESEARCH.md` | CURRENT_SCOPE_MAP | Main entrypoint for deciding whether a file belongs to current research, supporting evidence, historical material, or local-only data. |

Historical documentation is kept under `_recycle_bin/` for audit only.
