# Expert Delivery Audit Index

## Scope

This audit inventories external expert deliveries under
`C:\Users\28898\Downloads`. It hashes and safely extracts archives without
executing delivered code. External archives remain source truth in Downloads;
the registered experiment stores only manifests, receipts, and review evidence.

## Versions

| Version | Status | Meaning |
|---|---|---|
| `expert_delivery_audit_v1/` | `FAIL` | Preserved first attempt. Windows rejected a long archive path during extraction. |
| `expert_delivery_audit_v2/` | `INCOMPLETE_SOURCE_MISSING` | Long-path extraction passed, but this transitional run predates the output manifest and README contract. |
| `expert_delivery_audit_v3/` | `INCOMPLETE_SOURCE_MISSING` | Current canonical audit. Full extraction, internal manifest verification, cleanup, and output hashes passed. |

Do not delete or rewrite prior versions. A later rerun must use a new versioned
directory because each evidence set is immutable by default.

## Current Verified Facts

- The first expert return archive SHA-256 matches its sidecar.
- Its 1,842 manifested payload files and 228,559,378 uncompressed payload bytes
  all match the internal manifest.
- The independent review ZIP has 19 regular members, valid CRC, and five
  ledger-bound member hashes; no mismatch was found.
- The canonical member ledger contains 1,863 rows across both archives.
- No delivered Python or training command was executed by this audit.
- Temporary extraction completed and was removed.

The following source carriers named by the release ledger are absent:

```text
Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz
Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip
stage1_budgeted_replay-1.0.0-py3-none-any.whl
```

Therefore the expert report and review evidence can be audited, but a complete
source-level comparison of the BudgetedReplay v1.0.0 repository cannot yet be
marked complete.

## Canonical Entry Points

- Human summary: `expert_delivery_audit_v3/README.md`
- Expected/observed inventory: `expert_delivery_audit_v3/expert_v1_inventory.csv`
- Archive members: `expert_delivery_audit_v3/expert_archive_member_manifest.csv`
- Machine receipt: `expert_delivery_audit_v3/expert_v1_hash_validation.json`
- Output hashes: `expert_delivery_audit_v3/expert_audit_output_manifest.csv`

## Reproduce

```powershell
uv run python scripts/stage1_dynamic_replay_v3/audit_expert_deliveries.py `
  --downloads-dir C:\Users\28898\Downloads `
  --output-dir artifacts\stage1_sample_value_experiments\experiments\dynamic_replay_budget_efficiency_20260807\01_field_audit\expert_delivery_audit_v4
```
