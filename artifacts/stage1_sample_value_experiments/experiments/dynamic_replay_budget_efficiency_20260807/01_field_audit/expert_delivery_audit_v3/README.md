# Expert Delivery Audit

- Status: `INCOMPLETE_SOURCE_MISSING`
- Expected artifact identities: 17
- Inventory rows: 19
- Archive member rows: 1863
- Required source artifacts missing: 3
- Hash or archive failures: 0
- Delivered code executed: no
- Temporary extraction retained: no

## Missing Required Source Artifacts

- `Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz`
- `Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip`
- `stage1_budgeted_replay-1.0.0-py3-none-any.whl`

A report or an archived code excerpt does not satisfy a missing source archive. The three missing BudgetedReplay source carriers keep source-level comparison blocked.

## Reproduce

```powershell
uv run python scripts/stage1_dynamic_replay_v3/audit_expert_deliveries.py `
  --downloads-dir C:\Users\28898\Downloads `
  --output-dir <NEW_VERSIONED_AUDIT_DIRECTORY>
```

Outputs are immutable by default. Use a new versioned directory for a later rerun.
