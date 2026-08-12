# Stage1 finite-budget dynamic replay preregistration v3

This directory is the construction contract for the evidence-driven Stage1 study. Its
scientific state is `PREREGISTERED_NOT_RUN`; candidate effectiveness is
`NOT_EVALUATED`. A PASS from `PREREGISTRATION_VALIDATION.json` proves only that these
files are internally complete and falsifiable. It does not prove that Q/R/A/D works and
does not authorize formal training, an engineering gate, assignments, a pilot release,
or blind holdout/test access.

Authoritative artifacts:

- `SCIENTIFIC_CONTRACT.json` and `SCIENTIFIC_CONTRACT.md`: concepts, data roles,
  evidence taxonomy, causality, and fairness invariants.
- `MINIMAL_FALSIFIABLE_MATRIX.csv` and `CONTRAST_REGISTRY.csv`: eight arm families and
  discovery/confirmation comparisons at the frozen 600-slot, 119,400-exposure budget.
- `MECHANISM_EVIDENCE_REGISTRY.csv`: pre-intervention mechanism states, all honestly
  recorded as `UNKNOWN_IN_STAGE1`.
- `DATA_COLLECTION_SCHEMA.json` and `DATA_COLLECTION_SCHEMA.md`: identity-bound signal,
  selection, exposure, prediction, closeout, and paired endpoint records.
- `STATISTICAL_DECISION_RULES.json` and `STATISTICAL_DECISION_RULES.md`: independent
  seed sets, fixed endpoint, multiplicity, stability, and stopping rules.
- `REPOSITORY_CHANGE_SPEC.md`: modules, CLIs, tests, outputs, migration, and rollback.

Validation command:

```powershell
uv run python scripts/stage1_dynamic_replay_v3/validate_preregistration_v3.py
```

The command reads contracts and writes a machine audit only. It executes no experiment.
