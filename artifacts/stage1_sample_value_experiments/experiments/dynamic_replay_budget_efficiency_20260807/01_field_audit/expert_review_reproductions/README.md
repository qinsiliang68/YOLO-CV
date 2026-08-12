# Expert Review Reproductions

This directory separates the expert deliveries from local current-v3 evidence:

- `source_evidence_v1/`: immutable files extracted from the expert review ZIP.
  `SOURCE_MEMBER_MANIFEST.csv` records their archive identity and hashes.
- `local_v3_baseline_20260809/`: captured pytest results from the current local
  worktree. The full suite fails collection because the untracked recovery test
  imports a module that does not yet exist; the remaining suite has 62 passes.
- `local_v3_p0_20260809/`: current-v3 minimal reproductions. These execute only
  local v3 code against synthetic fixtures and are engineering evidence, not a
  scientific result.
- `src_dynamic_review_v1/`: byte-identical copy of the 2026-08-08 DynamicReplay
  review and its source manifest. The review SHA-256 is
  `DACC13945F54B563F08E042F2BFB42E54E284C6A1258BE988AE5E6C6539E6940`.
- `src_dynamic_code_v1/`: 44-file source subset from the fully verified expert
  DynamicReplay TAR. The complete 1,842-row package manifest was validated
  before extraction; `SOURCE_SELECTION_MANIFEST.csv` binds every selected file.
- `local_dynamic_review_validation_20260809/`: 32 current-v3 negative-contract
  tests and a hash-bound validation receipt. No delivered expert code was run.

The two current comparison ledgers have different scopes:

- `expert_vs_v3_full_matrix_v1.csv`: all 31 BudgetedReplay findings
  (`7 P0 + 16 High + 8 Moderate`). Current-v3 assessment counts are 6 present,
  10 partially mitigated, 8 absent, and 7 not applicable because the reviewed
  capability is absent.
- `dynamic_review_vs_v3_matrix_v1.csv`: all 15 DynamicReplay review findings.
  Current-v3 assessment counts are 3 present, 2 partially mitigated, 8 absent,
  and 2 not applicable because the reviewed capability is absent.

The expert BudgetedReplay source TAR/ZIP/Wheel is absent. Therefore the supplied
expert reproduction scripts are retained but not rewritten or reported as a
fresh independent rerun. Four scripts import the missing source through a
hard-coded `/mnt/data/...` path.

No blind holdout, formal training data, model checkpoint, or delivered training
entrypoint is used here.

Formal training remains blocked. In particular, current v3 still lacks the
fleet AIOps event materializer, a portable formal assignment builder, a clean
timing-only estimand, and the target-direction gradient path. The 236-run
RHO-only matrix remains `HELD` and is scientifically superseded.

## Tripartite Crosswalk Contract

The v1 matrices remain preserved as the pre-migration evidence. The authoritative
strict matrices are now `expert_vs_v3_tripartite_v2.csv` (31 BudgetedReplay
findings) and `dynamic_review_vs_v3_tripartite_v2.csv` (15 DynamicReplay
findings). `TRIPARTITE_CROSSWALK_VALIDATION_v2.json` is their generated,
fail-closed schema and line-reference check. The required CSV fields, in exact
order, are:

```text
requirement_id,overall_status,expert_source_status,expert_claim_refs,
expert_source_refs,v3_status,v3_source_refs,reproduction_command,exit_code,
result_artifact_sha,observed_result,remaining_risk,required_action
```

Only `CONFIRMED_PRESENT`, `CONFIRMED_ABSENT`, `PARTIALLY_MITIGATED`,
`CONTRADICTED_BY_EVIDENCE`, `NOT_TESTABLE_SOURCE_MISSING`, and
`NOT_APPLICABLE` are accepted status values. Because the BudgetedReplay source
carriers are missing, each BudgetedReplay row must use
`expert_source_status=NOT_TESTABLE_SOURCE_MISSING` and
`expert_source_refs=NOT_APPLICABLE_SOURCE_MISSING`. Reports and excerpts do not
substitute for source lines.

The validator checks recorded commands but never executes them. The separate
`build_tripartite_crosswalk_v2.py` builder executes every row before publication
and writes a dedicated deterministic JSON result under
`tripartite_reproduction_v2/results/`; `EXECUTION_RECEIPT.json` binds all 46
commands, exit codes, paths, and SHA-256 values. A command must
be one exact `uv run pytest ...::test_name -q` node or one exact read-only
`rg -n` query. Every row also needs an integer exit code and a 64-hex result-log
SHA-256.

These v2 reproductions are deliberately limited to static source-line presence.
They do not execute formal training, do not prove runtime behavior, and are not
scientific utility evidence. The BudgetedReplay source-side conclusion remains
`NOT_TESTABLE_SOURCE_MISSING` for all 31 rows.

Run the current audit with:

```powershell
uv run python scripts/stage1_dynamic_replay_v3/validate_tripartite_crosswalk.py
```

The migration has a preserved failure-first receipt at
`tripartite_reproduction_v2/TDD_RED.txt`; the passing test receipt is
`tripartite_reproduction_v2/TDD_GREEN.txt`. Rebuilding v2 is explicit:

```powershell
uv run python scripts/stage1_dynamic_replay_v3/build_tripartite_crosswalk_v2.py
```
