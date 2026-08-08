# Stage1 GapValue 240-run pattern analysis v2

## Purpose

This command analyzes the frozen 240-run result set without changing training
outputs, the experiment contracts, the matrix, or any selection manifest.  It
tests whether training-dynamics-guided GapCritical replay improves the
operational normal/defect separation relative to the two registered controls
and the other frozen ranking methods.

The analysis is not a best-run search.  Its primary paired estimands are:

```text
delta_FN = FN_at_TN68253(T) - FN_at_TN68253(control)  # lower is safer
delta_TN = TN_at_FN95(T) - TN_at_FN95(control)        # higher is better
```

R1 and R2 are always reported separately.  Gap and tail-quantile changes are
mechanism evidence, not replacements for the two operational metrics.

## Read-only boundary

The following inputs are immutable evidence:

- the fully extracted machine uploads;
- the canonical validated-run inventory;
- the global completeness audit;
- the frozen experiment matrix;
- optional frozen selection manifests and sample-value table.

The analysis code reads those inputs but never writes inside them.  The output
directory must be outside `--extracted-root`, must not already exist, and is
created through an `.inprogress` staging directory before final publication.
Raw predictions, checkpoints, and the extracted result tree must not be copied
into Git.

Only attempts named by the canonical inventory are eligible.  Historical failed
attempts remain reliability evidence but cannot replace a canonical VALIDATED
attempt.

## Formal command

Run from the repository root:

```powershell
uv run python scripts/stage1_gapvalue240/analyze_results.py `
  --extracted-root C:\baidunetdiskdownload\stage1_gapvalue240_all_uploads_extracted `
  --inventory C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728\GLOBAL_VALIDATED_RUN_INVENTORY.csv `
  --completeness-audit C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728\GLOBAL_COMPLETENESS_AUDIT.json `
  --matrix artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\generated\frozen_experiment_matrix.csv `
  --selection-root artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\generated\selections `
  --value-table artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\frozen_inputs\reference_tables\sample_value_table.csv `
  --output-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_pattern_analysis_20260728_v2
```

Prediction recomputation is enabled by default and is required for the formal
report.  `--skip-prediction-recompute` exists only for fast development of
tables or report layout; an output created with it is not formal metric
recalculation evidence.

`--selection-root` and `--value-table` have repository defaults, but the formal
command supplies both paths explicitly.  The resolved selection directory and
value table must exist; the analyzer fails instead of guessing or silently
omitting the score-linked analyses.

The command prints a JSON completion summary.  A successful run returns exit
code `0`; validation or analysis failures propagate as a nonzero process exit.

## Outputs

The finalized output contains:

- `index.html` and `FINAL_REPORT_CN.md`;
- `analysis_contract.yaml`, `manifest.json`, and audit evidence;
- canonical run metrics;
- T-R1 and T-R2 triad deltas;
- raw-score T-control deltas and raw-versus-Platt operational sensitivity;
- condition, A02 discovery/confirmation/pooled, sensitivity, method, budget,
  and guard summaries;
- per-seed method/budget/guard delta-of-deltas;
- selection-value summaries, value/effect associations, and R2 overlap audits;
- 48,000 raw epoch rows, 32,000 paired T-control epoch differences, and
  descriptive paired-curve summaries;
- capability/provenance and four-layer hypothesis registries;
- metric-recomputation, prediction-tail, selection-overlap, execution
  reliability, and sensitivity tables;
- charts linked to the source tables.

The report distinguishes four different statements:

1. numerically better;
2. passed the frozen statistical contract;
3. supports the proposed training-dynamics mechanism;
4. supports a causal conclusion.

These labels must not be collapsed into a single success flag.

## Scientific boundaries

- The 240 runs are discovery and internal confirmation on `val_op`; they do not
  replace a blind or external test.
- All three-seed condition results are exploratory.
- A02 Phase C has treatment/control machine confounding.  Its five seeds and
  the preregistered pooled eight-seed arithmetic are reported, but Phase C
  cannot independently establish an unconfounded causal effect.
- R2 is a high-overlap, low-power near-treatment control.  R1 is the fully
  disjoint random baseline.  R2 overlap and effective unique contrast must be
  displayed beside its effect estimates.
- A sample-value ranking is an experiment-level intervention rule; the results
  do not establish a causal value for every individual image.
- No per-epoch `val_op` predictions were saved, so per-epoch TN, FN, or gap
  curves must not be inferred from training loss or accuracy.

## Acceptance checks

A formal analysis is acceptable only when:

1. exactly 240 canonical VALIDATED runs, 80 complete triads, and 160 separate
   T-control comparisons are present;
2. the matrix and canonical inventory agree one-to-one;
3. every formal `val_op` prediction file has the frozen row and sample-ID set;
4. recomputed tie-safe integer operational metrics exactly match saved metrics,
   floating quantiles match within the declared numerical tolerance, and all
   160 raw/calibrated paired integer effects are identical;
5. A02 is represented once each as discovery (3), confirmation (5), and pooled
   (8), separately for R1 and R2;
6. all 240 selection manifests match their frozen SHA and join to the frozen
   120,000-row sample-value table;
7. exactly 48,000 epoch records and 32,000 T-control paired epoch records are
   emitted; epochs remain descriptive rather than statistical replicates;
8. same-machine/cross-machine, resumed/non-resumed, snapshot, discovery, and
   confirmation sensitivity tables are emitted;
9. the finalized manifest covers every permanent report artifact;
10. source evidence remains unchanged and no output is written under the
   extracted root;
11. HTML links and chart files validate successfully.

Run the focused verification with:

```powershell
uv run --extra dev python -m pytest -q `
  tests/stage1_gapvalue240/test_deep_analysis_cli.py `
  tests/stage1_gapvalue240/test_deep_analysis_ingestion.py `
  tests/stage1_gapvalue240/test_deep_statistics.py `
  tests/stage1_gapvalue240/test_deep_mechanisms.py `
  tests/stage1_gapvalue240/test_deep_subgroups.py `
  tests/stage1_gapvalue240/test_deep_patterns.py `
  tests/stage1_gapvalue240/test_deep_pipeline.py `
  tests/stage1_gapvalue240/test_deep_reporting.py
```
