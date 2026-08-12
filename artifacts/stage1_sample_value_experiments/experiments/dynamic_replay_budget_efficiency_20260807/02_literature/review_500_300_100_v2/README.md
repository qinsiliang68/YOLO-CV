# Stage1 Literature Evidence v2

Status: `INCOMPLETE_HELD`

This is the only counted evidence corpus for the new `500/300/100` review. The older
`LITERATURE_EVIDENCE_MATRIX.csv`, `FULL_TEXT_READING_LEDGER.csv`, and 50 notes remain
historical candidate material. They do not count here until their source bytes,
identity, reading depth, and per-paper record pass the v2 validator.

## Scientific Question

Under a fixed base training process and fixed cumulative replay exposure, which
samples are still learnable, point in a direction favorable to the independent FN95
target, are sufficiently reliable, and add non-redundant set coverage? Can a replay
policy based on those conditions beat global random, method-matched random,
current-loss, and no-replay across unseen training seeds?

The corpus may support or contradict candidate mechanisms. It cannot establish
Stage1 utility without a real paired replay intervention.

## Counted Tiers

- `BROAD`: exactly 500 canonical works, IDs `P0001` through `P0500`.
- `SCREENED`: exactly 300 of those 500, including all `DEEP` works.
- `DEEP`: exactly 100 of those 300 with local full-text PDF identity and page evidence.
- One counted work has exactly one note: `notes/Pxxxx.md`.
- Conference, journal, workshop, and arXiv versions of one study are merged into one
  canonical work and listed under `merged_versions`.

## Completion Gate

```powershell
uv run python scripts/stage1_dynamic_replay_v3/validate_literature_evidence_v2.py `
  --corpus-root artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/02_literature/review_500_300_100_v2
```

The command defaults to the full gate: corpus, discovery provenance, source
acquisition, deterministic manual audit, and 30-paper time-separated second pass.
`--corpus-only` and `--skip-pdf-page-inspection` are development diagnostics and
cannot authorize completion.

Current absence of 500 records is intentional and honest. No placeholder paper rows
are generated to make the count appear closer to completion.

## Canonical Repair Staging

The v4 repair has a mechanically de-duplicated 500-work BROAD staging freeze, a
300-work PRIMARY SCREENED queue, and 24 hash-preserving migrated SCREENED reviews.
These are not yet formal counted tiers. The inherited review content is accepted by
user instruction and explicitly marked `USER_ACCEPTED_INHERITED_EVIDENCE`; it is not
represented as an independent rereview. See `CANONICAL_REPAIR_v4.md` for exact
counts, provenance, and promotion boundaries.

## Source Staging Progress

Fifty high-priority legacy candidates now have locally acquired primary or official
repository PDFs. The first acquisition obtained 48 files; the two immutable failure
records remain in `discovery/LEGACY_50_SOURCE_FAILURES_v1.csv`. A separate correction
request then acquired `LEGACY-P025` from UCLA and `LEGACY-P036` from the PDF location
published by PMLR.

`validation/LEGACY_50_SOURCE_VALIDATION_v2.json` currently reports `50/50` source
files verified. Its inventory checks PDF signatures, byte counts, SHA-256, immutable
receipts, Poppler page counts, and title tokens from the first two pages against the
historical bibliographic ledger. This proves source identity only. It grants zero
`BROAD`, `SCREENED`, or `DEEP` reading credit until stable v2 paper IDs exist and each
paper's new note passes the corresponding reading-depth requirements.

## Directory Contract

- `CANONICAL_WORKS.csv`: one row per counted canonical work.
- `notes/`: independent per-paper Markdown with an embedded structured evidence block.
- `sources/`: local primary-source HTML/PDF bytes; ignored by Git, hash-bound by ledgers.
- `SOURCE_ACQUISITION.csv`: URL, retrieval time, authority, bytes, and SHA for every role.
- `discovery/QUERY_LOG.csv`: exact database queries and raw result snapshots.
- `discovery/CANDIDATE_LEDGER.csv`: every included/excluded candidate and reason.
- `validation/RANDOM_AUDIT.csv`: fixed nested 10%/15%/20% manual audit.
- `validation/SECOND_PASS_30.csv`: critical 30-paper re-read after at least 24 hours.
- `validation/COMPLETION_AUDIT.json`: machine-readable current truth.

Formal training, engineering gate generation, pilot release, assignments, and blind
holdout access remain forbidden while this corpus is `INCOMPLETE`.

## Literature Scope Decision

On `2026-08-10`, the user declared the assembled literature sufficient and stopped
further paper discovery and reading. The decision and its non-claim boundary are
recorded in `USER_LITERATURE_SUFFICIENCY_DECISION_20260810.md`. It authorizes moving
to synthesis and construction specifications; it does not relabel the incomplete
formal corpus audit as `PASS` or authorize training, gates, assignments, or blind
data access.
