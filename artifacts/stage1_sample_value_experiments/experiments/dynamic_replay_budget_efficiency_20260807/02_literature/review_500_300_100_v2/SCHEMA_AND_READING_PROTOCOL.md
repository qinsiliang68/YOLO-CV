# Schema And Reading Protocol

## Canonical Identity

`CANONICAL_WORKS.csv` is ordered exactly `P0001..P0500`. DOI, arXiv ID,
OpenReview ID, normalized title, primary URL, and `canonical_work_id` are deduplicated.
Versions of the same research work are merged, not counted twice. Each registry row
must agree byte-for-byte with the identity block in its Markdown note.

## Broad 500

Before a work can enter `BROAD`, read and record its title, abstract, research
problem, method overview, and conclusion from a primary source. The note must contain
an independent Chinese summary and critical mini-review, a direct relevance chain,
supported/refuted judgment, transferable mechanism, unsupported inference, and
Stage1 boundary. A local official page or full-text artifact is required with bytes
and SHA-256.

## Screened 300

Before a work can enter `SCREENED`, inspect methods, experiments, ablations, and
limitations. Record equations or the exact allowed missing marker, algorithm steps,
variables, selection timing, refresh rule, budget unit and denominator, unique/repeat
and cumulative exposure semantics, compute cost, random controls, data, model, seed
count, checkpoint rule, locator-bound results, negative results, failure conditions,
and transfer class.

Transfer classes are:

- `REPLICATION`: the original method and estimand are genuinely reproduced.
- `INSPIRED_ADAPTATION`: a declared task-specific modification is proposed.
- `MECHANISM_ONLY`: the paper motivates a falsifiable mechanism arm only.
- `NOT_TRANSFERABLE`: evidence is relevant but the method should not be moved to Stage1.

## Deep 100

Before a work can enter `DEEP`, retain the complete primary PDF, verify bytes,
SHA-256, PDF signature, and page count, then read every substantive section. Record
section-page coverage, at least three page-level evidence anchors, formula assumptions,
algorithm complexity/randomness, data roles and leakage risk, fair-budget semantics,
seed variation and worst case, key ablations and limitations, Stage1 fields/interfaces/
cost/code mapping, and an explicit counter-check against overclaiming.

## Missing Information

Only these machine-readable forms are accepted:

- `NOT_ASSESSED_AT_BROAD_LEVEL`
- `NOT_REPORTED_BY_PAPER`
- `NOT_APPLICABLE_WITH_REASON:<specific reason>`
- `SOURCE_UNAVAILABLE_EXCLUDED` only in the excluded-candidate ledger, never in the
  counted 500.

Empty values, `TODO`, `TBD`, `unknown`, `待补`, `待确认`, and `同上` fail validation.
Exact reuse of per-paper summary/review/relevance/boundary prose also fails.

## Anti-Fabrication Checks

- Exact 500/300/100 counts and strict nesting.
- Exact one-note-per-work mapping.
- Canonical identity and version deduplication.
- Primary-source acquisition URL, authority, retrieval time, bytes, and SHA.
- Deep PDF page-count and anchor bounds.
- Exact query snapshots and exclusion reasons.
- Fixed-seed nested manual audit: 50 broad, 45 screened, and 20 deep records.
- At least 30 critical deep works re-read after 24 hours against the same PDF hash.
- All eight research questions covered by the critical second-pass set.

Long files or large word counts are not evidence of reading. A paper only counts at
the highest tier whose concrete contract passes.
