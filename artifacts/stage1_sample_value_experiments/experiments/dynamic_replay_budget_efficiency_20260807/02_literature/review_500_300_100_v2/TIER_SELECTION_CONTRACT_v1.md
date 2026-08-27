# Stage1 500/300/100 Tier Selection Contract v1

Status: `PREREGISTERED_NOT_YET_FROZEN`

This contract decides which canonical works may enter `BROAD_500`,
`SCREENED_300`, and `DEEP_100`. It does not decide whether any sample-value
mechanism works in Stage1. Literature evidence can nominate and constrain a
mechanism; only the paired replay intervention can establish utility.

## 1. Candidate Universe

The current OpenAlex discovery contains 5,171 deduplicated candidate records.
The manual queue also contains 108 legacy-only studies that are not represented
by those OpenAlex records, plus manually discovered primary studies needed for
under-covered questions. Therefore `CANDIDATE_INVENTORY_OPENALEX_v1.csv` is a
discovery component, not the final candidate universe.

Before screening decisions are counted, the repository must publish a new
versioned universe with:

- every OpenAlex v1 candidate;
- every legacy-only candidate;
- every targeted-search candidate and its query provenance;
- one row per candidate version and an explicit candidate-version group;
- a manifest of all input files and their SHA-256 values;
- `candidate_universe_id = SHA256(canonical manifest bytes)`.

No paper may be inserted after the universe is frozen. New discoveries require
a new universe version and a new decision manifest. OpenAlex's result window is
a reproducible search universe, not a claim to cover all world literature.

## 2. Research Questions

Each eligible canonical work has exactly one `quota_rq`. Other directly supported
questions are stored in `secondary_rqs` and do not fill more than one quota.
Definitions are fixed by `RESEARCH_QUESTIONS.md`:

- `RQ1`: training dynamics and learnable/already-learned/unlearnable states;
- `RQ2`: reducible or residual learnability beyond current loss/confidence;
- `RQ3`: gradient direction, target alignment, influence, or harmful updates;
- `RQ4`: reliability, label noise, outliers, and hard-clean separation;
- `RQ5`: diversity, redundancy, set utility, and coverage;
- `RQ6`: replay timing, frequency, repeat intensity, and cumulative exposure;
- `RQ7`: random controls, seed/order/state dependence, and value reversals;
- `RQ8`: Neyman-Pearson, partial-AUC, and FN95-local objectives.

## 3. Hard Eligibility Gate

A work is eligible only after a reviewer checks the primary title, abstract,
problem, method overview, and conclusion. It must directly support or contradict
at least one RQ in a machine-learning training context. Keyword overlap alone is
insufficient.

Eligible works receive exactly one directness class:

1. `D1_DIRECT_UTILITY`: selects, replays, removes, or reweights training samples
   under a finite budget and measures the trained model afterward.
2. `D2_DIRECT_MECHANISM`: directly studies Q/R/A/D, training dynamics, timing,
   repetition, or optimizer-visible exposure.
3. `D3_INFERENCE_CONTROL`: supplies random-control, seed, checkpoint, statistical,
   Neyman-Pearson, partial-AUC, or FN95-local inference needed by the study.
4. `D4_TRANSFER_ANALOG`: studies a different domain but preserves the training
   sample as intervention unit and a model-update mechanism that can be mapped.

Surveys may guide discovery but do not displace an available primary study.
Papers about human learning, physical dynamics, test-set difficulty only, generic
optimization without sample identity, or application performance without a sample
utility mechanism are excluded with an explicit reason.

## 4. Version Identity

Normalized titles create candidate version groups; they do not prove identity.
Two versions are merged only when DOI/official version links establish identity,
or when authors, method, data, and experiments jointly show that they are the same
study. Same-title works with different authors are not merged. A journal extension
that adds a principal method, estimand, or independent experimental conclusion is
a separate canonical work linked through `related_work_id`.

Unresolved identity is `IDENTITY_UNRESOLVED` and cannot enter a counted tier.
Merged works retain all DOI, arXiv, OpenReview, venue, year, and result differences.

## 5. Coverage Constraints

Only the single `quota_rq` counts toward these bounds:

| Tier | Exact total | Minimum per RQ | Maximum per RQ | Maximum D4 transfer works |
|---|---:|---:|---:|---:|
| BROAD | 500 | 40 | 100 | 100 |
| SCREENED | 300 | 25 | 60 | 60 |
| DEEP | 100 | 10 | 20 | 20 |

Failure to meet a minimum keeps the tier `HELD`; the threshold cannot be relaxed
to make a count pass. The remedy is targeted discovery or re-review of candidates
whose title-only RQ mapping was incomplete.

## 6. Effect Direction and Counterevidence

Every eligible work is classified as `SUPPORTED`, `NULL_NEGATIVE`, `MIXED`, or
`METHOD_ONLY` for its claimed mechanism. Null and adverse findings receive the
same directness priority as favorable findings. A paper cannot be excluded because
it contradicts Q/R/A/D.

For every RQ, `SCREENED` must contain at least two and `DEEP` at least one
`NULL_NEGATIVE` or `MIXED` work. If none is eligible after targeted search, the
corpus records `NO_ELIGIBLE_COUNTEREVIDENCE_FOUND` with its query evidence.

## 7. Selection Algorithm Without a Weighted Score

Selection is deterministic and lexicographic:

1. Meet every RQ minimum using the directness order `D1`, `D2`, `D3`, `D4`.
2. For `SCREENED` and `DEEP`, order design evidence as strict randomized
   intervention, partial randomized intervention, non-random intervention, then
   observational/theoretical evidence.
3. Preserve null, negative, and mixed evidence under the counterevidence rule.
4. Fill remaining places globally in the same directness/design order while
   respecting RQ and transfer maxima.
5. Resolve a genuinely equal stratum with
   `SHA256(canonical_work_id | tier | frozen_seed)`.

The following fields are forbidden in the selection key:

- citation count or OpenAlex relevance;
- publication year;
- queue ID or current row order;
- old reading depth;
- venue prestige;
- open-access status or ease of downloading a PDF.

The formal manual-review order is also a frozen hash order, so the reviewer does
not see a legacy-first or newest-first sequence.

## 8. Source Availability

Relevance order is frozen before source download availability is considered.

- `BROAD`: hash-bound official abstract or primary landing page.
- `SCREENED`: primary text sufficient to inspect methods, experiments, ablations,
  and limitations, including supplements when those sections live there.
- `DEEP`: complete primary PDF, byte count, SHA-256, page count, section coverage,
  and page-level anchors required by `SCHEMA_AND_READING_PROTOCOL.md`.

A source failure marks `SOURCE_BLOCKED` and triggers a reserve from the same
`quota_rq × directness × effect_relation` stratum. It does not retroactively make
an easier-to-download work more relevant.

## 9. Required Machine Tests

The implementation must prove:

- every frozen-universe candidate has exactly one final decision;
- counts are exactly 500/300/100 and tiers are strictly nested;
- shuffling input rows leaves selection unchanged;
- changing citation count, year, relevance, legacy depth, or open-access state
  leaves selection unchanged;
- each work fills only one quota and all minima/maxima hold;
- versions of one work count once;
- null or negative direction cannot lower priority;
- source failure only invokes a same-stratum reserve;
- every inclusion and exclusion links to a rule, primary evidence locator, reviewer,
  decision time, and decision hash.

Until those tests and all manual decisions pass, the counted corpus remains
`0/0/0` and no formal experiment release is authorized.
