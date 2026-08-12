# Canonical literature repair v4

Status: `STAGING_PASS_FORMAL_NOT_PUBLISHED`

This repair preserves the prior literature analysis as user-accepted inherited
evidence. It performs only mechanical identity, hash, tier-membership, and stable-ID
work. It is not an independent rereview of the inherited paper content and does not
provide replay-utility evidence.

## Provenance rule

- Inherited reviews are classified as `USER_ACCEPTED_INHERITED_EVIDENCE`.
- `content_review_reused=true` and
  `independent_content_rereview_performed=false` are mandatory in migrated records.
- These records may inform mechanism synthesis and experiment design. They cannot be
  cited as a new independent review or as evidence that a Stage1 replay method works.

## Canonical identity repair

`discovery/CANONICAL_MERGES_v4.csv` contains five explicit merges: one pre-existing
exact-metadata merge and four version-identity merges. Each version merge requires a
registered arXiv/DOI identity and an identical normalized author set. The builder
hash-binds this exact merge ledger in `BUILD_INPUT_MANIFEST.csv`.

The repaired BROAD staging freeze reuses the v3 frozen seed and policy. It removes
four duplicate-version rows and fills the four vacancies deterministically from the
existing reserve pool; no paper was hand-picked for replacement.

## Current staging counts

- `staging/broad_freeze_canonical_repaired_v4`: 500 selected canonical works and 35
  reserves; all 500 selected source SHA-256 values are unique.
- `staging/screened_queue_canonical_repaired_pdf_v4`: 300 PRIMARY, 30 reserves, and
  20 reserve-read works, for a 320-paper reading queue.
- `staging/screened_text_canonical_repaired_v4`: 320/320 local text extractions.
- `discovery/screened_fulltext_reviews_v4`: 24 inherited SCREENED records migrated
  by canonical work ID and byte-identical PDF/text evidence, plus two current
  full-text reviews; 10 inherited paper IDs changed.
- `validation/SCREENED_FULLTEXT_REVIEW_VALIDATION_v4.json`: `PASS` for all 26
  partial reviews, with `formal_screened_increment=0`.
- `staging/deep_review_queue_v4`: 100 PRIMARY and 200 reserve DEEP review
  candidates. The fixed-seed queue requires at least 10 works per RQ, contains all
  40 core anchors, and marks 31 works with existing hash-bound evidence;
  69 PRIMARY works still require full review. Its `formal_deep_increment` is zero.

After the DEEP queue freeze was added, the complete literature test set passed `135`
tests (`94` unrelated v3 tests deselected).

## Formal boundary

This directory's formal `CANONICAL_WORKS.csv` is still empty, so the counted corpus
remains `0/0/0`, not `500/300/100`. Promotion is prohibited until a fail-closed
publisher atomically binds the repaired staging files, creates exactly one
`Pxxxx.md` per formal work, and the full completion audit verifies nesting,
required fields, source identity, sampling audits, and second-pass evidence.

No formal training, engineering gate, pilot release, assignment generation, or
blind holdout/test access is authorized by this repair.
