# Readiness fix 05: byte-level dataset identity

## Finding

The prepared dataset was previously aligned to the frozen identity manifest by
filename and binary label. A different valid image with the same filename and
label could therefore enter training without changing the manifest or asset
registry. A successful process exit would not prove that the intended image
bytes were used.

## Failure-first evidence

- `RED_BEHAVIOR.junit.xml`
- bytes: 1,099
- SHA-256: `D18087E350D50106E625CE406628A75E13D565FAE7B8D0BFAA236A5315D70FBF`
- observed failure: collection failed because the required
  `stage1_sctsr_v4.dataset_content_ledger` implementation did not exist.

The regression fixture deliberately replaces a valid PNG with another valid
PNG at the same path while preserving filename, label and dimensions. The
formal content validator must return `DATASET_CONTENT_MISMATCH`.

## Fix

- Every registered non-test image receives a canonical relative identity,
  source-manifest SHA, binary label, byte count, SHA-256, width, height, mode
  and format in immutable Zstd Parquet.
- Formal trainer construction rehashes all registered physical images before
  importing or constructing the Ultralytics adapter.
- The resulting `dataset_content_binding` is stable across fresh/resume setup
  and is revalidated at closeout.
- Test/blind identities are forbidden from the ledger.
- No directory glob, similar filename, `latest` path or source-path fallback is
  allowed.

## Frozen full-data evidence

- rows: 384,000
- physical image bytes: 82,637,967,451
- ledger bytes: 19,350,859
- ledger SHA-256:
  `B2B61509AB4451C881FE7E9D0AAFB3F9D3CC0981A78AB9337C54C320E3E96D2C`
- content identity digest:
  `EDA93977CE43E946D4C795A8FBA30BF39B6AF510034276E739BC51D88DB1DD6E`
- full physical validation: `PASS`
- full validation receipt:
  `FULL_PHYSICAL_VALIDATION.json`, 3,403 bytes,
  SHA-256 `9DF8C3B8D818231634209045C8B8CE0B144F485DCACCB1DE718BAC0AE3ED450A`
- `test_accessed=false`
- `blind_holdout_opened=false`
- `formal_training_started=false`

## Green evidence

- `GREEN_BEHAVIOR.junit.xml`
- bytes: 3,397
- SHA-256: `CBC17F4C4D4742144470543C5C397C290D83D0267FE49FC65084EF2CA8E5E209`
- result: 20 passed, covering the new content tests and adjacent schema,
  formal-resume and immutable-input behavior.

This is an engineering/data-identity gate. It is not a scientific result and
does not claim that SCTSR, T, or any selector is effective.
