# Readiness fix 06: failure-preserving epoch cleanup

## Finding

The epoch exception handler called `recorder.abort()` before
`transaction.abort()`. If telemetry shutdown or either open Parquet writer
raised during cleanup, transaction quarantine was skipped and the cleanup
exception replaced the original training/evidence failure. Recorder
construction failure also occurred before the old `try` boundary.

## Failure-first evidence

- `RED_BEHAVIOR.junit.xml`
- bytes: 1,228
- SHA-256: `5856DE60F7F90B53ADAF99348EED5A2C9447E294D09F804D646DBE40B5993E6C`
- result: collection failed because `_abort_failed_epoch` did not exist.

A second failure-first assertion then demonstrated that telemetry stop failure
prevented both writer abort calls in the old recorder implementation.

## Fix

- The recorder is created inside the epoch transaction `try` boundary.
- Recorder construction cleans up every component already opened before it
  re-raises its original failure.
- Recorder abort attempts telemetry and both Parquet writers even when one
  cleanup action fails.
- Epoch cleanup always attempts transaction quarantine after recorder cleanup.
- Cleanup failures are attached as Python exception notes; the original
  training/evidence exception remains the primary raised error.

## Green evidence

- `GREEN_BEHAVIOR.junit.xml`
- bytes: 9,312
- SHA-256: `5BD6F7CBB2A6F4B2712A231E020F9935A1EBC476FE65751B3944BED106B213A2`
- result: 52 passed across the new cleanup tests and adjacent evidence,
  transaction and formal-resume suites.

No formal training, release, seed, assignment, engineering gate, pilot release,
blind/test access or method-effectiveness claim was generated.
