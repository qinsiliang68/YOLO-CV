# Readiness fix 03: formal execution token and claim

- Owner: SCTSR v4 readiness-fix review.
- Source: failure-first and post-fix authorization tests in a clean worktree.
- Consumer: formal training release review and multi-machine deployment preflight.
- Lifecycle: immutable rollback-unit evidence; later runs use a new directory.
- Scope: matrix-release versus one-process execution authorization only.

No token, release, seed, assignment, gate, claim registry, or formal run created
by these tests is a scientific or production artifact. All authorization
fixtures use an in-test key and temporary directories.

## Failure-first evidence

- `RED_IMPORT.junit.xml`: the first test collection failed because
  `stage1_sctsr_v4.formal_execution` did not exist.
- `RED_BEHAVIOR.junit.xml`: after adding an empty scaffold, 11 tests failed
  because job-bound verification, exclusive claim and CLI enforcement were
  not implemented.
- `RED_PUBLIC_SCHEMAS.junit.xml`: 2 tests failed because the four public
  schemas and inactive templates were absent.
- `RED_REGISTRY_SNAPSHOT.junit.xml`: the claim-registry descriptor was not
  copied into the immutable run attempt snapshot.
- `RED_CLOSEOUT_EXECUTION_EVIDENCE.junit.xml`: closeout had no registered
  validator for the in-run token/claim/registry snapshot.

## Green evidence

- `GREEN_TARGETED.junit.xml`: 25 passed for token, CLI and schema behavior.
- `GREEN_FORMAL_ADJACENT.junit.xml`: 19 passed for formal execution and resume
  integration after updating the formal-resume unit fixture to provide a
  claim.
- `GREEN_REGISTRY_SNAPSHOT.junit.xml`: registry/token/claim byte snapshot and
  tamper rejection passed.
- `GREEN_COMPLETE_UNIT.junit.xml`: 34 passed for the completed rollback unit.
- `GREEN_CLOSEOUT_EXECUTION_EVIDENCE.junit.xml`: 2 passed for immutable
  snapshot validation and fail-closed formal closeout.
- `GREEN_POST_REVIEW_HARDENING.junit.xml`: 29 passed after byte-binding the
  token before its atomic claim and verifying snapshot-copy bytes.
- `GREEN_COMPLETE_UNIT_V2.junit.xml`: 33 passed across execution, closeout,
  CLI, resume and schema tests after the second review pass.
- `GREEN_V4_PY311.junit.xml`: the complete v4 suite passed with 359 tests on
  isolated CPython 3.11.

The intermediate `ADJACENT_FORMAL.junit.xml` records the expected one-test
failure of the pre-existing formal-resume fixture after the new fail-closed
claim requirement became active. It is retained rather than overwritten.

All claim-registry concurrency in this rollback unit is local engineering
evidence. A real ten-machine release still requires a cross-host exclusive
create probe on the exact shared filesystem named by the future production
registry.
