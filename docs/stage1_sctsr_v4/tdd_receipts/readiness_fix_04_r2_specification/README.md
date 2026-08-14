# Readiness fix 04: R2 specification audit

- Owner: SCTSR v4 readiness review.
- Scope: read-only frozen-asset feasibility and candidate-specification comparison.
- Formal side effects: none.
- Scientific result: none.

Failure-first evidence:

- `RED_IMPORT_AND_SCHEMA.junit.xml`: audit module and public schema absent.
- `RED_CLI.junit.xml`: audit CLI absent.
- `GREEN_AUDIT_CORE.junit.xml`: retained intermediate expectation mismatch;
  the canonical explicit hash token produced group TV `0.392333...`, not the
  exploratory tuple-repr value `0.387`.

Green evidence:

- `GREEN_AUDIT_CORE_V2.junit.xml`: 3 passed, including the real 120,000-row
  shortage and minimum-displacement comparison.
- `GREEN_AUDIT_AND_CLI.junit.xml`: 3 passed, including byte-bound machine report
  generation with no identity-pool directory.
- `GREEN_ADJACENT.junit.xml`: 19 passed across the audit, current strict
  fail-closed matcher, random controls, schema registry and documentation
  contracts.

The proposed matcher remains audit-only. The formal strict matcher still
fails closed until an owner preregistration decision is recorded.
