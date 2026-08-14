# Readiness fix 01: clean-checkout asset identity

- Owner: SCTSR v4 readiness-fix review.
- Source: failure-first and post-fix pytest runs in a clean worktree based on `f285754`.
- Consumer: training-readiness review and Appendix-D SA-040/SA-260/SA-261.
- Lifecycle: immutable rollback-unit evidence; later runs must use a new directory.
- Scope: OOF metadata byte identity only. No dataset row, OOF fold, group, label, or training behavior changes.

The red run must fail because the frozen registry still describes the stale CRLF
working-copy representation while Git checks out LF bytes. The green run must
execute the same test after the registry is corrected to the tracked bytes.
