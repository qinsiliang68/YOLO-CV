# Readiness fix 02: epoch publication recovery

- Owner: SCTSR v4 readiness-fix review.
- Source: failure-first and post-fix transaction/recovery tests in a clean worktree.
- Consumer: Appendix-D SA-220 through SA-231 and formal-resume review.
- Lifecycle: immutable rollback-unit evidence; later runs use a new directory.
- Scope: durable epoch publication and crash recovery only.

The tests distinguish the atomic receipt commit point from secondary metadata.
A failure before the receipt must quarantine the renamed generation. A failure
after the receipt must preserve the append-only evidence and deterministically
rebuild the artifact index and rolling pointer before resume is allowed.
