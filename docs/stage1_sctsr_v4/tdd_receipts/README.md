# SCTSR v4 TDD receipt registry

## Governance

- Owner: SCTSR v4 implementation evidence custodian.
- Source: per-rollback-unit raw test outputs; `commit_08*` directories were already present in the working tree before the Codex completion pass.
- Consumer: Appendix-D SA-260 through SA-275 review and independent reproduction.
- Lifecycle: immutable historical evidence. New runs create a new named directory; they do not overwrite prior red/green logs.
- Verification: every committed file is covered by the SCTSR source-tree manifest with path, bytes and SHA-256. Final validation logs are additionally indexed under the registered experiment `08_reports` directory.

## Status by inherited directory

- `commit_01` through `commit_07`: inherited rollback receipts containing the recorded red/green artifacts described by each `RECEIPT.json` or README.
- `commit_08`: inherited five-test red evidence and 31-test green evidence for semantic run validation.
- `commit_08_cli_hardening`: inherited red-only evidence. It is not, by itself, a red-green completion claim; current green reproduction must be supplied separately.
- `commit_08_formal_bindings`: inherited red-only evidence. It is not, by itself, a red-green completion claim; current green reproduction must be supplied separately.
- `commit_09_resume`: inherited recovery red/green evidence.

The final self-audit must not infer PASS merely because these files exist. It must bind each relevant current command, exit code and unmodified stdout/stderr hash, and must retain the v3 regression mismatch as FAIL until its specification change is resolved.
