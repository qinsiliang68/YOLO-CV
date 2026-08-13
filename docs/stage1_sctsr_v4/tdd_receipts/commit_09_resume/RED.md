# Commit 09 resume TDD red receipt

- Command: `uv run pytest tests/stage1_sctsr_v4/test_formal_resume.py -q`
- Exit code: `1`
- Observed at: `2026-08-13` in `C:\GitHub\YOLO-CV`
- Failure-first condition: test collection raised `ImportError` because
  `stage1_sctsr_v4.recovery.prepare_formal_resume_context` did not exist.
- Required behavior introduced by the red test: revalidate a contiguous formal
  epoch prefix, reconstruct replay history from immutable occurrence Parquet,
  bind the checkpoint RNG and generation/receipt/index chains, quarantine an
  uncommitted next epoch, and continue the branch without overwriting a complete
  generation.

The adjacent green command and its exact exit code are preserved in
`GREEN.txt` and `GREEN.exitcode.txt`.
